//! The scroll policy: decides, from the viewport's movement over time, whether the user is
//! looking at rows (fetch them) or flying past them (fetch nothing).
//!
//! Pure logic over the caller's clock, so it can be tested with synthetic times. The speed
//! is the distance moved over the last `window_seconds` (measured from the newest sample at
//! or before the window's start, and never over more than the window), in viewport
//! heights per second. An update that moved is *fast* when that speed exceeds the threshold
//! or when it jumped more than `jump_viewports` (a scrollbar drag); an update that did not
//! move never is. The view is *settled* once no fast update has been seen for
//! `settle_seconds`.

use std::collections::VecDeque;

/// Tuning of the scroll policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolicyConfig {
    /// Speed above which movement is fast, in viewport heights per second.
    pub fast_viewports_per_second: f64,
    /// A single update moving more than this many viewport heights is a jump.
    pub jump_viewports: f64,
    /// How long movement must stay slow before the view counts as settled.
    pub settle_seconds: f64,
    /// The span over which speed is measured.
    pub window_seconds: f64,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            fast_viewports_per_second: 4.0,
            jump_viewports: 2.0,
            settle_seconds: 0.12,
            window_seconds: 0.1,
        }
    }
}

/// What the view is doing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Motion {
    /// At rest or moving slowly: fetch what is visible, then prefetch.
    Settled,
    /// Moving fast (or recently): fetch nothing new.
    Moving {
        /// Seconds until the view settles if it stays slow.
        settles_in: f64,
    },
}

/// The scroll direction of the latest movement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Direction {
    /// No movement yet.
    #[default]
    None,
    /// Towards the first row.
    Up,
    /// Towards the last row.
    Down,
}

const MAX_SAMPLES: usize = 64;

/// Tracks the viewport's top row over time.
#[derive(Debug, Clone, Default)]
pub struct ScrollPolicy {
    config: PolicyConfig,
    /// `(time, top row)`, oldest first.
    samples: VecDeque<(f64, u64)>,
    last_fast: Option<f64>,
    direction: Direction,
}

impl ScrollPolicy {
    /// A policy with the given tuning.
    pub fn new(config: PolicyConfig) -> Self {
        Self {
            config,
            ..Self::default()
        }
    }

    /// The tuning.
    pub fn config(&self) -> &PolicyConfig {
        &self.config
    }

    /// The direction of the latest movement.
    pub fn direction(&self) -> Direction {
        self.direction
    }

    /// Records the viewport's top row and height (in rows) at time `now` (seconds, any
    /// epoch, non-decreasing) and returns the motion.
    #[allow(clippy::cast_precision_loss)]
    pub fn observe(&mut self, top: u64, height: u64, now: f64) -> Motion {
        let now = match self.samples.back() {
            Some(&(last, _)) if now < last || !now.is_finite() => last,
            _ => now,
        };
        let height = height.max(1) as f64;
        if let Some(&(_, previous)) = self.samples.back() {
            if top != previous {
                self.direction = if top > previous {
                    Direction::Down
                } else {
                    Direction::Up
                };
            }
            let step = top.abs_diff(previous) as f64;
            let jump = step > self.config.jump_viewports * height;
            self.samples.push_back((now, top));
            let window_start = now - self.config.window_seconds;
            while self.samples.len() > MAX_SAMPLES
                || self
                    .samples
                    .get(1)
                    .is_some_and(|&(time, _)| time <= window_start)
            {
                self.samples.pop_front();
            }
            // A frame without movement is never fast: a stop registers at once.
            let fast = jump
                || (step > 0.0 && {
                    let (base_time, base_top) = self.samples.front().copied().unwrap_or((now, top));
                    let elapsed = (now - base_time).min(self.config.window_seconds).max(1e-3);
                    let speed = top.abs_diff(base_top) as f64 / elapsed / height;
                    speed > self.config.fast_viewports_per_second
                });
            if fast {
                self.last_fast = Some(now);
            }
        } else {
            self.samples.push_back((now, top));
        }
        match self.last_fast {
            Some(at) if now - at < self.config.settle_seconds => Motion::Moving {
                settles_in: (self.config.settle_seconds - (now - at)).max(0.0),
            },
            _ => Motion::Settled,
        }
    }
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::*;

    const FRAME: f64 = 1.0 / 60.0;

    #[test]
    fn the_first_view_is_settled() {
        let mut policy = ScrollPolicy::default();
        assert_eq!(policy.observe(1000, 40, 5.0), Motion::Settled);
    }

    #[test]
    fn fast_scrolling_moves_and_then_settles() {
        let mut policy = ScrollPolicy::default();
        let mut now = 0.0;
        policy.observe(0, 40, now);
        // 10 viewports per second.
        for frame in 1..=60u64 {
            now = frame as f64 * FRAME;
            let top = frame * 400 / 60;
            let motion = policy.observe(top, 40, now);
            if frame > 6 {
                assert!(matches!(motion, Motion::Moving { .. }), "frame {frame}");
            }
        }
        assert_eq!(policy.direction(), Direction::Down);
        // Stop: still moving until the settle time has passed.
        now += 0.05;
        assert!(matches!(
            policy.observe(400, 40, now),
            Motion::Moving { .. }
        ));
        now += 0.1;
        assert_eq!(policy.observe(400, 40, now), Motion::Settled);
    }

    #[test]
    fn slow_scrolling_stays_settled() {
        let mut policy = ScrollPolicy::default();
        policy.observe(0, 40, 0.0);
        // One wheel step of 3 rows every 100 ms: 0.75 viewports per second.
        for step in 1..=30u64 {
            let motion = policy.observe(step * 3, 40, step as f64 * 0.1);
            assert_eq!(motion, Motion::Settled, "step {step}");
        }
    }

    #[test]
    fn a_jump_is_fast_even_after_idling() {
        let mut policy = ScrollPolicy::default();
        policy.observe(0, 40, 0.0);
        let motion = policy.observe(100_000, 40, 10.0);
        assert!(matches!(motion, Motion::Moving { settles_in } if settles_in > 0.1));
        assert_eq!(policy.direction(), Direction::Down);
        assert_eq!(policy.observe(100_000, 40, 10.2), Motion::Settled);
        policy.observe(50, 40, 11.0);
        assert_eq!(policy.direction(), Direction::Up);
    }

    #[test]
    fn time_never_runs_backwards() {
        let mut policy = ScrollPolicy::default();
        policy.observe(0, 40, 5.0);
        assert!(matches!(
            policy.observe(10_000, 40, 1.0),
            Motion::Moving { .. }
        ));
        assert!(matches!(
            policy.observe(10_000, 40, f64::NAN),
            Motion::Moving { .. }
        ));
        assert_eq!(policy.observe(10_000, 40, 6.0), Motion::Settled);
    }
}

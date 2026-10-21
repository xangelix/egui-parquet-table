//! Cooperative cancellation of one fetch.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// A shared flag that tells a running fetch to stop.
///
/// Cloning shares the flag. Fetches check it before every read, so a cancelled fetch ends
/// at its next read with a [`Cancelled`](crate::FetchErrorKind::Cancelled) error. Setting
/// and checking never blocks.
#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    /// A token that is not cancelled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Asks the fetch to stop.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// Whether [`cancel`](Self::cancel) has been called on this token or a clone of it.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

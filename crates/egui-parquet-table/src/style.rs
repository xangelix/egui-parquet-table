//! How a [`crate::TableView`] looks: the host's colours ([`TableStyle`]), each defaulting to the
//! [`Ui`]'s visuals, and the small pieces drawn with them (gauges, chips).

use egui::{Color32, CornerRadius, Response, RichText, Sense, Stroke, Ui, vec2};

/// The colours of a [`crate::TableView`]. Each one left `None` follows the [`Ui`]'s visuals, so a
/// view looks at home in any application; a host with its own palette sets what it wants.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub struct TableStyle {
    /// The header row's background.
    pub header: Option<Color32>,
    /// Alternate rows' background.
    pub stripe: Option<Color32>,
    /// The overlay of the row under the pointer.
    pub hover: Option<Color32>,
    /// The row-number column's background, and the status line's.
    pub gutter: Option<Color32>,
    /// Gauges, `true`, and other highlights.
    pub accent: Option<Color32>,
    /// Secondary text: row numbers, nulls, types, labels.
    pub muted: Option<Color32>,
    /// Hairlines: the gutter's edge, the status line's border, chips.
    pub border: Option<Color32>,
}

impl TableStyle {
    /// Every colour from the visuals.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            header: None,
            stripe: None,
            hover: None,
            gutter: None,
            accent: None,
            muted: None,
            border: None,
        }
    }

    /// With the header row's background.
    #[must_use]
    pub const fn with_header(mut self, color: Color32) -> Self {
        self.header = Some(color);
        self
    }

    /// With alternate rows' background.
    #[must_use]
    pub const fn with_stripe(mut self, color: Color32) -> Self {
        self.stripe = Some(color);
        self
    }

    /// With the hovered row's overlay.
    #[must_use]
    pub const fn with_hover(mut self, color: Color32) -> Self {
        self.hover = Some(color);
        self
    }

    /// With the row-number column's and status line's background.
    #[must_use]
    pub const fn with_gutter(mut self, color: Color32) -> Self {
        self.gutter = Some(color);
        self
    }

    /// With the accent.
    #[must_use]
    pub const fn with_accent(mut self, color: Color32) -> Self {
        self.accent = Some(color);
        self
    }

    /// With secondary text's colour.
    #[must_use]
    pub const fn with_muted(mut self, color: Color32) -> Self {
        self.muted = Some(color);
        self
    }

    /// With the hairlines' colour.
    #[must_use]
    pub const fn with_border(mut self, color: Color32) -> Self {
        self.border = Some(color);
        self
    }

    /// Every colour, the visuals' where the style has none.
    pub(crate) fn resolve(&self, ui: &Ui) -> Paint {
        let visuals = ui.visuals();
        Paint {
            header: self.header.unwrap_or(visuals.faint_bg_color),
            stripe: self
                .stripe
                .unwrap_or_else(|| visuals.faint_bg_color.gamma_multiply(0.6)),
            hover: self
                .hover
                .unwrap_or_else(|| visuals.selection.bg_fill.gamma_multiply(0.25)),
            gutter: self.gutter.unwrap_or(visuals.faint_bg_color),
            accent: self.accent.unwrap_or(visuals.selection.stroke.color),
            muted: self.muted.unwrap_or_else(|| visuals.weak_text_color()),
            border: self
                .border
                .unwrap_or(visuals.widgets.noninteractive.bg_stroke.color),
            danger: visuals.error_fg_color,
        }
    }
}

/// A [`TableStyle`] resolved against the visuals, for one frame.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Paint {
    pub header: Color32,
    pub stripe: Color32,
    pub hover: Color32,
    pub gutter: Color32,
    pub accent: Color32,
    pub muted: Color32,
    pub border: Color32,
    pub danger: Color32,
}

/// A small horizontal gauge, `fraction` full (clamped to 0..=1), with `hover` on hover.
pub(crate) fn gauge(ui: &mut Ui, paint: &Paint, fraction: f32, hover: &str) -> Response {
    let (rect, response) = ui.allocate_exact_size(vec2(48.0, 6.0), Sense::hover());
    if ui.is_rect_visible(rect) {
        let painter = ui.painter();
        painter.rect_filled(rect, CornerRadius::same(3), paint.border);
        let fraction = if fraction.is_finite() {
            fraction.clamp(0.0, 1.0)
        } else {
            0.0
        };
        if fraction > 0.0 {
            let mut filled = rect;
            filled.set_width((rect.width() * fraction).max(rect.height()));
            painter.rect_filled(filled, CornerRadius::same(3), paint.accent);
        }
    }
    response.on_hover_text(hover)
}

/// A chip: small text in a rounded outline.
pub(crate) fn chip(ui: &mut Ui, paint: &Paint, text: &str) -> Response {
    egui::Frame::new()
        .stroke(Stroke::new(1.0, paint.border))
        .corner_radius(CornerRadius::same(255))
        .inner_margin(egui::Margin::symmetric(8, 1))
        .show(ui, |ui| {
            ui.label(RichText::new(text).small().color(paint.muted));
        })
        .response
}

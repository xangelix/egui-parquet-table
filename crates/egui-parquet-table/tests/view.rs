//! The widget driven frame by frame with `egui::Context::run_ui`: what it shows, what it
//! fetches (and when, and from which thread), the toolbar's controls, and huge row counts.

#![allow(
    missing_docs,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects
)]

mod common;

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use egui::accesskit::Role;
use egui::{Event, Key, Modifiers};
use egui_parquet_table::parquet_table_core::{Cell, LazyOptions};
use egui_parquet_table::{CopySummary, TableView, TableViewOptions};

use common::*;

/// The first rows of a Parquet file show their values within a few frames, styled by kind:
/// a null is italic and dimmed, unlike the string `"null"`.
#[test]
fn the_first_rows_show_their_values_after_a_few_frames() {
    let mut frames = Frames::new();
    let mut view = view(
        &frames,
        people_source(20_000),
        Arc::new(Inline),
        TableViewOptions::default(),
    );
    let (_, first) = frames.frame(&mut view, vec![]);
    assert!(labels(&first).contains("Loading"), "nothing loaded yet");

    let (output, full) = until(&mut frames, &mut view, 10, false, |view, output, _| {
        all_loaded(view, output)
    });
    assert_eq!(output.rows.start, 0);
    assert_eq!(output.columns, vec![0, 1, 2, 3, 4]);
    let labels = labels(&full);
    for expected in [
        "name 0", "name 2", "1.0", "true", "false", "note 1", "1", "20",
    ] {
        assert!(labels.contains(expected), "{expected} in {labels:?}");
    }

    // Row 3's name is null, row 4's is the string "null": drawn alike in text, not in style.
    let nulls: Vec<Drawn> = drawn(&full)
        .into_iter()
        .filter(|text| text.text == "null")
        .collect();
    assert!(nulls.iter().any(|text| text.italics), "{nulls:?}");
    assert!(nulls.iter().any(|text| !text.italics), "{nulls:?}");
    let null = nulls.iter().find(|text| text.italics).unwrap();
    let string = nulls.iter().find(|text| !text.italics).unwrap();
    assert_ne!(null.color, string.color, "a null is dimmed");

    // Numbers are right-aligned in their column: the ids' right edges line up.
    let ids: Vec<Drawn> = drawn(&full)
        .into_iter()
        .filter(|text| text.text == "7" || text.text == "17")
        .collect();
    let rights: Vec<f32> = ids.iter().map(|text| text.rect.right()).collect();
    assert!(rights.len() >= 4, "{ids:?}");
    // Each id appears twice (its row number in the sticky column, and the id): both columns
    // right-align, so the two "7"s and the two "17"s each share a right edge with the other.
    let mut edges: Vec<i32> = rights.iter().map(|x| x.round() as i32).collect();
    edges.sort_unstable();
    edges.dedup();
    assert_eq!(edges.len(), 2, "{rights:?}");

    // A long note is cut with an ellipsis.
    assert!(
        labels
            .iter()
            .any(|label| label.starts_with("long 0 xxx") && label.ends_with('…')),
        "{labels:?}"
    );
}

/// While the view flies through the rows nothing is fetched; once it rests, the rows there
/// are, and nothing between.
#[test]
fn a_fast_scroll_starts_no_fetch_until_it_settles() {
    let source = Synthetic::new(8, 10_000_000, 1_000);
    let mut frames = Frames::new();
    let mut view = view(
        &frames,
        source.clone(),
        Arc::new(Inline),
        TableViewOptions::default(),
    );
    // Settle at the top until nothing more is fetched.
    let mut last = None;
    until(&mut frames, &mut view, 200, false, |view, output, _| {
        let stats = view.stats();
        let stable = last == Some(stats.fetches_started) && stats.inflight_jobs == 0;
        last = Some(stats.fetches_started);
        stable && all_loaded(view, output)
    });
    let before = view.stats().fetches_started;
    let fetched_at_top = source.fetched().len();

    // A fling: 60,000 rows a frame for half a second.
    let mut moved = false;
    for step in 1..=30u64 {
        view.scroll_to_row(step * 60_000);
        let (output, _) = frames.frame(&mut view, vec![]);
        moved |= view.stats().moving;
        assert_eq!(
            view.stats().fetches_started,
            before,
            "step {step}: fetched while flying past rows {:?}",
            output.rows
        );
    }
    assert!(moved);

    // At rest where the fling ended (the kit scrolls a frame or two after it is asked):
    // within the settle time, still nothing.
    let (output, _) = frames.frames(&mut view, 3);
    let destination = output.rows.clone();
    assert!(destination.contains(&1_800_000), "{destination:?}");
    assert_eq!(view.stats().fetches_started, before);
    frames.wait(0.2);
    let (output, _) = until(&mut frames, &mut view, 30, false, |view, output, _| {
        all_loaded(view, output)
    });
    assert_eq!(output.rows, destination);
    assert!(view.stats().fetches_started > before);
    let row = destination.start + 3;
    let Cell::Ready { array, index } = view.engine().cell(row, 2) else {
        panic!("loaded");
    };
    let values = array
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .unwrap();
    assert_eq!(values.value(index), row as i64 * 1000 + 2);

    // Nothing between the top and the destination was ever read.
    let between = 20_000..destination.start - 20_000;
    let stray: Vec<_> = source.fetched()[fetched_at_top..]
        .iter()
        .filter(|block| {
            let start = u64::from(block.index) * 1_000;
            between.contains(&start)
        })
        .copied()
        .collect();
    assert!(stray.is_empty(), "{stray:?}");
}

/// Hidden columns are never fetched: hidden before the first frame, or in the column chooser.
#[test]
fn hidden_columns_are_never_fetched() {
    let source = Synthetic::new(12, 1_000_000, 1_000);
    let mut frames = Frames::new();
    let mut view = view(
        &frames,
        source.clone(),
        Arc::new(Inline),
        TableViewOptions::default(),
    );
    view.set_visible_columns([0, 2, 5, 11]);
    assert_eq!(view.visible_columns(), [0, 2, 5, 11]);
    let (output, full) = until(&mut frames, &mut view, 50, false, |view, output, _| {
        all_loaded(view, output) && view.stats().inflight_jobs == 0
    });
    assert_eq!(output.columns, vec![0, 2, 5, 11]);
    let fetched: HashSet<usize> = source.fetched().iter().map(|block| block.column).collect();
    assert!(
        fetched.is_subset(&HashSet::from([0, 2, 5, 11])),
        "{fetched:?}"
    );
    assert!(matches!(view.engine().cell(0, 1), Cell::Deferred));
    // Values sit under their own headers.
    let labels = labels(&full);
    assert!(
        labels.contains("10005") && labels.contains("10011"),
        "row 10"
    );
    assert!(!labels.contains("10001"), "column 1 is hidden");

    // Hide column 5 in the chooser.
    let menu = find(&full, Role::Button, "Columns (4/12)").expect("the chooser");
    frames.click(&mut view, menu.center());
    // The menu is laid out in its first frame and placed in the next.
    let (_, full) = frames.frame(&mut view, vec![]);
    let checkbox = find(&full, Role::CheckBox, "c5").expect("a checkbox per column");
    frames.click(&mut view, checkbox.center());
    assert_eq!(view.visible_columns(), [0, 2, 11]);

    // Elsewhere, after it was hidden, column 5 is never read.
    let fetches_before = source.fetched().len();
    view.go_to_row(500_000);
    frames.frame(
        &mut view,
        vec![Event::Key {
            key: Key::Escape,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: Modifiers::NONE,
        }],
    );
    frames.wait(0.3);
    let (output, _) = until(&mut frames, &mut view, 50, false, |view, output, _| {
        all_loaded(view, output) && output.rows.contains(&500_000)
    });
    assert_eq!(output.columns, vec![0, 2, 11]);
    let later: HashSet<usize> = source.fetched()[fetches_before..]
        .iter()
        .map(|block| block.column)
        .collect();
    assert!(!later.is_empty() && !later.contains(&5), "{later:?}");

    // Reset shows every column again.
    view.reset_columns();
    assert_eq!(view.visible_columns(), (0..12).collect::<Vec<_>>());
}

/// Copy puts the selected rows' loaded cells on the clipboard as tab-separated text with a
/// header row, and says how many selected rows weren't loaded.
#[test]
fn copy_puts_the_selected_loaded_rows_on_the_clipboard() {
    let mut frames = Frames::new();
    let mut view = view(
        &frames,
        people_source(20_000),
        Arc::new(Inline),
        TableViewOptions::default(),
    );
    view.set_visible_columns([0, 1]);
    let (_, full) = until(&mut frames, &mut view, 20, false, |view, output, _| {
        all_loaded(view, output) && view.stats().inflight_jobs == 0
    });
    // Copy is disabled until rows are selected.
    assert!(find(&full, Role::Button, "Copy").is_some());

    let selection = &mut view.table_state_mut().selected_rows;
    for row in [2u32, 3, 4, 19_000] {
        selection.insert(row);
    }
    let (_, full) = frames.frame(&mut view, vec![]);
    let copy = find(&full, Role::Button, "Copy").unwrap();
    frames.frame(
        &mut view,
        vec![
            Event::PointerMoved(copy.center()),
            press(copy.center(), true),
        ],
    );
    let (output, full) = frames.frame(&mut view, vec![press(copy.center(), false)]);
    assert_eq!(
        clipboard(&full).as_deref(),
        Some("id\tname\n2\tname 2\n3\t\n4\tnull\n")
    );
    assert_eq!(
        output.copied,
        Some(CopySummary {
            rows: 3,
            not_loaded: 1,
            over_limit: 0
        })
    );
    let (_, full) = frames.frame(&mut view, vec![]);
    assert!(
        labels(&full).contains("Copied 3 rows; 1 row selected wasn't loaded"),
        "the status line says so"
    );

    // A copy stops at its cell limit, a row at a time.
    let mut limited = TableView::new(
        people_source(20_000),
        Arc::new(Inline),
        &frames.ctx,
        TableViewOptions {
            copy_max_cells: 5,
            ..TableViewOptions::default()
        },
    );
    limited.set_visible_columns([0, 1]);
    until(&mut frames, &mut limited, 20, false, |view, output, _| {
        all_loaded(view, output)
    });
    limited.table_state_mut().selected_rows.insert_range(0..=5);
    let copied = limited.copy_selection();
    assert_eq!(copied.text, "id\tname\n0\tname 0\n1\tname 1\n");
    assert_eq!(
        copied.summary,
        CopySummary {
            rows: 2,
            not_loaded: 0,
            over_limit: 4
        }
    );
}

/// Typing a row number and pressing Enter scrolls there and selects the row.
#[test]
fn go_to_row_scrolls_there_and_selects_it() {
    let source = Synthetic::new(4, 1_000_000, 1_000);
    let mut frames = Frames::new();
    let mut view = view(
        &frames,
        source,
        Arc::new(Inline),
        TableViewOptions::default(),
    );
    let (_, full) = frames.frames(&mut view, 3);
    let field = nodes(&full)
        .into_iter()
        .find(|node| node.role == Role::TextInput)
        .expect("the row field");
    frames.click(&mut view, field.center());
    frames.frame(&mut view, vec![Event::Text("123,457".to_owned())]);
    frames.frame(
        &mut view,
        vec![Event::Key {
            key: Key::Enter,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: Modifiers::NONE,
        }],
    );
    assert_eq!(
        view.table_state().selected_rows.iter().collect::<Vec<_>>(),
        [123_456]
    );
    frames.wait(0.3);
    let (output, full) = until(&mut frames, &mut view, 30, false, |view, output, _| {
        output.rows.contains(&123_456) && all_loaded(view, output)
    });
    assert_eq!(output.selected_rows, 1);
    let labels = labels(&full);
    assert!(labels.contains("123457"), "its row number is drawn");
    assert!(labels.contains("123456002"), "its value is drawn");

    // A number past the end goes to the last row.
    let mut field_text = None;
    for node in nodes(&full) {
        if node.role == Role::TextInput {
            field_text = node.value;
        }
    }
    assert_eq!(field_text.as_deref(), Some("123,457"));
    view.go_to_row(5_000_000);
    frames.wait(0.3);
    let (output, _) = until(&mut frames, &mut view, 30, false, |view, output, _| {
        output.rows.contains(&999_999) && all_loaded(view, output)
    });
    assert_eq!(output.rows.end, 1_000_000);
    assert_eq!(
        view.table_state().selected_rows.iter().collect::<Vec<_>>(),
        [999_999]
    );
}

/// A failing source shows failed cells (labelled with the failure) and a Retry button, which
/// fetches them again.
#[test]
fn failed_blocks_show_and_retry_fetches_them_again() {
    let source = Synthetic::new(3, 100_000, 1_000);
    source.failing.store(true, Ordering::SeqCst);
    let mut frames = Frames::new();
    let mut view = view(
        &frames,
        source.clone(),
        Arc::new(Inline),
        TableViewOptions::default(),
    );
    let (_, full) = until(&mut frames, &mut view, 30, false, |view, _, _| {
        view.stats().failed_blocks > 0 && view.stats().inflight_jobs == 0
    });
    let (_, full) = {
        let _ = full;
        frames.frame(&mut view, vec![])
    };
    let shown = labels(&full);
    assert!(shown.contains("Failed: checksum mismatch"), "{shown:?}");
    assert!(
        shown.iter().any(|label| label.ends_with("blocks failed")),
        "the status line counts them"
    );
    let calls = source.fetch_calls();
    frames.frames(&mut view, 5);
    assert_eq!(
        source.fetch_calls(),
        calls,
        "permanent failures aren't retried"
    );

    source.failing.store(false, Ordering::SeqCst);
    let retry = find(&full, Role::Button, "Retry failed").expect("a retry button");
    frames.click(&mut view, retry.center());
    let (_, full) = until(&mut frames, &mut view, 30, false, |view, output, _| {
        all_loaded(view, output) && view.stats().failed_blocks == 0
    });
    assert!(source.fetch_calls() > calls);
    assert!(labels(&full).contains("5001"), "row 5's column 1");
    assert!(find(&full, Role::Button, "Retry failed").is_none());
}

/// Ten billion rows: drawn, scrolled to the end, selected where the kit's selection reaches
/// (rows below 2^32), and copied in time bounded by what is loaded.
#[cfg(target_pointer_width = "64")]
#[test]
fn more_rows_than_32_bits_hold() {
    const ROWS: u64 = 10_000_000_000;
    let source = Synthetic::new(3, ROWS, 1_000);
    let mut frames = Frames::new();
    let mut view = view(
        &frames,
        source,
        Arc::new(Inline),
        TableViewOptions::default(),
    );
    until(&mut frames, &mut view, 20, false, |view, output, _| {
        all_loaded(view, output)
    });

    view.go_to_row(ROWS - 1);
    frames.wait(0.3);
    let (output, full) = until(&mut frames, &mut view, 40, false, |view, output, _| {
        output.rows.contains(&(ROWS - 1)) && all_loaded(view, output)
    });
    assert_eq!(output.rows.end, ROWS);
    let shown = labels(&full);
    assert!(shown.contains("10000000000"), "the last row number");
    assert!(shown.contains("9999999999002"), "the last row's value");
    assert!(
        view.table_state().selected_rows.is_empty(),
        "past the selection's reach"
    );
    assert!(shown.contains("Rows 9,999,999,9") || shown.iter().any(|s| s.starts_with("Rows ")));

    // The last row the selection holds.
    // (The kit scrolls a table taller than 2^22 points on a scaled offset, so here it lands
    // within a few hundred rows of the row.)
    let last = u64::from(u32::MAX);
    view.go_to_row(last);
    frames.frames(&mut view, 3);
    frames.wait(0.3);
    let (output, _) = until(&mut frames, &mut view, 40, false, |view, output, _| {
        all_loaded(view, output)
    });
    assert!(
        output.rows.start.abs_diff(last) < 2_000,
        "{:?}",
        output.rows
    );
    assert_eq!(
        view.table_state().selected_rows.iter().collect::<Vec<_>>(),
        [u32::MAX]
    );

    // Selecting every row the selection holds, then copying: only loaded rows are visited.
    view.table_state_mut().selected_rows.insert_range(..);
    let copied = view.copy_selection();
    let summary = copied.summary;
    assert!(summary.rows > 0 && summary.rows < 100_000, "{summary:?}");
    assert_eq!(
        summary.rows + summary.not_loaded + summary.over_limit,
        1 << 32
    );
    assert_eq!(copied.text.lines().count() as u64, summary.rows + 1);
}

/// Drawing never reads the source: with fetches held, cells stay loading and the source is
/// never called; once the held fetches run (on another thread), the values appear.
#[test]
fn drawing_never_reads_the_source() {
    let source = Synthetic::new(5, 100_000, 500);
    let held = Arc::new(Held::default());
    let mut frames = Frames::new();
    let mut view = view(
        &frames,
        source.clone(),
        held.clone(),
        TableViewOptions::default(),
    );
    let (_, full) = frames.frames(&mut view, 8);
    assert_eq!(source.fetch_calls(), 0, "the frames called the source");
    assert!(held.len() > 0, "fetches wait for the spawner");
    assert!(labels(&full).contains("Loading"));
    assert!(!labels(&full).contains("2001"), "no value yet");

    let ui_thread = std::thread::current().id();
    for _ in 0..20 {
        let jobs = held.take();
        if jobs.is_empty() && view.stats().inflight_jobs == 0 {
            break;
        }
        std::thread::spawn(move || jobs.into_iter().for_each(|job| job()))
            .join()
            .unwrap();
        frames.frame(&mut view, vec![]);
    }
    assert!(source.fetch_calls() > 0);
    assert!(source.threads().iter().all(|thread| *thread != ui_thread));
    let (_, full) = frames.frame(&mut view, vec![]);
    assert!(labels(&full).contains("2001"), "row 2's column 1");

    // More frames, nothing new to fetch: the source isn't called.
    let calls = source.fetch_calls();
    frames.frames(&mut view, 10);
    let _ = held.take();
    assert_eq!(source.fetch_calls(), calls);
}

/// With fetches on worker threads, the view fills while the frames poll (bounded).
#[test]
fn worker_threads_fill_the_view() {
    let mut frames = Frames::new();
    let spawner = Arc::new(|job: Box<dyn FnOnce() + Send + 'static>| {
        std::thread::spawn(job);
    });
    let mut view = view(
        &frames,
        people_source(30_000),
        spawner,
        TableViewOptions {
            lazy: LazyOptions {
                max_jobs: 2,
                ..LazyOptions::default()
            },
            ..TableViewOptions::default()
        },
    );
    view.go_to_row(25_000);
    frames.frames(&mut view, 3);
    frames.wait(0.3);
    let (output, full) = until(&mut frames, &mut view, 5_000, true, |view, output, _| {
        output.rows.contains(&25_000) && all_loaded(view, output)
    });
    assert!(output.rows.contains(&25_000));
    let shown = labels(&full);
    assert!(
        shown.contains("name 25000") && shown.contains("note 25001"),
        "{shown:?}"
    );
}

/// A header describes its column on hover: types, codecs, sizes, nulls, and range.
#[test]
fn a_header_describes_its_column_on_hover() {
    let mut frames = Frames::new();
    let mut view = view(
        &frames,
        people_source(5_000),
        Arc::new(Inline),
        TableViewOptions::default(),
    );
    let (_, full) = frames.frames(&mut view, 3);
    let header = drawn(&full)
        .into_iter()
        .find(|text| text.text == "value")
        .expect("the header");
    frames.frame(&mut view, vec![Event::PointerMoved(header.rect.center())]);
    frames.wait(1.0);
    let (_, full) = frames.frames(&mut view, 2);
    let tooltip = drawn(&full)
        .into_iter()
        .find(|text| text.text.starts_with("value\nType: Float64"))
        .expect("the column's description");
    for part in [
        "Stored as: DOUBLE",
        "Nulls: ",
        "Min: 0.0",
        "Max: 2499.5",
        "Compression: ",
    ] {
        assert!(tooltip.text.contains(part), "{part} in {}", tooltip.text);
    }
}

/// Values are drawn by kind: booleans, binary as short hexadecimal (the rest on hover), nested
/// values compactly, and a value that can't be shown as a warning sign with a reason on hover.
#[test]
fn values_are_drawn_by_their_kind() {
    use arrow_array::builder::{ListBuilder, TimestampSecondBuilder};
    use arrow_array::types::Int32Type;
    use arrow_array::{ArrayRef, BinaryArray, BooleanArray, Int32Array, ListArray, StructArray};
    use egui_parquet_table::parquet_table_core::arrow::schema::{DataType, Field};

    let binary: ArrayRef = Arc::new(BinaryArray::from(vec![&[0xab_u8; 64][..], &[1, 2][..]]));
    let list: ArrayRef = Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
        Some(vec![Some(1), Some(2)]),
        None,
    ]));
    let record: ArrayRef = Arc::new(StructArray::from(vec![(
        Arc::new(Field::new("a", DataType::Int32, false)),
        Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
    )]));
    // Timestamps in a named zone inside a list can't be shown (no time-zone database).
    let mut zoned = ListBuilder::new(TimestampSecondBuilder::new().with_timezone("Europe/Berlin"));
    zoned.values().append_value(0);
    zoned.append(true);
    zoned.values().append_value(1);
    zoned.append(true);
    let zoned: ArrayRef = Arc::new(zoned.finish());
    let flags: ArrayRef = Arc::new(BooleanArray::from(vec![true, false]));
    let source = Arrays::new(vec![
        ("flag", flags),
        ("bytes", binary),
        ("list", list),
        ("record", record),
        ("zoned", zoned),
    ]);

    let mut frames = Frames::new();
    frames.size = egui::vec2(1600.0, 400.0);
    let mut view = view(
        &frames,
        source,
        Arc::new(Inline),
        TableViewOptions::default(),
    );
    let (_, full) = until(&mut frames, &mut view, 10, false, |view, output, _| {
        all_loaded(view, output)
    });
    let texts = text_counts(&full);
    let hex = "ab".repeat(16) + "…";
    for expected in [
        "true",
        "false",
        hex.as_str(),
        "0102",
        "[1, 2]",
        "{a: 1}",
        "{a: 2}",
        "⚠",
    ] {
        assert!(texts.contains_key(expected), "{expected} in {texts:?}");
    }
    let null = drawn(&full)
        .into_iter()
        .find(|text| text.text == "null")
        .expect("the null list");
    assert!(null.italics);

    // On hover: the whole value, or why it can't be shown.
    let hover = |frames: &mut Frames, view: &mut TableView, text: &str| {
        let (_, full) = frames.frame(view, vec![]);
        let at = drawn(&full)
            .into_iter()
            .find(|drawn| drawn.text == text)
            .unwrap()
            .rect
            .center();
        frames.frame(view, vec![Event::PointerMoved(at)]);
        frames.wait(1.0);
        frames.frames(view, 2).1
    };
    let full = hover(&mut frames, &mut view, &hex);
    assert!(
        text_counts(&full).contains_key(&"ab".repeat(64)),
        "the whole value"
    );
    let full = hover(&mut frames, &mut view, "⚠");
    assert!(text_counts(&full).contains_key("This value can't be shown"));
}

/// Once everything in view is loaded and nothing moves, the view asks for no repaint, and the
/// status line shows the cache, the fetches, and the caller's text.
#[test]
fn an_idle_view_rests() {
    let mut frames = Frames::new();
    let mut view = view(
        &frames,
        people_source(20_000),
        Arc::new(Inline),
        TableViewOptions::default(),
    );
    view.set_status_text(Some("page index: loaded".to_owned()));
    let mut last = None;
    until(&mut frames, &mut view, 100, false, |view, output, _| {
        let stats = view.stats();
        let stable = last == Some(stats.fetches_started) && stats.inflight_jobs == 0;
        last = Some(stats.fetches_started);
        stable && all_loaded(view, output)
    });
    // (egui's scroll bars fade in over the first frames.)
    frames.wait(2.0);
    let (_, full) = frames.frames(&mut view, 2);
    let delay = full.viewport_output[&egui::ViewportId::ROOT].repaint_delay;
    assert_eq!(delay, std::time::Duration::MAX, "no repaint asked for");
    let shown = labels(&full);
    for expected in [
        "Rows 1–",
        "Cache ",
        "/ 128.0 MiB",
        "0 fetches in flight",
        " decoded",
        "page index: loaded",
    ] {
        assert!(
            shown.iter().any(|label| label.contains(expected)),
            "{expected} in {shown:?}"
        );
    }
    assert!(shown.contains("20,000 rows × 5 columns"));

    // The glyphs drawn are in egui's default fonts: none is drawn as the replacement glyph
    // (`has_glyph` can't tell for the emoji font, which also holds the replacement).
    let font = egui::FontId::proportional(14.0);
    let uv = |c: char| {
        frames.ctx.fonts_mut(|fonts| {
            let galley = fonts.layout_no_wrap(c.to_string(), font.clone(), egui::Color32::WHITE);
            galley.rows[0].row.glyphs[0].uv_rect
        })
    };
    let missing = uv('\u{10FFFD}');
    assert_eq!(missing, uv('◻'), "the replacement glyph");
    for c in "⚠×–…#".chars() {
        assert_ne!(uv(c), missing, "{c} is in the default fonts");
    }
}

#[test]
fn the_view_can_move_between_threads() {
    fn send<T: Send>() {}
    send::<TableView>();
}

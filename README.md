# egui-parquet-table

Lazy, virtualized table views over Parquet data.

| Crate | What it is |
| --- | --- |
| [`parquet-table-core`](crates/parquet-table-core) | The engine, free of any UI toolkit: reads only the columns and rows in view, caches decoded blocks, and decides from the scroll motion when to fetch. |
| [`egui-parquet-table`](crates/egui-parquet-table) | The egui widget over the engine: `TableView`, for any `TableSource`. |

## parquet-table-core

The engine is built in layers:

- **`ByteSource`**: positional, concurrent reads of the file's bytes. `MemorySource` and
  `FileSource` (native) are included; integrators implement the trait for other storage
  (an encrypted archive, a remote object). A read error says whether it may be retried, so
  a failed integrity check is never retried.
- **`ParquetSource`**: opens a file by reading its footer, checks it against
  `ParquetLimits` before decoding (files may be hostile: footer size, schema depth, column
  and row-group counts, chunk ranges, page sizes, decompression bombs), loads the offset
  index when it fits under a cap, and divides every column into row blocks. A fetch decodes
  one row range of one row group for several columns at once; with the offset index only the
  pages holding those rows are read. Decoder panics become errors.
- **`TableSource`**: the format-independent interface the engine and the widget use.
- **`LazyTable`**: owned by the UI thread, which it never blocks. Once per frame it takes
  the viewport and the caller's clock; while the view moves fast it fetches nothing, once
  it settles it fetches the visible blocks (nearest the centre first) and then a prefetch
  ring. Fetches run through a caller-provided `Spawn` and return through a bounded,
  lock-free queue; stale fetches are cancelled; the cache is LRU under a byte budget and
  never evicts what is in view.
- **`format_cell`**: bounded, one-line display text for any Arrow value.

The crate starts no threads, takes no locks on the UI thread, reads no clock, and builds
for `wasm32-unknown-unknown`.

```rust,ignore
use std::sync::Arc;
use parquet_table_core::*;

let source = ParquetSource::open(FileSource::open("data.parquet")?, &ParquetLimits::default())?;
let spawner = Arc::new(|job: Box<dyn FnOnce() + Send>| { std::thread::spawn(job); });
let mut table = LazyTable::new(Arc::new(source), spawner, Arc::new(|| {}), LazyOptions::default());

// Every frame:
table.update(&Viewport { rows: 0..40, columns: 0..8 }, seconds_since_start);
match table.cell(3, 2) {
    Cell::Ready { array, index } => println!("{}", format_cell(array.as_ref(), index, 80).as_str()),
    Cell::Loading | Cell::Deferred => {}
    Cell::Failed(error) => println!("{error}"),
    Cell::OutOfRange => {}
}
```

`cargo run --release --example scroll_sim` writes a 5-million-row file and simulates a
fling from the first row to the last.

## egui-parquet-table

`TableView` draws a `TableSource` through
[`egui-table-kit`](https://github.com/xangelix/egui-table-kit)'s lazy mode. The caller owns
it and calls `show(ui)` every frame; it never blocks or locks, and takes its time from egui's
input. It has:

- a sticky row-number column, values styled by Arrow type (numbers right-aligned in
  monospace, nulls dimmed and italic, binary as short hex, nested values compact, long text cut
  with the rest on hover, undisplayable values as `⚠`), starting widths by type, resizable;
- column descriptions on the headers (types, codecs, sizes, nulls, range);
- a toolbar: size, Go to row, a column chooser (hidden columns are never fetched), Copy (the
  selected rows' loaded cells as TSV with a header row, bounded, saying what was left out), and
  Retry failed;
- a status line: the rows in view, the selection, the cache against its budget, fetches in
  flight, bytes fetched, failures, and an optional caller text.

`cargo run --release --example viewer -- FILE.parquet` opens a file (or, without one, writes and
opens a demo file of 2,000,000 rows and 30 columns).

## License

MIT; see [LICENSE](LICENSE).

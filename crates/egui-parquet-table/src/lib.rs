//! An egui widget that shows a large table (a Parquet file, or any [`TableSource`]) in a
//! virtualized view that loads lazily in both directions: only the columns and rows in view,
//! plus a prefetch ring, are read, and nothing is read while the view flies past rows.
//!
//! [`TableView`] is the widget. It drives a [`parquet_table_core::LazyTable`] once per frame
//! on the UI thread, without blocking or locking: fetches run through the caller's
//! [`Spawn`], and a finished fetch asks egui for a repaint. It draws through
//! `egui-table-kit` in its lazy mode, with a sticky row-number column, values styled by their
//! Arrow type, column descriptions on the headers, a toolbar (go to row, column chooser, copy,
//! retry), and a status line.
//!
//! ```no_run
//! use std::sync::Arc;
//! use egui_parquet_table::{TableView, TableViewOptions};
//! use parquet_table_core::{FileSource, ParquetLimits, ParquetSource};
//!
//! # fn open(ctx: &egui::Context) -> Result<TableView, Box<dyn std::error::Error>> {
//! let file = FileSource::open("data.parquet")?;
//! let source = Arc::new(ParquetSource::open(file, &ParquetLimits::default())?);
//! // Any executor: a thread pool natively, web workers in a browser.
//! let spawner = Arc::new(|job: Box<dyn FnOnce() + Send>| {
//!     std::thread::spawn(job);
//! });
//! let view = TableView::new(source, spawner, ctx, TableViewOptions::default());
//! # Ok(view)
//! # }
//! // Every frame, in the caller's UI:
//! // let output = view.show(ui);
//! ```
//!
//! The crate starts no threads, reads no clock (time is egui's input), and builds for
//! `wasm32-unknown-unknown`.

mod copy;
mod provider;
mod render;
mod text;
mod view;

pub use crate::copy::{Copied, CopySummary};
pub use crate::view::{TableView, TableViewOptions, TableViewOutput};
pub use egui_table_kit;
pub use parquet_table_core;
pub use parquet_table_core::{Spawn, TableSource};

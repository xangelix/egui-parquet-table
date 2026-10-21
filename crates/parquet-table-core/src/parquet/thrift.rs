//! A small reader of the Thrift compact protocol, used to check hostile structures before
//! the `parquet` crate decodes them.
//!
//! Two structures are pre-checked:
//!
//! * the whole footer ([`scan_footer`]): the schema tree and every row-group list header.
//!   The `parquet` crate builds the schema tree recursively and reserves vectors by
//!   declared counts (`num_children`, the row-group list size), so a few bytes of a hostile
//!   footer could exhaust the stack or ask for gigabytes, which aborts rather than panics;
//!   and
//! * page headers ([`parse_page_header`]), whose declared sizes drive allocations when the
//!   page is decompressed and decoded.
//!
//! A check is only worth something if this reader sees the same structure the `parquet`
//! crate will decode. Its primitives therefore mirror the crate's own (varints, zig-zag
//! truncation, field deltas, how unknown fields and boolean list elements are skipped,
//! the skip depth of 64). Where the crate reads a known field by its expected type
//! regardless of the declared one, this reader refuses a mismatched declared type instead:
//! both then consume the same bytes for every input this reader accepts.

use core::fmt;

/// Compact-protocol field and element types.
mod ty {
    pub(super) const BOOL_TRUE: u8 = 1;
    pub(super) const BOOL_FALSE: u8 = 2;
    pub(super) const BYTE: u8 = 3;
    pub(super) const I16: u8 = 4;
    pub(super) const I32: u8 = 5;
    pub(super) const I64: u8 = 6;
    pub(super) const DOUBLE: u8 = 7;
    pub(super) const BINARY: u8 = 8;
    pub(super) const LIST: u8 = 9;
    pub(super) const SET: u8 = 10;
    pub(super) const MAP: u8 = 11;
    pub(super) const STRUCT: u8 = 12;
    pub(super) const UUID: u8 = 13;
}

/// The skip depth the `parquet` crate uses for unknown fields.
const SKIP_DEPTH: u32 = 64;

/// Why a structure was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ThriftError {
    /// The input ended inside the structure.
    Eof,
    /// The structure is invalid (or uses a declared type this reader refuses).
    Invalid(&'static str),
    /// The structure exceeds a limit.
    Limit(&'static str),
}

impl fmt::Display for ThriftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Eof => f.write_str("the structure is truncated"),
            Self::Invalid(m) | Self::Limit(m) => f.write_str(m),
        }
    }
}

type Result<T> = core::result::Result<T, ThriftError>;

/// A cursor over a byte slice with the `parquet` crate's primitive semantics.
struct Input<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Input<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn byte(&mut self) -> Result<u8> {
        let b = *self.buf.get(self.pos).ok_or(ThriftError::Eof)?;
        self.pos = self.pos.checked_add(1).ok_or(ThriftError::Eof)?;
        Ok(b)
    }

    fn advance(&mut self, n: usize) -> Result<()> {
        if n > self.remaining() {
            return Err(ThriftError::Eof);
        }
        self.pos = self.pos.checked_add(n).ok_or(ThriftError::Eof)?;
        Ok(())
    }

    /// An unsigned varint, accumulated with wrapping shifts like the crate does.
    fn varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        let mut shift = 0u32;
        loop {
            let b = self.byte()?;
            value |= u64::from(b & 0x7f).wrapping_shl(shift);
            if b & 0x80 == 0 {
                return Ok(value);
            }
            shift = shift.wrapping_add(7);
        }
    }

    fn skip_varint(&mut self) -> Result<()> {
        while self.byte()? & 0x80 != 0 {}
        Ok(())
    }

    fn zigzag(&mut self) -> Result<i64> {
        let v = self.varint()?;
        Ok((v.wrapping_shr(1) as i64) ^ 0i64.wrapping_sub((v & 1) as i64))
    }

    /// An `i32` as the crate reads it (a zig-zag `i64` truncated).
    fn i32(&mut self) -> Result<i32> {
        Ok(self.zigzag()? as i32)
    }

    /// An `i16` as the crate reads it (a zig-zag `i64` truncated).
    fn i16(&mut self) -> Result<i16> {
        Ok(self.zigzag()? as i16)
    }

    fn skip_binary(&mut self) -> Result<()> {
        let len = self.varint()? as usize;
        self.advance(len)
    }

    /// A field header. `None` is the stop field.
    fn field(&mut self, last_id: i16) -> Result<Option<(i16, u8)>> {
        let b = self.byte()?;
        let field_type = b & 0x0f;
        if field_type == 0 {
            return Ok(None);
        }
        if field_type > ty::UUID {
            return Err(ThriftError::Invalid("unknown Thrift field type"));
        }
        let delta = b.wrapping_shr(4);
        let id = if delta != 0 {
            last_id
                .checked_add(i16::from(delta))
                .ok_or(ThriftError::Invalid("Thrift field id overflow"))?
        } else {
            self.i16()?
        };
        Ok(Some((id, field_type)))
    }

    /// A list or set header: (element type, size). Boolean elements are reported as
    /// `BOOL_TRUE`, as the crate does.
    fn list_header(&mut self) -> Result<(u8, usize)> {
        let header = self.byte()?;
        if header == 0 {
            return Ok((ty::BYTE, 0));
        }
        let element = element_type(header & 0x0f)?;
        let count = header.wrapping_shr(4);
        let size = if count != 15 {
            i32::from(count)
        } else {
            i32::try_from(self.varint()?)
                .map_err(|_| ThriftError::Invalid("Thrift list size overflow"))?
        };
        let size = usize::try_from(size).map_err(|_| ThriftError::Invalid("negative list size"))?;
        Ok((element, size))
    }

    /// Skips a value of the given type the way the crate's `skip` does.
    fn skip(&mut self, field_type: u8, depth: u32) -> Result<()> {
        let depth = depth
            .checked_sub(1)
            .ok_or(ThriftError::Invalid("Thrift structure nested too deeply"))?;
        match field_type {
            // Boolean values live in the field header (and, for list elements, the crate
            // skips nothing either).
            ty::BOOL_TRUE | ty::BOOL_FALSE => Ok(()),
            ty::BYTE => self.advance(1),
            ty::I16 | ty::I32 | ty::I64 => self.skip_varint(),
            ty::DOUBLE => self.advance(8),
            ty::BINARY => self.skip_binary(),
            ty::UUID => self.advance(16),
            ty::STRUCT => {
                while let Some((_, t)) = self.field(0)? {
                    self.skip(t, depth)?;
                }
                Ok(())
            }
            ty::LIST | ty::SET => {
                let (element, size) = self.list_header()?;
                if element != ty::BOOL_TRUE {
                    for _ in 0..size {
                        self.skip(element, depth)?;
                    }
                }
                Ok(())
            }
            ty::MAP => {
                let size = i32::try_from(self.varint()?)
                    .map_err(|_| ThriftError::Invalid("Thrift map size overflow"))?;
                if size > 0 {
                    let kv = self.byte()?;
                    let key = element_type(kv.wrapping_shr(4))?;
                    let value = element_type(kv & 0x0f)?;
                    if key != ty::BOOL_TRUE || value != ty::BOOL_TRUE {
                        for _ in 0..size {
                            self.skip(key, depth)?;
                            self.skip(value, depth)?;
                        }
                    }
                }
                Ok(())
            }
            _ => Err(ThriftError::Invalid("unknown Thrift field type")),
        }
    }

    /// Skips an unknown field like the crate does (a fresh depth budget).
    fn skip_unknown(&mut self, field_type: u8) -> Result<()> {
        self.skip(field_type, SKIP_DEPTH)
    }
}

fn element_type(t: u8) -> Result<u8> {
    match t {
        1 | 2 => Ok(ty::BOOL_TRUE),
        3..=13 => Ok(t),
        _ => Err(ThriftError::Invalid("unknown Thrift element type")),
    }
}

fn expect(actual: u8, expected: u8) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(ThriftError::Invalid("unexpected Thrift field type"))
    }
}

fn expect_bool(actual: u8) -> Result<()> {
    if actual == ty::BOOL_TRUE || actual == ty::BOOL_FALSE {
        Ok(())
    } else {
        Err(ThriftError::Invalid("unexpected Thrift field type"))
    }
}

/// How the crate decodes a known field.
#[derive(Debug, Clone, Copy)]
enum Spec {
    /// A scalar of this declared type (enums are `I32`, strings `BINARY`).
    Scalar(u8),
    /// A boolean (its value lives in the field header).
    Bool,
    /// A struct (or union) with these known fields.
    Struct(Fields),
    /// A list of elements of this type; `Fields` when the element is a struct.
    List(u8, Fields),
}

/// The known fields of a struct: `(id, how it is decoded)`. Other ids are skipped by
/// their declared type, as the crate does.
type Fields = &'static [(i16, Spec)];

const EMPTY: Fields = &[];
const I32: Spec = Spec::Scalar(ty::I32);
const I64: Spec = Spec::Scalar(ty::I64);
const BIN: Spec = Spec::Scalar(ty::BINARY);
const DOUBLE: Spec = Spec::Scalar(ty::DOUBLE);
const UNIT: Spec = Spec::Struct(EMPTY);

/// `union TimeUnit { 1: MILLIS, 2: MICROS, 3: NANOS }`
const TIME_UNIT: Fields = &[(1, UNIT), (2, UNIT), (3, UNIT)];
/// `TimeType` and `TimestampType`.
const TIME_TYPE: Fields = &[(1, Spec::Bool), (2, Spec::Struct(TIME_UNIT))];
/// `union LogicalType`.
const LOGICAL_TYPE: Fields = &[
    (1, UNIT),
    (2, UNIT),
    (3, UNIT),
    (4, UNIT),
    (5, Spec::Struct(&[(1, I32), (2, I32)])),
    (6, UNIT),
    (7, Spec::Struct(TIME_TYPE)),
    (8, Spec::Struct(TIME_TYPE)),
    (
        10,
        Spec::Struct(&[(1, Spec::Scalar(ty::BYTE)), (2, Spec::Bool)]),
    ),
    (11, UNIT),
    (12, UNIT),
    (13, UNIT),
    (14, UNIT),
    (15, UNIT),
    (16, Spec::Struct(&[(1, Spec::Scalar(ty::BYTE))])),
    (17, Spec::Struct(&[(1, BIN)])),
    (18, Spec::Struct(&[(1, BIN), (2, I32)])),
    (19, UNIT),
];
/// `SchemaElement` (field 5, `num_children`, is captured separately).
const SCHEMA_ELEMENT: Fields = &[
    (1, I32),
    (2, I32),
    (3, I32),
    (4, BIN),
    (5, I32),
    (6, I32),
    (7, I32),
    (8, I32),
    (9, I32),
    (10, Spec::Struct(LOGICAL_TYPE)),
];
const STATISTICS: Fields = &[
    (1, BIN),
    (2, BIN),
    (3, I64),
    (4, I64),
    (5, BIN),
    (6, BIN),
    (7, Spec::Bool),
    (8, Spec::Bool),
    (9, I64),
];
const PAGE_ENCODING_STATS: Fields = &[(1, I32), (2, I32), (3, I32)];
const SIZE_STATISTICS: Fields = &[
    (1, I64),
    (2, Spec::List(ty::I64, EMPTY)),
    (3, Spec::List(ty::I64, EMPTY)),
];
const BOUNDING_BOX: Fields = &[
    (1, DOUBLE),
    (2, DOUBLE),
    (3, DOUBLE),
    (4, DOUBLE),
    (5, DOUBLE),
    (6, DOUBLE),
    (7, DOUBLE),
    (8, DOUBLE),
];
const GEOSPATIAL_STATISTICS: Fields = &[
    (1, Spec::Struct(BOUNDING_BOX)),
    (2, Spec::List(ty::I32, EMPTY)),
];
/// `ColumnMetaData` as the crate reads it (it skips 3 and 8).
const COLUMN_META_DATA: Fields = &[
    (1, I32),
    (2, Spec::List(ty::I32, EMPTY)),
    (4, I32),
    (5, I64),
    (6, I64),
    (7, I64),
    (9, I64),
    (10, I64),
    (11, I64),
    (12, Spec::Struct(STATISTICS)),
    (13, Spec::List(ty::STRUCT, PAGE_ENCODING_STATS)),
    (14, I64),
    (15, I32),
    (16, Spec::Struct(SIZE_STATISTICS)),
    (17, Spec::Struct(GEOSPATIAL_STATISTICS)),
];
/// `ColumnChunk` as the crate reads it without encryption (it skips 8 and 9).
const COLUMN_CHUNK: Fields = &[
    (1, BIN),
    (2, I64),
    (3, Spec::Struct(COLUMN_META_DATA)),
    (4, I64),
    (5, I32),
    (6, I64),
    (7, I32),
];
const SORTING_COLUMN: Fields = &[(1, I32), (2, Spec::Bool), (3, Spec::Bool)];
/// `RowGroup` as the crate reads it (it skips 6).
const ROW_GROUP: Fields = &[
    (1, Spec::List(ty::STRUCT, COLUMN_CHUNK)),
    (2, I64),
    (3, I64),
    (4, Spec::List(ty::STRUCT, SORTING_COLUMN)),
    (5, I64),
    (7, Spec::Scalar(ty::I16)),
];
const KEY_VALUE: Fields = &[(1, BIN), (2, BIN)];
/// `union ColumnOrder`.
const COLUMN_ORDER: Fields = &[(1, UNIT), (2, UNIT), (3, UNIT)];

fn find(fields: Fields, id: i16) -> Option<Spec> {
    fields
        .iter()
        .find(|(known, _)| *known == id)
        .map(|(_, spec)| *spec)
}

/// Walks a struct with the given known fields.
fn walk_struct(input: &mut Input<'_>, fields: Fields) -> Result<()> {
    let mut last = 0i16;
    while let Some((id, t)) = input.field(last)? {
        match find(fields, id) {
            Some(spec) => walk_value(input, t, spec)?,
            None => input.skip_unknown(t)?,
        }
        last = id;
    }
    Ok(())
}

/// Reads a list header of the expected element type. An empty list passes whatever its
/// declared element type (the crate may refuse it; nothing is consumed either way).
fn walk_list_header(input: &mut Input<'_>, element: u8) -> Result<usize> {
    let (declared, size) = input.list_header()?;
    if size == 0 {
        return Ok(0);
    }
    expect(declared, element)?;
    // Every element we accept takes at least one byte.
    if size > input.remaining() {
        return Err(ThriftError::Invalid("a list is longer than its input"));
    }
    Ok(size)
}

/// Walks one value of a known field whose declared type is `declared`.
fn walk_value(input: &mut Input<'_>, declared: u8, spec: Spec) -> Result<()> {
    match spec {
        Spec::Scalar(expected) => {
            expect(declared, expected)?;
            input.skip(declared, SKIP_DEPTH)
        }
        Spec::Bool => expect_bool(declared),
        Spec::Struct(fields) => {
            expect(declared, ty::STRUCT)?;
            walk_struct(input, fields)
        }
        Spec::List(element, fields) => {
            expect(declared, ty::LIST)?;
            let size = walk_list_header(input, element)?;
            for _ in 0..size {
                if element == ty::STRUCT {
                    walk_struct(input, fields)?;
                } else {
                    input.skip(element, SKIP_DEPTH)?;
                }
            }
            Ok(())
        }
    }
}

/// One `SchemaElement`; returns its `num_children` (the last occurrence wins, as in the
/// crate).
fn walk_schema_element(input: &mut Input<'_>) -> Result<Option<i32>> {
    let mut num_children = None;
    let mut last = 0i16;
    while let Some((id, t)) = input.field(last)? {
        if id == 5 {
            expect(t, ty::I32)?;
            num_children = Some(input.i32()?);
        } else {
            match find(SCHEMA_ELEMENT, id) {
                Some(spec) => walk_value(input, t, spec)?,
                None => input.skip_unknown(t)?,
            }
        }
        last = id;
    }
    Ok(num_children)
}

/// Limits checked before the footer is decoded.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FooterLimits {
    pub max_columns: usize,
    pub max_schema_depth: usize,
    pub max_row_groups: usize,
    pub max_column_chunks: u64,
}

/// What the footer pre-check learned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct FooterSummary {
    /// Leaf (stored) columns of the schema.
    pub leaves: usize,
    /// The schema tree's depth (the root is depth 1).
    pub depth: usize,
    /// Row groups declared.
    pub row_groups: usize,
}

/// Checks the schema tree the crate will build from `num_children` counts, without
/// recursion: one root, every count within the elements that remain, bounded depth and
/// leaf count.
fn check_schema_tree(children: &[Option<i32>], limits: &FooterLimits) -> Result<(usize, usize)> {
    let mut iter = children.iter();
    let root = iter
        .next()
        .ok_or(ThriftError::Invalid("the schema is empty"))?;
    let root_children = root.unwrap_or(0);
    let root_children =
        usize::try_from(root_children).map_err(|_| ThriftError::Invalid("negative child count"))?;
    if root_children == 0 {
        // An empty schema; any further element would be a second root.
        return if children.len() == 1 {
            Ok((0, 1))
        } else {
            Err(ThriftError::Invalid("the schema has more than one root"))
        };
    }
    let total = children.len();
    if root_children >= total {
        return Err(ThriftError::Invalid(
            "a schema child count exceeds the elements",
        ));
    }
    // Remaining children per open group, innermost last.
    let mut stack: Vec<usize> = vec![root_children];
    let mut max_depth = 1usize;
    let mut leaves = 0usize;
    for (index, element) in children.iter().enumerate().skip(1) {
        // Close finished groups.
        while stack.last() == Some(&0) {
            stack.pop();
        }
        let open = stack
            .last_mut()
            .ok_or(ThriftError::Invalid("the schema has more than one root"))?;
        *open = open.saturating_sub(1);
        let n = element.unwrap_or(0);
        let n = usize::try_from(n).map_err(|_| ThriftError::Invalid("negative child count"))?;
        if n == 0 {
            leaves = leaves.saturating_add(1);
            if leaves > limits.max_columns {
                return Err(ThriftError::Limit("too many columns"));
            }
        } else {
            let after = total.saturating_sub(index).saturating_sub(1);
            if n > after {
                return Err(ThriftError::Invalid(
                    "a schema child count exceeds the elements",
                ));
            }
            stack.push(n);
            max_depth = max_depth.max(stack.len());
            if max_depth > limits.max_schema_depth {
                return Err(ThriftError::Limit("the schema is nested too deeply"));
            }
        }
    }
    if stack.iter().any(|&n| n != 0) {
        return Err(ThriftError::Invalid("the schema is truncated"));
    }
    Ok((leaves, max_depth))
}

/// Pre-checks a whole footer (`FileMetaData`) the way the crate will decode it: the
/// schema tree, every row-group list header (the crate reserves by its declared size), and
/// the declared types of every field the crate reads.
pub(crate) fn scan_footer(buf: &[u8], limits: &FooterLimits) -> Result<FooterSummary> {
    let mut input = Input::new(buf);
    let mut summary: Option<FooterSummary> = None;
    let mut last = 0i16;
    while let Some((id, t)) = input.field(last)? {
        match id {
            1 => walk_value(&mut input, t, I32)?,
            3 => walk_value(&mut input, t, I64)?,
            // The crate skips a repeated schema.
            2 if summary.is_some() => input.skip_unknown(t)?,
            2 => {
                expect(t, ty::LIST)?;
                let (element, size) = input.list_header()?;
                expect(element, ty::STRUCT)?;
                if size > input.remaining() {
                    return Err(ThriftError::Invalid("a list is longer than its input"));
                }
                // Each element is at least one byte, so this is bounded by the footer size.
                let mut children = Vec::with_capacity(size);
                for _ in 0..size {
                    children.push(walk_schema_element(&mut input)?);
                }
                let (leaves, depth) = check_schema_tree(&children, limits)?;
                summary = Some(FooterSummary {
                    leaves,
                    depth,
                    row_groups: 0,
                });
            }
            4 => {
                let current =
                    summary.ok_or(ThriftError::Invalid("the row groups precede the schema"))?;
                expect(t, ty::LIST)?;
                let size = walk_list_header(&mut input, ty::STRUCT)?;
                if size > limits.max_row_groups {
                    return Err(ThriftError::Limit("too many row groups"));
                }
                let chunks = u64::try_from(size)
                    .unwrap_or(u64::MAX)
                    .saturating_mul(u64::try_from(current.leaves).unwrap_or(u64::MAX));
                if chunks > limits.max_column_chunks {
                    return Err(ThriftError::Limit("too many column chunks"));
                }
                for _ in 0..size {
                    walk_struct(&mut input, ROW_GROUP)?;
                }
                // The last list wins, as in the crate.
                summary = Some(FooterSummary {
                    row_groups: size,
                    ..current
                });
            }
            5 => walk_value(&mut input, t, Spec::List(ty::STRUCT, KEY_VALUE))?,
            6 => walk_value(&mut input, t, BIN)?,
            7 => walk_value(&mut input, t, Spec::List(ty::STRUCT, COLUMN_ORDER))?,
            // 8 and 9 (encryption) are skipped by a crate built without encryption.
            _ => input.skip_unknown(t)?,
        }
        last = id;
    }
    summary.ok_or(ThriftError::Invalid("the footer has no schema"))
}

/// What the guard needs from a page header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct PageHeaderInfo {
    /// The encoded header's length.
    pub header_len: usize,
    /// `PageType` (0 data, 1 index, 2 dictionary, 3 data v2).
    pub page_type: i32,
    pub uncompressed: i32,
    pub compressed: i32,
    /// `num_values` of a data (v1 or v2) page.
    pub data_values: Option<i32>,
    /// `num_rows` of a data page v2.
    pub data_rows: Option<i32>,
    /// `num_values` of a dictionary page.
    pub dictionary_values: Option<i32>,
    /// Level byte lengths of a data page v2 (definition, repetition).
    pub v2_levels: Option<(i32, i32)>,
    /// `is_compressed` of a data page v2 (defaults to true).
    pub v2_compressed: bool,
}

/// `DataPageHeader` (statistics skipped, as the crate does).
fn walk_data_page_header(input: &mut Input<'_>, info: &mut PageHeaderInfo) -> Result<()> {
    let mut last = 0i16;
    while let Some((id, t)) = input.field(last)? {
        match id {
            1 => {
                expect(t, ty::I32)?;
                info.data_values = Some(input.i32()?);
            }
            2..=4 => {
                expect(t, ty::I32)?;
                input.skip_varint()?;
            }
            _ => input.skip_unknown(t)?,
        }
        last = id;
    }
    Ok(())
}

/// `DataPageHeaderV2` (statistics skipped, as the crate does).
fn walk_data_page_header_v2(input: &mut Input<'_>, info: &mut PageHeaderInfo) -> Result<()> {
    let mut last = 0i16;
    let mut def = 0i32;
    let mut rep = 0i32;
    while let Some((id, t)) = input.field(last)? {
        match id {
            1 => {
                expect(t, ty::I32)?;
                info.data_values = Some(input.i32()?);
            }
            3 => {
                expect(t, ty::I32)?;
                info.data_rows = Some(input.i32()?);
            }
            2 | 4 => {
                expect(t, ty::I32)?;
                input.skip_varint()?;
            }
            5 => {
                expect(t, ty::I32)?;
                def = input.i32()?;
            }
            6 => {
                expect(t, ty::I32)?;
                rep = input.i32()?;
            }
            7 => {
                expect_bool(t)?;
                info.v2_compressed = t == ty::BOOL_TRUE;
            }
            _ => input.skip_unknown(t)?,
        }
        last = id;
    }
    info.v2_levels = Some((def, rep));
    Ok(())
}

/// Why a page header could not be parsed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HeaderError {
    /// The buffer ends inside the header: read more and try again.
    Incomplete,
    /// The header is invalid.
    Malformed(&'static str),
}

/// Parses a page header at the start of `buf` the way the crate's
/// `read_thrift_without_stats` will.
pub(crate) fn parse_page_header(buf: &[u8]) -> core::result::Result<PageHeaderInfo, HeaderError> {
    let mut input = Input::new(buf);
    let mut info = PageHeaderInfo {
        v2_compressed: true,
        ..PageHeaderInfo::default()
    };
    let mut seen = (false, false, false);
    let result = (|| -> Result<()> {
        let mut last = 0i16;
        while let Some((id, t)) = input.field(last)? {
            match id {
                1 => {
                    expect(t, ty::I32)?;
                    info.page_type = input.i32()?;
                    seen.0 = true;
                }
                2 => {
                    expect(t, ty::I32)?;
                    info.uncompressed = input.i32()?;
                    seen.1 = true;
                }
                3 => {
                    expect(t, ty::I32)?;
                    info.compressed = input.i32()?;
                    seen.2 = true;
                }
                4 => {
                    expect(t, ty::I32)?;
                    input.skip_varint()?;
                }
                5 => {
                    expect(t, ty::STRUCT)?;
                    walk_data_page_header(&mut input, &mut info)?;
                }
                6 => {
                    expect(t, ty::STRUCT)?;
                    walk_struct(&mut input, EMPTY)?;
                }
                7 => {
                    expect(t, ty::STRUCT)?;
                    let mut last_inner = 0i16;
                    while let Some((inner, it)) = input.field(last_inner)? {
                        match inner {
                            1 => {
                                expect(it, ty::I32)?;
                                info.dictionary_values = Some(input.i32()?);
                            }
                            2 => {
                                expect(it, ty::I32)?;
                                input.skip_varint()?;
                            }
                            3 => expect_bool(it)?,
                            _ => input.skip_unknown(it)?,
                        }
                        last_inner = inner;
                    }
                }
                8 => {
                    expect(t, ty::STRUCT)?;
                    walk_data_page_header_v2(&mut input, &mut info)?;
                }
                _ => input.skip_unknown(t)?,
            }
            last = id;
        }
        Ok(())
    })();
    match result {
        Ok(()) => {}
        Err(ThriftError::Eof) => return Err(HeaderError::Incomplete),
        Err(ThriftError::Invalid(m) | ThriftError::Limit(m)) => {
            return Err(HeaderError::Malformed(m));
        }
    }
    if !(seen.0 && seen.1 && seen.2) {
        return Err(HeaderError::Malformed(
            "a page header lacks a required field",
        ));
    }
    info.header_len = input.pos;
    Ok(info)
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::*;

    fn limits() -> FooterLimits {
        FooterLimits {
            max_columns: 100,
            max_schema_depth: 4,
            max_row_groups: 10,
            max_column_chunks: 100,
        }
    }

    #[test]
    fn schema_trees() {
        let l = limits();
        // root { a, b { c, d } }
        let tree = [Some(2), None, Some(2), None, Some(0)];
        assert_eq!(check_schema_tree(&tree, &l), Ok((3, 2)));
        // A child count beyond the elements.
        let tree = [Some(2), Some(i32::MAX), None];
        assert!(matches!(
            check_schema_tree(&tree, &l),
            Err(ThriftError::Invalid(_))
        ));
        // Negative.
        let tree = [Some(1), Some(-1)];
        assert!(check_schema_tree(&tree, &l).is_err());
        // Truncated.
        let tree = [Some(3), None, None];
        assert!(check_schema_tree(&tree, &l).is_err());
        // Two roots.
        let tree = [Some(1), None, None];
        assert!(check_schema_tree(&tree, &l).is_err());
        // Too deep: root > g > g > g > g > leaf.
        let tree = [Some(1), Some(1), Some(1), Some(1), Some(1), None];
        assert_eq!(
            check_schema_tree(&tree, &l),
            Err(ThriftError::Limit("the schema is nested too deeply"))
        );
        // Empty schema.
        assert_eq!(check_schema_tree(&[None], &l), Ok((0, 1)));
    }

    #[test]
    fn varints_mirror_the_crate() {
        let mut input = Input::new(&[0x96, 0x01]);
        assert_eq!(input.varint(), Ok(150));
        // zig-zag: 3 -> -2
        let mut input = Input::new(&[0x03]);
        assert_eq!(input.i32(), Ok(-2));
        let mut input = Input::new(&[0x80]);
        assert_eq!(input.varint(), Err(ThriftError::Eof));
    }

    #[test]
    fn page_headers() {
        // type=0 (i32 0), uncompressed=100, compressed=50, data_page_header{num_values=10,
        // encoding=0, def=3, rep=3}, stop.
        let header = [
            0x15, 0x00, // field 1, i32, 0
            0x15, 0xc8, 0x01, // field 2, i32, 100
            0x15, 0x64, // field 3, i32, 50
            0x2c, // field 5 (delta 2), struct
            0x15, 0x14, // field 1, i32, 10
            0x15, 0x00, 0x15, 0x06, 0x15, 0x06, // fields 2..4
            0x00, // stop (inner)
            0x00, // stop
        ];
        let info = parse_page_header(&header).unwrap();
        assert_eq!(info.header_len, header.len());
        assert_eq!(
            (info.page_type, info.uncompressed, info.compressed),
            (0, 100, 50)
        );
        assert_eq!(info.data_values, Some(10));
        for cut in 0..header.len() {
            assert_eq!(
                parse_page_header(&header[..cut]),
                Err(HeaderError::Incomplete),
                "cut at {cut}"
            );
        }
        // A known field with the wrong declared type is refused.
        let bad = [0x18, 0x00, 0x00];
        assert!(matches!(
            parse_page_header(&bad),
            Err(HeaderError::Malformed(_))
        ));
    }
}

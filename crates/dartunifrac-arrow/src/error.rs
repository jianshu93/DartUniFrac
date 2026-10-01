use std::fmt;

/// What went wrong converting a caller's data into [`dartunifrac_core`]'s types.
///
/// The variants separate *where* the problem is, because a caller can act on
/// that: a `Schema` error means the query projected the wrong columns, while a
/// `Data` error names a row it can go and look at. [`Stream`](Self::Stream) is
/// the caller's own reader failing, not us rejecting anything.
#[derive(Debug, PartialEq, Eq)]
pub enum MarshalError {
    /// A required column is missing, has the wrong type, or holds a null.
    Schema(String),
    /// A value is out of range or not finite. The message names the row.
    Data(String),
    /// The tree arrays are inconsistent with each other or with themselves.
    Tree(String),
    /// The caller's `RecordBatchReader` returned an error; text preserved.
    Stream(String),
    /// The sketch set cannot be compared as given: rows of differing length,
    /// rows with no values at all, an unsupported `bbits`, or an id count that
    /// disagrees with the sketch count.
    Sketch(String),
}

impl fmt::Display for MarshalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Schema(m) => write!(f, "schema: {m}"),
            Self::Data(m) => write!(f, "data: {m}"),
            Self::Tree(m) => write!(f, "tree: {m}"),
            Self::Stream(m) => write!(f, "stream: {m}"),
            Self::Sketch(m) => write!(f, "sketch: {m}"),
        }
    }
}

impl std::error::Error for MarshalError {}

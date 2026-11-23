//! Vector fields: east/north components at cell centres.

/// East and north components of a vector field, row-major like the
/// [`Slice`](crate::Slice) they belong to; `NaN` marks missing data.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorParts {
    pub east: Vec<f32>,
    pub north: Vec<f32>,
}

use std::fmt::Debug;

use crate::face::Face;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CullMode {
    /// Fully opaque. Neighbour faces against this block are always culled.
    Opaque,
    /// Transparent; faces between two blocks with the same ID are culled
    /// (e.g. glass, water).
    TransparentMerged,
    /// Transparent; faces are always drawn even between identical blocks
    /// (e.g. leaves).
    TransparentUnmerged,
    /// Invisible and non-renderable (e.g. air). Never produces geometry.
    /// Behaves like [`TransparentMerged`] for culling.
    Empty,
}

impl CullMode {
    #[inline]
    pub fn is_renderable(self) -> bool {
        !matches!(self, CullMode::Empty)
    }
}

/// Thickness in 1/16ths of a block. Full block = 16. Slab range: 1..=15.
pub type Thickness = u32;
pub const FULL_THICKNESS: Thickness = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlabInfo {
    /// Which face the slab is fully touching (e.g. PosY for an upper-slab).
    pub face: Face,
    /// Thickness in 1/16ths of a block: 1..=15.
    pub thickness: Thickness,
}

/// Horizontal stretch for cross-shaped billboard blocks, in 1/16ths.
/// 0 means the quad corners sit on the block's diagonal (standard sugar cane).
/// Higher values push the quad edges outward toward the block corners (cobwebs).
pub type CrossStretch = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CrossInfo {
    /// The face the cross takes root on (e.g. NegY for a shrub on the ground).
    /// The face's axis becomes the merge axis; the two perpendicular axes
    /// form the crossing plane.
    pub face: Face,
    /// Horizontal stretch in 1/16ths. 0 = square, positive = wider.
    pub stretch: CrossStretch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    WholeBlock,
    Slab(SlabInfo),
    /// X-shaped diagonal billboard (e.g. sugar cane, cobwebs).
    /// Two quads crossing diagonally through the block center, oriented
    /// by the root face.
    Cross(CrossInfo),
    /// Flat zero-thickness face hugging one side of the block, offset
    /// 1/16th inward. Rendered double-sided (e.g. ladders, rails).
    Facade(Face),
    /// Block with 4 horizontal faces inset by `n` sixteenths. Top/bottom
    /// are flush. Each side face is still full block size (e.g. cactus).
    Inset(Thickness),
}

pub trait Block: Copy + PartialEq + Debug {
    fn shape(&self) -> Shape;
    fn cull_mode(&self) -> CullMode;
}

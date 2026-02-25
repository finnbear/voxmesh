use std::fmt::Debug;

use crate::face::Face;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CullMode<T: PartialEq = ()> {
    /// Fully opaque. Neighbor faces against this block are always culled.
    Opaque,
    /// Transparent. Faces between two blocks whose `T` values are equal
    /// are culled (e.g. glass, water).
    TransparentMerged(T),
    /// Transparent. Faces are always drawn even between identical blocks
    /// (e.g. leaves).
    TransparentUnmerged,
    /// Invisible and non-renderable (e.g. air). Never produces geometry.
    Empty,
}

impl<T: PartialEq> CullMode<T> {
    #[inline]
    pub fn is_renderable(&self) -> bool {
        !matches!(self, CullMode::Empty)
    }
}

/// Thickness in 1/16ths of a block. Full block = 16, slab range 1..=15.
pub type Thickness = u32;
pub const FULL_THICKNESS: Thickness = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlabInfo {
    /// Which face the slab is flush with (e.g. PosY for an upper slab).
    pub face: Face,
    /// Thickness in 1/16ths of a block, range 1..=15.
    pub thickness: Thickness,
}

/// Horizontal stretch for cross-shaped billboard blocks in 1/16ths.
/// 0 means the quad corners sit on the block diagonal (sugar cane).
/// Higher values push the edges outward toward the block corners (cobwebs).
pub type CrossStretch = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CrossInfo {
    /// The face the cross is rooted on (e.g. NegY for a ground shrub).
    /// The face's axis becomes the merge axis and the two perpendicular
    /// axes form the crossing plane.
    pub face: Face,
    /// Horizontal stretch in 1/16ths. 0 = square, positive = wider.
    pub stretch: CrossStretch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    WholeBlock,
    Slab(SlabInfo),
    /// X-shaped diagonal billboard (e.g. sugar cane, cobwebs). Two
    /// quads crossing diagonally through the block center, oriented by
    /// the root face.
    Cross(CrossInfo),
    /// Flat zero-thickness face on one side of the block, offset 1/16th
    /// inward. Rendered double-sided (e.g. ladders, rails).
    Facade(Face),
    /// Block with horizontal faces inset by `n` sixteenths. Top and
    /// bottom are flush. Side faces are still full height (e.g. cactus).
    Inset(Thickness),
}

pub trait Block: Copy + PartialEq + Debug {
    /// Type used to determine whether two transparent blocks should have
    /// their shared face culled. Two `TransparentMerged` blocks cull
    /// their shared face when their `TransparentGroup` values are equal.
    /// This does not replace `Self: PartialEq` for greedy meshing.
    type TransparentGroup: Copy + PartialEq + Debug;

    fn shape(&self) -> Shape;
    fn cull_mode(&self) -> CullMode<Self::TransparentGroup>;
}

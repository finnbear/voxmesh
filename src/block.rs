use std::fmt::Debug;

use crate::face::AlignedFace;
use crate::light::Light;

/// How a block interacts with neighbor face culling.
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

/// A partial-height slab attached to one face.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlabInfo {
    /// Which face the slab is flush with (e.g. PosY for an upper slab).
    pub face: AlignedFace,
    /// Thickness in 1/16ths of a block, range 1..=15.
    pub thickness: Thickness,
}

/// Horizontal stretch for cross-shaped blocks in 1/16ths.
pub type CrossStretch = u32;

/// Configuration for an X-shaped diagonal cross block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CrossInfo {
    /// The face the cross is rooted on (e.g. NegY for a ground shrub).
    /// The face's axis becomes the merge axis and the two perpendicular
    /// axes form the crossing plane.
    pub face: AlignedFace,
    /// Horizontal stretch in 1/16ths. 0 = square, positive = wider.
    pub stretch: CrossStretch,
}

/// The geometric shape of a block, controlling quad generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// Full 1×1×1 cube.
    WholeBlock,
    /// Partial-height slab flush with one face.
    Slab(SlabInfo),
    /// X-shaped diagonal billboard (e.g. sugar cane, cobwebs).
    Cross(CrossInfo),
    /// Flat face offset 1/16th inward, rendered double-sided (e.g. ladders).
    Facade(AlignedFace),
    /// Side faces inset by `n` sixteenths; top/bottom flush (e.g. cactus).
    Inset(Thickness),
}

/// A voxel block type. Implement this to describe your block's shape,
/// culling behavior, and lighting for the mesher.
pub trait Block: Copy + PartialEq + Debug {
    /// Type used to determine whether two transparent blocks should have
    /// their shared face culled. Two `TransparentMerged` blocks cull
    /// their shared face when their `TransparentGroup` values are equal.
    /// This does not replace `Self: PartialEq` for greedy meshing.
    type TransparentGroup: Copy + PartialEq + Debug;

    /// Per-vertex light type for smooth lighting. Set to `()` (the
    /// default) to disable lighting and AO at zero cost.
    type Light: Light = ();

    fn shape(&self) -> Shape;
    fn cull_mode(&self) -> CullMode<Self::TransparentGroup>;

    /// Whether this block's material occludes ambient occlusion.
    ///
    /// This should reflect the material only (e.g. stone → true, glass
    /// → true, leaves → false, air → false). Shape-dependent logic
    /// (slabs only occlude on their flush side) is handled by the mesher.
    fn ao_opaque(&self) -> bool {
        matches!(self.cull_mode(), CullMode::Opaque)
    }

    /// The light value of this voxel for smooth per-vertex lighting.
    fn light(&self) -> Self::Light {
        Self::Light::default()
    }
}

use std::fmt::Debug;

use crate::face::{AlignedFace, Axis};
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

/// Configuration for a flat double-sided quad parallel to one face.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FacadeInfo {
    /// The face the quad is parallel to and offset inward from.
    pub face: AlignedFace,
    /// Offset from the named face in 1/16ths of a block, range 0..=16.
    /// Ladders use 1; a block-centered plane uses 8.
    pub offset: u8,
}

/// Configuration for a fluid column whose surface is a shared height
/// field rather than a flat face.
///
/// Unlike every other shape, a fluid's geometry is not decided by the
/// block alone. The mesher raises [`height`](Self::height) to
/// [`FULL_THICKNESS`] when the cell one step along [`face`](Self::face)
/// holds the same fluid, and it lifts each surface vertex to the tallest
/// of the four columns meeting at that vertex. Both rules read only the
/// neighborhood the vertex is shared with, so two adjacent fluid blocks
/// always agree on the position of a vertex they share and their
/// surfaces meet without a seam.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FluidInfo {
    /// The face the fluid's surface faces, and the direction its depth
    /// is measured from. `PosY` for water; a fluid with a different
    /// face fills its cell from the opposite side.
    ///
    /// All blocks sharing an [`id`](Self::id) must agree on this.
    pub face: AlignedFace,
    /// This column's height in 1/16ths of a block, range 1..=16.
    ///
    /// How a consumer derives this from its own flow state is up to it,
    /// so the number of levels is not baked into the mesher — a source
    /// is 16 and anything shallower is less.
    pub height: Thickness,
    /// Which fluid this is. Two adjacent fluid blocks join their
    /// surfaces only when their ids match, so water does not stitch
    /// itself onto lava.
    pub id: u8,
}

/// Configuration for a stair: a half-cell slab with a quarter-cell step on
/// it.
///
/// Two perpendicular faces fix it completely. The slab is flush with
/// [`floor`](Self::floor) and fills that half of the cell; the step stands
/// on the slab, flush with [`back`](Self::back), and fills the back half of
/// the remaining half. That gives 24 orientations: a stair on the floor
/// (`floor: NegY`), one hung from the ceiling (`floor: PosY`), and the
/// wall-mounted ones in between.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StairInfo {
    /// The face the full half is flush with.
    pub floor: AlignedFace,
    /// The face the step is flush with. Must not share an axis with
    /// [`floor`](Self::floor); the mesher debug-asserts it doesn't.
    pub back: AlignedFace,
}

impl StairInfo {
    /// The face the step rises from — opposite [`back`](Self::back), the
    /// one that shows a tread and a riser.
    #[inline]
    pub fn front(&self) -> AlignedFace {
        self.back.opposite()
    }

    /// The axis neither `floor` nor `back` lies on — the one the stair runs
    /// along. The two faces on it show the stair's L-shaped profile.
    #[inline]
    pub fn side_axis(&self) -> Axis {
        match (self.floor.axis(), self.back.axis()) {
            (Axis::X, Axis::Y) | (Axis::Y, Axis::X) => Axis::Z,
            (Axis::X, Axis::Z) | (Axis::Z, Axis::X) => Axis::Y,
            _ => Axis::X,
        }
    }
}

/// The geometric shape of a block, controlling quad generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// Full 1×1×1 cube.
    WholeBlock,
    /// Partial-height slab flush with one face.
    Slab(SlabInfo),
    /// A half slab with a quarter step on it, in any of 24 orientations.
    Stair(StairInfo),
    /// X-shaped diagonal billboard (e.g. sugar cane, cobwebs).
    Cross(CrossInfo),
    /// Flat quad offset inward from one face, rendered double-sided
    /// (e.g. ladders).
    Facade(FacadeInfo),
    /// Side faces inset by `n` sixteenths; top/bottom flush (e.g. cactus).
    Inset(Thickness),
    /// A fluid column whose surface follows a height field shared with
    /// its neighbors (e.g. flowing water). Requires
    /// [`Block::FLUID_ENABLED`].
    Fluid(FluidInfo),
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

    /// Whether any block of this type can return [`Shape::Fluid`].
    ///
    /// Left `false`, the surface height field is never gathered and the
    /// whole fluid path compiles out; every [`Quad`](crate::Quad) then
    /// has zero [`corner_offsets`](crate::Quad::corner_offsets). Set it
    /// and forget to, and the mesher debug-asserts rather than quietly
    /// drawing fluids as cubes.
    const FLUID_ENABLED: bool = false;

    fn shape(&self) -> Shape;

    /// The fluid standing in this cell, if any — the column the surface
    /// height field is built from.
    ///
    /// For a block whose [`shape`](Self::shape) is [`Shape::Fluid`] this is
    /// that fluid, and the default answers so. Override it to put a fluid
    /// *and* a solid in one cell — a waterlogged slab, stair or ladder —
    /// by returning `Some` from a block whose shape is something else. The
    /// mesher then emits the solid's quads as usual and the fluid's into
    /// [`Quads::fluid`](crate::Quads::fluid), culled against the solid's
    /// own faces and stitched to neighboring fluid of the same
    /// [`id`](FluidInfo::id) exactly as a plain fluid cell would be.
    ///
    /// Requires [`FLUID_ENABLED`](Self::FLUID_ENABLED), like the shape.
    #[inline]
    fn fluid(&self) -> Option<FluidInfo> {
        match self.shape() {
            Shape::Fluid(info) => Some(info),
            _ => None,
        }
    }

    fn cull_mode(&self) -> CullMode<Self::TransparentGroup>;

    /// Whether this block's material occludes ambient occlusion.
    ///
    /// This should reflect the material only (e.g. stone → true, glass
    /// → true, leaves → false, air → false). Shape-dependent logic
    /// (slabs only occlude on their flush side) is handled by the mesher.
    #[inline]
    fn ao_opaque(&self) -> bool {
        matches!(self.cull_mode(), CullMode::Opaque)
    }

    /// The light value of this voxel for smooth per-vertex lighting.
    #[inline]
    fn light(&self) -> Self::Light {
        Self::Light::default()
    }
}

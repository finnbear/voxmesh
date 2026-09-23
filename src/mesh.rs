use glam::{UVec2, UVec3, Vec2, Vec3};

use crate::block::{
    Block, CrossInfo, CullMode, FluidInfo, Shape, StairInfo, Thickness, FULL_THICKNESS,
};
use crate::chunk::{ChunkShape, PaddedChunk, PADDING};
use crate::face::{AlignedFace, Axis, DiagonalFace, Face};
use crate::light::Light;

/// A single output quad from the mesher, with per-vertex AO and light.
#[derive(Debug, Clone, PartialEq)]
pub struct Quad<L: Light = ()> {
    /// Position of the lowest-coordinate corner in 1/16ths of a block,
    /// in the padded 3D space of the chunk.
    pub origin_padded: UVec3,
    /// Size of the quad in 1/16ths of a block.
    pub size: UVec2,
    /// Per-vertex displacement along the fluid axis in 1/16ths, applied
    /// by [`positions`](Self::positions) when the quad's shape is
    /// [`Shape::Fluid`]. Vertex order matches
    /// [`positions`](Self::positions).
    ///
    /// Always `[0; 4]` for every other shape, and for every quad at all
    /// unless [`Block::FLUID_ENABLED`](crate::Block::FLUID_ENABLED).
    pub corner_offsets: [i8; 4],
    /// Per-vertex ambient occlusion (0=fully occluded, 3=fully lit).
    /// Vertex order matches [`positions`](Self::positions).
    ///
    /// Read at this quad's own corners, which for geometry that does not
    /// fill its cell are not the cell's: a stair's tread takes the value
    /// the field has halfway across the cell, not the one at the edge.
    pub ao: [u8; 4],
    /// Per-vertex averaged light values.
    /// Vertex order matches [`positions`](Self::positions), and they are
    /// read where the vertices are, as [`ao`](Self::ao) is.
    pub light: [L::Average; 4],
}

/// Returns the (u, v) tangent vectors for a face.
/// u corresponds to `size.x`, v corresponds to `size.y`.
/// Matches the block-mesh axis permutation convention (Xzy, Yzx, Zxy).
fn face_tangents(face: AlignedFace) -> (Vec3, Vec3) {
    match face {
        AlignedFace::PosX | AlignedFace::NegX => (Vec3::Z, Vec3::Y),
        AlignedFace::PosY | AlignedFace::NegY => (Vec3::Z, Vec3::X),
        AlignedFace::PosZ | AlignedFace::NegZ => (Vec3::X, Vec3::Y),
    }
}

impl<L: Light> Quad<L> {
    /// Returns the minimum voxel coordinate (excluding padding) of the
    /// block that produced this quad. Use this to look up the block type
    /// in a chunk or flat voxel array.
    ///
    /// `face` must match the face under which this quad was generated: an
    /// [`AlignedFace`] for quads from [`Quads::faces`], or a
    /// [`DiagonalFace`] for quads from [`Quads::diagonals`].
    pub fn voxel_position(&self, face: impl Into<Face>) -> UVec3 {
        let ft = FULL_THICKNESS;
        let pad = PADDING as u32;
        match face.into() {
            Face::Aligned(f) => {
                let (normal_idx, _, _) = face_axis_indices(f);
                let mut result = UVec3::ZERO;
                for axis in 0..3 {
                    let o = self.origin_padded[axis];
                    if axis == normal_idx && f.is_positive() {
                        // Positive faces sit at the far edge, step back.
                        result[axis] = (o - 1) / ft - pad;
                    } else {
                        result[axis] = o / ft - pad;
                    }
                }
                result
            }
            Face::Diagonal(_) => UVec3::new(
                self.origin_padded.x / ft - pad,
                self.origin_padded.y / ft - pad,
                self.origin_padded.z / ft - pad,
            ),
        }
    }

    /// Returns the 4 vertex positions for this quad in CCW winding order
    /// when viewed from outside.
    ///
    /// For diagonal ([`Shape::Cross`]) faces, the shape's [`CrossInfo`]
    /// determines the stretch and orientation. For axis-aligned faces,
    /// the shape is ignored.
    pub fn positions(&self, face: impl Into<Face>, shape: Shape) -> [Vec3; 4] {
        match face.into() {
            Face::Aligned(face) => {
                let scale = 1.0 / FULL_THICKNESS as f32;
                let pad = PADDING as f32;
                let base = Vec3::new(
                    self.origin_padded.x as f32 * scale - pad,
                    self.origin_padded.y as f32 * scale - pad,
                    self.origin_padded.z as f32 * scale - pad,
                );

                let (u_dir, v_dir) = face_tangents(face);
                let du = u_dir * self.size.x as f32 * scale;
                let dv = v_dir * self.size.y as f32 * scale;

                // Emit CCW winding when viewed from outside. The vertex order
                // [base, base+du, base+du+dv, base+dv] is CCW when u x v
                // aligns with the outward normal. Otherwise swap du/dv.
                let mut verts = if face.tangent_cross_positive() {
                    [base, base + du, base + du + dv, base + dv]
                } else {
                    [base, base + dv, base + dv + du, base + du]
                };

                // A fluid surface rides a height field its neighbors share,
                // so each vertex sits wherever the four columns meeting at
                // it agree it does rather than on the block boundary.
                if let Shape::Fluid(info) = shape {
                    let axis = info.face.axis().index();
                    for (vert, offset) in verts.iter_mut().zip(self.corner_offsets) {
                        vert[axis] += offset as f32 * scale;
                    }
                }

                verts
            }
            Face::Diagonal(diag) => {
                let info = match shape {
                    Shape::Cross(info) => info,
                    _ => CrossInfo {
                        face: AlignedFace::NegY,
                        stretch: 0,
                    },
                };
                let scale = 1.0 / FULL_THICKNESS as f32;
                let pad = PADDING as f32;

                // The root face axis is the merge/height axis.
                // The two perpendicular axes form the crossing plane.
                let merge_axis = info.face.axis().index();
                let (cross_a, cross_b) = cross_axes(info.face.axis());

                let origin = self.origin_padded.as_vec3() * scale - pad;

                let base_merge = origin[merge_axis];
                let height = self.size.y as f32 * scale;

                let ca = origin[cross_a] + 0.5;
                let cb = origin[cross_b] + 0.5;

                let half_diag = 0.5 + info.stretch as f32 * scale;

                // DiagonalFace direction is in the XZ plane (.x and .z).
                // Map those two components onto the crossing axes.
                let dir = diag.direction();
                let da = dir.x * half_diag;
                let db = dir.z * half_diag;

                let mut p0 = [0.0f32; 3];
                let mut p1 = [0.0f32; 3];
                let mut p2 = [0.0f32; 3];
                let mut p3 = [0.0f32; 3];

                p0[cross_a] = ca - da;
                p0[cross_b] = cb - db;
                p0[merge_axis] = base_merge;
                p1[cross_a] = ca + da;
                p1[cross_b] = cb + db;
                p1[merge_axis] = base_merge;
                p2[cross_a] = ca + da;
                p2[cross_b] = cb + db;
                p2[merge_axis] = base_merge + height;
                p3[cross_a] = ca - da;
                p3[cross_b] = cb - db;
                p3[merge_axis] = base_merge + height;

                [
                    Vec3::from_array(p0),
                    Vec3::from_array(p1),
                    Vec3::from_array(p2),
                    Vec3::from_array(p3),
                ]
            }
        }
    }

    /// Returns the 4 texture coordinates for this quad.
    ///
    /// `u_flip_face` and `flip_v` control UV mirroring for axis-aligned
    /// faces and are ignored for diagonal faces.
    ///
    /// `shape` is used to follow a [`Shape::Fluid`] surface: the side
    /// face of a half-full fluid is a trapezoid, and its texture is
    /// cropped to match rather than squashed into it.
    pub fn texture_coordinates(
        &self,
        face: impl Into<Face>,
        shape: Shape,
        u_flip_face: Axis,
        flip_v: bool,
    ) -> [Vec2; 4] {
        match face.into() {
            Face::Aligned(face) => {
                let scale = 1.0 / FULL_THICKNESS as f32;
                let (_, u_idx, v_idx) = face_axis_indices(face);
                let u_off = (self.origin_padded[u_idx] % FULL_THICKNESS) as f32 * scale;
                let v_off = (self.origin_padded[v_idx] % FULL_THICKNESS) as f32 * scale;
                let u_size = self.size.x as f32 * scale;
                let v_size = self.size.y as f32 * scale;
                // Flip mirrors about the full block (1.0) for sub-block
                // quads, or about the merged extent for merged quads.
                let u_extent = 1.0f32.max(u_size);
                let v_extent = 1.0f32.max(v_size);

                let flip_u = if face.is_positive() {
                    face.axis() == u_flip_face
                } else {
                    face.axis() != u_flip_face
                };

                let mut raw = if face.tangent_cross_positive() {
                    [
                        Vec2::new(u_off, v_off),
                        Vec2::new(u_off + u_size, v_off),
                        Vec2::new(u_off + u_size, v_off + v_size),
                        Vec2::new(u_off, v_off + v_size),
                    ]
                } else {
                    [
                        Vec2::new(u_off, v_off),
                        Vec2::new(u_off, v_off + v_size),
                        Vec2::new(u_off + u_size, v_off + v_size),
                        Vec2::new(u_off + u_size, v_off),
                    ]
                };

                // Track a fluid surface, so a side face crops its texture
                // instead of stretching it. On the surface face itself the
                // fluid axis is the normal, so nothing moves. Applied
                // before the flip, which mirrors the finished coordinate.
                if let Shape::Fluid(info) = shape {
                    let fluid_idx = info.face.axis().index();
                    let component = if fluid_idx == u_idx {
                        Some(0)
                    } else if fluid_idx == v_idx {
                        Some(1)
                    } else {
                        None
                    };
                    if let Some(component) = component {
                        for (uv, offset) in raw.iter_mut().zip(self.corner_offsets) {
                            uv[component] += offset as f32 * scale;
                        }
                    }
                }

                raw.map(|uv| {
                    Vec2::new(
                        if flip_u { u_extent - uv.x } else { uv.x },
                        if flip_v { v_extent - uv.y } else { uv.y },
                    )
                })
            }
            Face::Diagonal(_) => {
                let v_size = self.size.y as f32 * (1.0 / FULL_THICKNESS as f32);
                let (v_lo, v_hi) = if flip_v { (v_size, 0.0) } else { (0.0, v_size) };
                [
                    Vec2::new(0.0, v_lo),
                    Vec2::new(1.0, v_lo),
                    Vec2::new(1.0, v_hi),
                    Vec2::new(0.0, v_hi),
                ]
            }
        }
    }

    /// Returns the 6 vertex indices for this quad (two triangles),
    /// suitable for indexed drawing.
    ///
    /// `start` is the index of the first vertex of this quad in the
    /// vertex buffer. The returned indices reference vertices in the
    /// order produced by [`positions`](Self::positions), which is always
    /// CCW when viewed from outside.
    ///
    /// The winding is compatible with block-mesh-rs `quad_mesh_indices`.
    #[inline]
    pub fn indices(start: u32) -> [u32; 6] {
        [start, start + 1, start + 2, start, start + 2, start + 3]
    }

    /// Returns the 6 vertex indices with the triangle diagonal chosen to
    /// maximize the sum of AO values along the diagonal. Equivalent to
    /// [`indices_optimal`](Self::indices_optimal) with a key of AO alone.
    #[inline]
    pub fn indices_ao(&self, start: u32) -> [u32; 6] {
        self.indices_optimal(start, |ao, _light| ao as u16)
    }

    /// Returns the 6 vertex indices with the triangle diagonal chosen
    /// to improve interpolation of some combination of AO and lighting.
    #[inline]
    pub fn indices_optimal<K>(
        &self,
        start: u32,
        mut key: impl FnMut(u8, L::Average) -> K,
    ) -> [u32; 6]
    where
        K: PartialOrd + std::ops::Add<Output = K>,
    {
        let [k0, k1, k2, k3] = std::array::from_fn(|i| key(self.ao[i], self.light[i]));
        if k0 + k2 >= k1 + k3 {
            [start, start + 1, start + 2, start, start + 2, start + 3]
        } else {
            [start, start + 1, start + 3, start + 1, start + 2, start + 3]
        }
    }
}

/// Output of the mesher: quads grouped by face direction.
pub struct Quads<L: Light = ()> {
    /// Axis-aligned quads indexed by [`AlignedFace`].
    pub faces: [Vec<Quad<L>>; 6],
    /// Diagonal quads for X-shaped cross blocks, indexed by [`DiagonalFace`].
    pub diagonals: [Vec<Quad<L>>; 2],
    /// The fluid standing in cells whose shape is something else — see
    /// [`Block::fluid`] — indexed by [`AlignedFace`]. Kept apart from
    /// [`faces`](Self::faces) because the consumer draws them as the fluid
    /// and the cell's own quads as the solid, and
    /// [`voxel_position`](Quad::voxel_position) leads to a cell holding
    /// both. Empty unless [`Block::FLUID_ENABLED`].
    pub fluid: [Vec<Quad<L>>; 6],
}

// Greedy meshing internals

#[derive(Clone, Copy, PartialEq)]
struct MaskEntry<B: Block> {
    block: B,
    /// Face surface position along the normal axis in 1/16ths from the
    /// block min-normal coordinate. Whole-block positive face = 16,
    /// negative = 0.
    normal_pos: u8,
    /// Quad start within the block along u, in 1/16ths.
    u_intra_offset: u8,
    /// Quad extent within one block cell along u, in 1/16ths.
    u_intra_extent: u8,
    /// Quad start within the block along v, in 1/16ths.
    v_intra_offset: u8,
    /// Quad extent within one block cell along v, in 1/16ths.
    v_intra_extent: u8,
    /// Per-vertex fluid displacement in mask-local order. Part of the
    /// entry's identity, so greedy merging never spans a slope — see
    /// the note on the emit phase in [`mesh_chunk_into`].
    corner_offsets: [i8; 4],
    /// Per-vertex AO in mask-local order: [umin/vmin, umax/vmin,
    /// umax/vmax, umin/vmax] — of the entry's own rectangle, not of the
    /// cell, so two entries that cover different parts of one cell carry
    /// different values and merging stays honest. See
    /// [`resample_to_rect`].
    ao: [u8; 4],
    /// Per-vertex light in mask-local order, read where [`Self::ao`] is.
    light: [<<B as Block>::Light as Light>::Average; 4],
}

/// Returns the two axis indices perpendicular to the given axis,
/// used by diagonal cross blocks to determine the crossing plane.
#[inline]
fn cross_axes(axis: Axis) -> (usize, usize) {
    match axis {
        Axis::X => (1, 2), // cross in YZ
        Axis::Y => (0, 2), // cross in XZ
        Axis::Z => (0, 1), // cross in XY
    }
}

/// Returns the (normal_idx, u_idx, v_idx) axis indices for a face, matching
/// the tangent convention in [`face_tangents`].
#[inline]
fn face_axis_indices(face: AlignedFace) -> (usize, usize, usize) {
    match face {
        AlignedFace::PosX | AlignedFace::NegX => (0, 2, 1), // normal=X, u=Z, v=Y
        AlignedFace::PosY | AlignedFace::NegY => (1, 2, 0), // normal=Y, u=Z, v=X
        AlignedFace::PosZ | AlignedFace::NegZ => (2, 0, 1), // normal=Z, u=X, v=Y
    }
}

/// A rectangle on a face plane, in 1/16ths along that face's `(u, v)`
/// tangents ([`face_axis_indices`]). Half-open: `u0 <= u < u1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Rect16 {
    u0: u8,
    u1: u8,
    v0: u8,
    v1: u8,
}

impl Rect16 {
    const FULL: Rect16 = Rect16 {
        u0: 0,
        u1: FULL_THICKNESS as u8,
        v0: 0,
        v1: FULL_THICKNESS as u8,
    };

    #[inline]
    fn is_full(&self) -> bool {
        *self == Self::FULL
    }

    #[inline]
    fn contains(&self, other: &Rect16) -> bool {
        self.u0 <= other.u0 && other.u1 <= self.u1 && self.v0 <= other.v0 && other.v1 <= self.v1
    }

    /// A strip that is full along one tangent and `[lo, hi)` along the
    /// other, the axis of the strip given as an index into `(u, v)`.
    #[inline]
    fn strip(along_v: bool, lo: u8, hi: u8) -> Rect16 {
        if along_v {
            Rect16 {
                u0: 0,
                u1: FULL_THICKNESS as u8,
                v0: lo,
                v1: hi,
            }
        } else {
            Rect16 {
                u0: lo,
                u1: hi,
                v0: 0,
                v1: FULL_THICKNESS as u8,
            }
        }
    }
}

/// The part of `face`'s boundary plane that `block` is flush against —
/// up to two rectangles, since a stair's side is an L.
///
/// This is the one place a shape says what it seals, and everything that
/// asks "is there something solid on the other side of this face" reads
/// it: culling (does the neighbor hide this quad), ambient occlusion (does
/// the neighbor darken this vertex). A new shape joins by answering here.
#[inline]
fn boundary_footprint<B: Block>(block: &B, face: AlignedFace) -> [Option<Rect16>; 2] {
    let ft = FULL_THICKNESS as u8;
    let (_, u_idx, v_idx) = face_axis_indices(face);
    match block.shape() {
        Shape::WholeBlock => [Some(Rect16::FULL), None],
        // Cross and facade blocks never cover any face region.
        Shape::Cross(_) | Shape::Facade(_) => [None; 2],
        // Inset blocks cover top/bottom (flush) but not sides (inset).
        Shape::Inset(_) => {
            if face.axis() == Axis::Y {
                [Some(Rect16::FULL), None]
            } else {
                [None; 2]
            }
        }
        // The declared height: a fluid seals a face only when it fills its
        // cell. A neighbor of the same fluid is handled by the caller,
        // which stitches to it whatever the height.
        Shape::Fluid(info) => {
            if info.height >= FULL_THICKNESS {
                [Some(Rect16::FULL), None]
            } else {
                [None; 2]
            }
        }
        Shape::Slab(info) => {
            let axis = info.face.axis().index();
            let thickness = info.thickness as u8;
            if face == info.face {
                [Some(Rect16::FULL), None]
            } else if face.axis() == info.face.axis() {
                // The inner face is inset, never at the boundary.
                [None; 2]
            } else {
                let (lo, hi) = if info.face.is_positive() {
                    (ft - thickness, ft)
                } else {
                    (0, thickness)
                };
                [Some(Rect16::strip(axis == v_idx, lo, hi)), None]
            }
        }
        Shape::Stair(info) => {
            debug_assert_ne!(
                info.floor.axis(),
                info.back.axis(),
                "a stair's floor and back must be perpendicular"
            );
            let half = ft / 2;
            let floor_axis = info.floor.axis().index();
            let back_axis = info.back.axis().index();
            // The floor half and the far half along the floor axis, and
            // the back half along the back axis, as 1/16th ranges.
            let (floor_lo, floor_hi, far_lo, far_hi) = if info.floor.is_positive() {
                (half, ft, 0, half)
            } else {
                (0, half, half, ft)
            };
            let (back_lo, back_hi) = if info.back.is_positive() {
                (half, ft)
            } else {
                (0, half)
            };
            if face == info.floor || face == info.back {
                // The slab fills the floor face; the slab and the step
                // between them fill the back face.
                [Some(Rect16::FULL), None]
            } else if face == info.floor.opposite() {
                // The step's top, on the back half.
                [
                    Some(Rect16::strip(back_axis == v_idx, back_lo, back_hi)),
                    None,
                ]
            } else if face == info.back.opposite() {
                // The slab's front, on the floor half.
                [
                    Some(Rect16::strip(floor_axis == v_idx, floor_lo, floor_hi)),
                    None,
                ]
            } else {
                // A side: the slab's strip and the step's quarter, an L.
                let slab = Rect16::strip(floor_axis == v_idx, floor_lo, floor_hi);
                let step = if floor_axis == u_idx {
                    Rect16 {
                        u0: far_lo,
                        u1: far_hi,
                        v0: back_lo,
                        v1: back_hi,
                    }
                } else {
                    Rect16 {
                        u0: back_lo,
                        u1: back_hi,
                        v0: far_lo,
                        v1: far_hi,
                    }
                };
                [Some(slab), Some(step)]
            }
        }
    }
}

/// Whether `footprint` covers all of `rect`: inside one of its rectangles,
/// or — the footprint being an L — inside their union with the split
/// running straight across. Anything cleverer than that is answered `false`,
/// which costs a quad that could have been culled and never hides one that
/// should show.
#[inline]
fn footprint_covers(footprint: &[Option<Rect16>; 2], rect: &Rect16) -> bool {
    match footprint {
        [None, _] => false,
        [Some(a), None] => a.contains(rect),
        [Some(a), Some(b)] => {
            if a.contains(rect) || b.contains(rect) {
                return true;
            }
            // Take what `a` covers of `rect` off it; if what is left is one
            // rectangle, ask `b` for that.
            let rest = if a.v0 <= rect.v0 && rect.v1 <= a.v1 {
                // `a` spans `rect` in v: it may cut off a u-end.
                if a.u0 <= rect.u0 && a.u1 > rect.u0 && a.u1 < rect.u1 {
                    Some(Rect16 { u0: a.u1, ..*rect })
                } else if a.u1 >= rect.u1 && a.u0 < rect.u1 && a.u0 > rect.u0 {
                    Some(Rect16 { u1: a.u0, ..*rect })
                } else {
                    None
                }
            } else if a.u0 <= rect.u0 && rect.u1 <= a.u1 {
                if a.v0 <= rect.v0 && a.v1 > rect.v0 && a.v1 < rect.v1 {
                    Some(Rect16 { v0: a.v1, ..*rect })
                } else if a.v1 >= rect.v1 && a.v0 < rect.v1 && a.v0 > rect.v0 {
                    Some(Rect16 { v1: a.v0, ..*rect })
                } else {
                    None
                }
            } else {
                None
            };
            rest.is_some_and(|rest| b.contains(&rest))
        }
    }
}

/// The part of the face plane `footprint` leaves open, as one rectangle:
/// the whole face for an empty footprint, nothing for a full one, the other
/// strip beside a strip, the open quadrant of a stair's L. `None` when
/// nothing is open.
///
/// Computed on the grid the rectangles' edges cut the face into, so it is
/// exact for anything whose complement *is* one rectangle — which every
/// shape here satisfies, the footprints all being anchored to the face's
/// edges. A footprint whose complement is more than one rectangle (none
/// exists) would be answered with the first, and debug-asserts.
fn uncovered(footprint: &[Option<Rect16>; 2]) -> Option<Rect16> {
    let ft = FULL_THICKNESS as u8;
    let mut us = [0u8, ft, ft, ft, ft, ft];
    let mut vs = [0u8, ft, ft, ft, ft, ft];
    let mut n = 2;
    for rect in footprint.iter().flatten() {
        us[n] = rect.u0;
        us[n + 1] = rect.u1;
        vs[n] = rect.v0;
        vs[n + 1] = rect.v1;
        n += 2;
    }
    us.sort_unstable();
    vs.sort_unstable();
    let covered = |u: u8, v: u8| {
        footprint
            .iter()
            .flatten()
            .any(|r| r.u0 <= u && u < r.u1 && r.v0 <= v && v < r.v1)
    };
    let mut open: Option<Rect16> = None;
    for i in 0..us.len() - 1 {
        for j in 0..vs.len() - 1 {
            let (u0, u1, v0, v1) = (us[i], us[i + 1], vs[j], vs[j + 1]);
            if u0 == u1 || v0 == v1 || covered(u0, v0) {
                continue;
            }
            open = Some(match open {
                None => Rect16 { u0, u1, v0, v1 },
                Some(r) => {
                    // The open cells so far and this one must together be
                    // a rectangle: extend the bounds, and check the result
                    // holds no covered cell.
                    let grown = Rect16 {
                        u0: r.u0.min(u0),
                        u1: r.u1.max(u1),
                        v0: r.v0.min(v0),
                        v1: r.v1.max(v1),
                    };
                    debug_assert!(
                        !covered(grown.u0, grown.v0)
                            && !covered(grown.u1 - 1, grown.v0)
                            && !covered(grown.u0, grown.v1 - 1)
                            && !covered(grown.u1 - 1, grown.v1 - 1),
                        "a footprint whose complement is not one rectangle"
                    );
                    grown
                }
            });
        }
    }
    open
}

/// Whether the neighbor fully covers `rect` of the block's `face` on the
/// shared boundary.
#[inline]
fn neighbor_covers_face_region<B: Block>(
    block: &B,
    neighbor: &B,
    face: AlignedFace,
    rect: &Rect16,
) -> bool {
    // The same fluid stitches its surface onto ours, so there is never a
    // gap behind the shared face however deep either column happens to
    // be — and a fluid standing in a solid's cell counts, since it is the
    // same body of water. Only for a fluid's *own* quads, though: a
    // waterlogged slab's stone still has to show where the neighbor's
    // stone doesn't reach.
    if let (Shape::Fluid(b_info), Some(n_info)) = (block.shape(), neighbor.fluid()) {
        if b_info.id == n_info.id {
            return true;
        }
    }
    // The neighbor's footprint on its side of the plane, which lies on the
    // same (u, v) axes as ours: opposite faces share tangents.
    footprint_covers(&boundary_footprint(neighbor, face.opposite()), rect)
}

/// Whether the current block's face is culled by the given neighbor.
/// Only valid for faces at the block boundary (flush or side); `rect` is
/// the part of the face the quad in question occupies.
#[inline]
fn is_culled_at_boundary<B: Block>(
    block: &B,
    neighbor: &B,
    face: AlignedFace,
    rect: &Rect16,
) -> bool {
    if !neighbor_covers_face_region(block, neighbor, face, rect) {
        return false;
    }
    match (block.cull_mode(), neighbor.cull_mode()) {
        (_, CullMode::Opaque) => true,
        (CullMode::TransparentMerged(a), CullMode::TransparentMerged(b)) => a == b,
        // Unmerged transparent: cull the negative face so only one
        // face is emitted per boundary, avoiding z-fighting.
        (CullMode::TransparentUnmerged, CullMode::TransparentUnmerged) => !face.is_positive(),
        _ => false,
    }
}

/// The rectangle a mask entry occupies on its face, in `(u, v)`.
#[inline]
fn entry_rect<B: Block>(entry: &MaskEntry<B>) -> Rect16 {
    Rect16 {
        u0: entry.u_intra_offset,
        u1: entry.u_intra_offset + entry.u_intra_extent,
        v0: entry.v_intra_offset,
        v1: entry.v_intra_offset + entry.v_intra_extent,
    }
}

/// A mask entry with the given plane depth and `(u, v)` rectangle, and no
/// lighting yet.
#[inline]
fn entry_at<B: Block>(block: &B, normal_pos: u8, rect: Rect16) -> MaskEntry<B> {
    MaskEntry {
        block: *block,
        normal_pos,
        u_intra_offset: rect.u0,
        u_intra_extent: rect.u1 - rect.u0,
        v_intra_offset: rect.v0,
        v_intra_extent: rect.v1 - rect.v0,
        corner_offsets: [0; 4],
        ao: [3; 4],
        light: Default::default(),
    }
}

/// The up-to-two quads a stair shows on `face`, by shape alone — no
/// neighbor culling. Each is `(entry, at_boundary)`: the boundary ones are
/// the caller's to cull, the inset ones (the tread's inner half and the
/// riser) never are.
///
/// The stair is its two boxes, the slab and the step, and each face is
/// what the two of them together present to it:
///
/// - the floor face and the back face are full, at the boundary;
/// - opposite the floor, the slab's exposed top (inset, front half) and
///   the step's top (boundary, back half);
/// - opposite the back, the slab's front (boundary, floor half) and the
///   riser (inset, far half);
/// - the two sides, an L at the boundary: the slab's strip and the step's
///   quarter.
#[inline]
fn stair_entries<B: Block>(
    block: &B,
    info: StairInfo,
    face: AlignedFace,
    u_idx: usize,
    v_idx: usize,
) -> [Option<(MaskEntry<B>, bool)>; 2] {
    let ft = FULL_THICKNESS as u8;
    let half = ft / 2;
    let floor_axis = info.floor.axis().index();
    let back_axis = info.back.axis().index();
    let (floor_lo, floor_hi, far_lo, far_hi) = if info.floor.is_positive() {
        (half, ft, 0, half)
    } else {
        (0, half, half, ft)
    };
    let (back_lo, back_hi, front_lo, front_hi) = if info.back.is_positive() {
        (half, ft, 0, half)
    } else {
        (0, half, half, ft)
    };
    let boundary = if face.is_positive() { ft } else { 0 };
    // The plane halfway along an axis, seen from `face`'s side.
    let inset = half;
    if face == info.floor || face == info.back {
        [Some((entry_at(block, boundary, Rect16::FULL), true)), None]
    } else if face == info.floor.opposite() {
        let along_v = back_axis == v_idx;
        [
            Some((
                entry_at(block, inset, Rect16::strip(along_v, front_lo, front_hi)),
                false,
            )),
            Some((
                entry_at(block, boundary, Rect16::strip(along_v, back_lo, back_hi)),
                true,
            )),
        ]
    } else if face == info.back.opposite() {
        let along_v = floor_axis == v_idx;
        [
            Some((
                entry_at(block, boundary, Rect16::strip(along_v, floor_lo, floor_hi)),
                true,
            )),
            Some((
                entry_at(block, inset, Rect16::strip(along_v, far_lo, far_hi)),
                false,
            )),
        ]
    } else {
        let slab = Rect16::strip(floor_axis == v_idx, floor_lo, floor_hi);
        let step = if floor_axis == u_idx {
            Rect16 {
                u0: far_lo,
                u1: far_hi,
                v0: back_lo,
                v1: back_hi,
            }
        } else {
            Rect16 {
                u0: back_lo,
                u1: back_hi,
                v0: far_lo,
                v1: far_hi,
            }
        };
        [
            Some((entry_at(block, boundary, slab), true)),
            Some((entry_at(block, boundary, step), true)),
        ]
    }
}

/// Compute the mask entry for a block/face based purely on shape,
/// ignoring neighbor culling. Returns `None` for faces that never
/// emit geometry (cross blocks, non-matching facades, etc.). A stair
/// has up to two; this is the first, and [`stair_entries`] has both.
#[inline]
fn mask_entry_for_shape<B: Block>(
    block: &B,
    face: AlignedFace,
    u_idx: usize,
    v_idx: usize,
) -> Option<MaskEntry<B>> {
    let ft = FULL_THICKNESS as u8;
    match block.shape() {
        // Cross blocks have no axis-aligned faces.
        Shape::Cross(_) => return None,
        Shape::Stair(info) => {
            return stair_entries(block, info, face, u_idx, v_idx)[0].map(|(entry, _)| entry)
        }
        // Facade emits one quad on its own face, offset `info.offset`
        // sixteenths inward.
        Shape::Facade(info) => {
            if face != info.face {
                return None;
            }
            let normal_pos = if face.is_positive() {
                ft - info.offset
            } else {
                info.offset
            };
            return Some(MaskEntry {
                block: *block,
                normal_pos,
                u_intra_offset: 0,
                u_intra_extent: ft,
                v_intra_offset: 0,
                v_intra_extent: ft,
                corner_offsets: [0; 4],
                ao: [3; 4],
                light: Default::default(),
            });
        }
        // A fluid with no neighbors to share a surface with is just its
        // cell: a held bucket of water is a cube, not a puddle. The
        // chunk mesher never reaches this arm; it has the neighborhood
        // and uses `compute_fluid_mask_entry` instead.
        Shape::WholeBlock | Shape::Fluid(_) => {
            let normal_pos = if face.is_positive() { ft } else { 0 };
            Some(MaskEntry {
                block: *block,
                normal_pos,
                u_intra_offset: 0,
                u_intra_extent: ft,
                v_intra_offset: 0,
                v_intra_extent: ft,
                corner_offsets: [0; 4],
                ao: [3; 4],
                light: Default::default(),
            })
        }
        Shape::Inset(n) => {
            // Side faces are inset, top/bottom are flush.
            let normal_pos = if face.axis() == Axis::Y {
                if face.is_positive() {
                    ft
                } else {
                    0
                }
            } else {
                if face.is_positive() {
                    ft - n as u8
                } else {
                    n as u8
                }
            };
            Some(MaskEntry {
                block: *block,
                normal_pos,
                u_intra_offset: 0,
                u_intra_extent: ft,
                v_intra_offset: 0,
                v_intra_extent: ft,
                corner_offsets: [0; 4],
                ao: [3; 4],
                light: Default::default(),
            })
        }
        Shape::Slab(info) => {
            let slab_axis_idx = info.face.axis().index();
            let thickness = info.thickness as u8;

            let (slab_min, slab_max) = if info.face.is_positive() {
                (ft - thickness, ft)
            } else {
                (0, thickness)
            };

            if face.axis() == info.face.axis() {
                // The inner face of a slab (opposite its flush face) is
                // never at the block boundary, so it always emits
                // geometry. The flush face may be culled by a neighbor
                // but that is handled by the caller.
                let normal_pos = if face.is_positive() {
                    slab_max
                } else {
                    slab_min
                };
                Some(MaskEntry {
                    block: *block,
                    normal_pos,
                    u_intra_offset: 0,
                    u_intra_extent: ft,
                    v_intra_offset: 0,
                    v_intra_extent: ft,
                    corner_offsets: [0; 4],
                    ao: [3; 4],
                    light: Default::default(),
                })
            } else {
                let normal_pos = if face.is_positive() { ft } else { 0 };

                let (u_off, u_ext, v_off, v_ext) = if slab_axis_idx == u_idx {
                    (slab_min, thickness, 0, ft)
                } else {
                    debug_assert_eq!(slab_axis_idx, v_idx);
                    (0, ft, slab_min, thickness)
                };

                Some(MaskEntry {
                    block: *block,
                    normal_pos,
                    u_intra_offset: u_off,
                    u_intra_extent: u_ext,
                    v_intra_offset: v_off,
                    v_intra_extent: v_ext,
                    corner_offsets: [0; 4],
                    ao: [3; 4],
                    light: Default::default(),
                })
            }
        }
    }
}

/// Compute the mask entry for a slab block/face combination, or `None`
/// if the face is not visible. Only called when `block.shape()` is `Slab`.
#[inline]
fn compute_slab_mask_entry<B: Block>(
    block: &B,
    neighbor: &B,
    face: AlignedFace,
    u_idx: usize,
    v_idx: usize,
) -> Option<MaskEntry<B>> {
    let info = match block.shape() {
        Shape::Slab(info) => info,
        Shape::WholeBlock
        | Shape::Cross(_)
        | Shape::Facade(_)
        | Shape::Inset(_)
        | Shape::Stair(_)
        | Shape::Fluid(_) => unreachable!(),
    };

    let entry = mask_entry_for_shape(block, face, u_idx, v_idx)?;
    // The flush face and the sides are at the boundary and the neighbor
    // may hide them; the inner face is inset and never is.
    let at_boundary = face == info.face || face.axis() != info.face.axis();
    if at_boundary && is_culled_at_boundary(block, neighbor, face, &entry_rect(&entry)) {
        return None;
    }
    Some(entry)
}

/// The [`FluidInfo`] of the block at `idx`, or `None` if it is not a
/// fluid of `id`.
///
/// # Safety
///
/// `idx` must be within `data`.
#[inline]
unsafe fn fluid_at<B: Block>(data: &[B], idx: usize, id: u8) -> Option<FluidInfo> {
    debug_assert!(idx < data.len(), "fluid neighborhood escaped the chunk");
    // Through `fluid`, not `shape`: a waterlogged cell is part of the
    // body of water for every purpose the height field has.
    unsafe { data.get_unchecked(idx) }
        .fluid()
        .filter(|info| info.id == id)
}

/// How much of the cell at `idx` the fluid `id` occupies, in 1/16ths, or
/// `None` if that cell is not part of the fluid at all.
///
/// A column with more of the same fluid stacked on it is full whatever
/// it claims: there is nothing above for its surface to be a surface of.
/// This is the rule a block cannot apply for itself, since
/// [`Block::shape`] cannot see its neighbor.
///
/// # Safety
///
/// `idx` and `idx + up_stride` must be within `data`.
#[inline]
unsafe fn fluid_height<B: Block>(
    data: &[B],
    idx: usize,
    id: u8,
    up_stride: isize,
) -> Option<Thickness> {
    let info = unsafe { fluid_at(data, idx, id) }?;
    let above = (idx as isize + up_stride) as usize;
    if unsafe { fluid_at(data, above, id) }.is_some() {
        Some(FULL_THICKNESS)
    } else {
        Some(info.height.min(FULL_THICKNESS))
    }
}

/// Index of the surface corner at the given signs along the two axes
/// perpendicular to the fluid axis, in the mask-local order
/// `[min/min, max/min, max/max, min/max]`.
#[inline]
fn corner_index(pos_a: bool, pos_b: bool) -> usize {
    match (pos_a, pos_b) {
        (false, false) => 0,
        (true, false) => 1,
        (true, true) => 2,
        (false, true) => 3,
    }
}

/// Surface heights in 1/16ths at the 4 corners of the cell at `idx`,
/// indexed by [`corner_index`] over the axes `a_stride` and `b_stride`.
///
/// Each corner takes the tallest of the four columns meeting at it,
/// skipping those that hold no fluid. Because that depends on nothing
/// but the 2×2 of columns the corner is shared with, every cell touching
/// a corner computes the same height for it — which is what lets
/// neighboring fluid surfaces meet exactly, and in turn what lets the
/// shared face between them be culled without leaving a crack.
///
/// # Safety
///
/// The 3×3 of columns around `idx`, and one step along `up_stride` from
/// each, must be within `data`. The padding ring guarantees this for any
/// cell the mesher visits.
unsafe fn fluid_corner_heights<B: Block>(
    data: &[B],
    idx: usize,
    id: u8,
    up_stride: isize,
    a_stride: isize,
    b_stride: isize,
) -> [Thickness; 4] {
    // The 3x3 of columns around this one, indexed [b + 1][a + 1].
    let mut heights = [[0 as Thickness; 3]; 3];
    for (b, row) in heights.iter_mut().enumerate() {
        for (a, cell) in row.iter_mut().enumerate() {
            let offset = (a as isize - 1) * a_stride + (b as isize - 1) * b_stride;
            *cell = unsafe { fluid_height(data, (idx as isize + offset) as usize, id, up_stride) }
                .unwrap_or(0);
        }
    }
    // Corner `(a, b)` is the max over the 2x2 at [b..b+2][a..a+2]. The
    // cell itself is heights[1][1], which every corner includes, so no
    // corner is ever left at zero.
    let corner = |a: usize, b: usize| {
        heights[b][a]
            .max(heights[b][a + 1])
            .max(heights[b + 1][a])
            .max(heights[b + 1][a + 1])
    };
    [corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1)]
}

/// Compute the mask entry for a fluid block/face combination, or `None`
/// if the face is not visible. Only called when `block.shape()` is
/// `Fluid` and [`Block::FLUID_ENABLED`].
///
/// # Safety
///
/// `idx` must be an inner (non-padding) cell of `data`, so that the 3×3
/// of columns around it and one step along the fluid axis are in bounds.
#[allow(clippy::too_many_arguments)]
unsafe fn compute_fluid_mask_entry<B: Block>(
    data: &[B],
    idx: usize,
    block: &B,
    neighbor: &B,
    info: FluidInfo,
    face: AlignedFace,
    normal_idx: usize,
    u_idx: usize,
    v_idx: usize,
    padded: usize,
) -> Option<MaskEntry<B>> {
    let ft = FULL_THICKNESS as u8;
    let up_idx = info.face.axis().index();
    let axis_strides = [1isize, padded as isize, (padded * padded) as isize];
    let up_stride = if info.face.is_positive() {
        axis_strides[up_idx]
    } else {
        -axis_strides[up_idx]
    };

    // A fluid standing in a solid's cell — the overlay — is hidden by the
    // solid's own faces before it is hidden by anything next door: the
    // water in a bottom slab has no underside to draw. And it is culled
    // by the neighbor on the fluid's terms, not the solid's: the same
    // fluid next door joins it, an opaque neighbor that seals the face
    // hides it, and nothing else does. The solid's cull mode would say
    // "opaque against transparent, draw" and put a water face inside the
    // pond.
    let overlay = !matches!(block.shape(), Shape::Fluid(_));
    // The part of the face the fluid shows: all of it for a fluid cell,
    // and for water in a solid's cell whatever the solid leaves open — the
    // strip above a slab, the quadrant beside a stair's step. Clipped
    // rather than drawn whole and hidden by depth, so the water never
    // shares a plane with the solid's own quad.
    let open = if overlay {
        uncovered(&boundary_footprint(block, face))?
    } else {
        Rect16::FULL
    };
    let culled = |face: AlignedFace| {
        // The same fluid next door joins this one whatever else is sharing
        // its cell, so there is never a surface between the two — and that
        // holds for a plain fluid cell just as much as for an overlay.
        //
        // Hoisted out of the `overlay` arm for exactly that reason. The
        // pond's own quads went to `is_culled_at_boundary`, which ends by
        // asking the two *blocks* whether they merge, and water against a
        // waterlogged ladder answers `TransparentMerged(Water)` against
        // `TransparentMerged(Ladder)` — no match, so it drew a pane of water
        // between two cells of one pond. A waterlogged *slab* hid the bug for
        // as long as it lasted, its stone being opaque and culling on the
        // arm above.
        if neighbor.fluid().is_some_and(|n| n.id == info.id) {
            return true;
        }
        if overlay {
            matches!(neighbor.cull_mode(), CullMode::Opaque)
                && footprint_covers(&boundary_footprint(neighbor, face.opposite()), &open)
        } else {
            is_culled_at_boundary(block, neighbor, face, &Rect16::FULL)
        }
    };
    if face == info.face {
        // The surface sits inside the block whenever the column is not
        // full, so whatever is above cannot hide it: water at half
        // height under a stone ceiling is visible through the gap. Only
        // a full column is flush enough to be culled at the boundary.
        let full = unsafe { fluid_height(data, idx, info.id, up_stride) }
            .is_some_and(|height| height >= FULL_THICKNESS);
        if full && culled(face) {
            return None;
        }
    } else if culled(face) {
        return None;
    }

    let mut entry = entry_at(block, if face.is_positive() { ft } else { 0 }, open);

    // The face opposite the surface is the floor of the cell, flat on
    // the block boundary however shallow the column is.
    if face == info.face.opposite() {
        return Some(entry);
    }

    let (a_idx, b_idx) = cross_axes(info.face.axis());
    let heights = unsafe {
        fluid_corner_heights(
            data,
            idx,
            info.id,
            up_stride,
            axis_strides[a_idx],
            axis_strides[b_idx],
        )
    };
    // Depth is measured from the surface face inward, so a shorter
    // column pulls its vertices away from that face.
    let sign = if info.face.is_positive() { 1 } else { -1 };

    for (vertex, offset) in entry.corner_offsets.iter_mut().enumerate() {
        // Where this mask-local vertex sits on each axis.
        let mut positive = [false; 3];
        positive[u_idx] = vertex == 1 || vertex == 2;
        positive[v_idx] = vertex == 2 || vertex == 3;
        positive[normal_idx] = face.is_positive();

        // On a side face only the two vertices at the surface end ride
        // the height field; the other two stay on the cell floor. On the
        // surface face itself the normal is the fluid axis, so all four
        // ride it.
        if face.axis().index() != up_idx && positive[up_idx] != info.face.is_positive() {
            continue;
        }

        let height = heights[corner_index(positive[a_idx], positive[b_idx])] as i32;
        *offset = ((height - FULL_THICKNESS as i32) * sign) as i8;
    }

    Some(entry)
}

/// Whether a block's shape fills the given face (whole blocks always,
/// slabs only on their flush face, others never).
#[inline]
fn shape_fills_face<B: Block>(block: &B, face: AlignedFace) -> bool {
    // The footprint's declared coverage — for a fluid, the declared
    // height, not the effective one: this is only consulted for AO, where
    // the neighbor context needed to raise a submerged column is not on
    // hand and the difference is a shade.
    boundary_footprint(block, face)[0].is_some_and(|rect| rect.is_full())
}

/// Whether a block occludes AO on the given face: material is
/// AO-opaque and shape fills that face.
#[inline]
fn shape_ao_opaque<B: Block>(block: &B, face: AlignedFace) -> bool {
    block.ao_opaque() && shape_fills_face(block, face)
}

/// Computes per-vertex AO and smooth light for a face cell.
///
/// `data` is the padded chunk array. `n_idx` is the linear index of the
/// voxel one step along the face normal from the current block.
/// `u_stride` and `v_stride` are the linear index steps along the face's
/// tangent axes. `face` is the face being meshed.
///
/// Returns `(ao, light)` arrays in mask-local vertex order:
/// `[umin/vmin, umax/vmin, umax/vmax, umin/vmax]`.
#[inline]
fn compute_ao_light<B: Block>(
    data: &[B],
    n_idx: usize,
    u_stride: isize,
    v_stride: isize,
    ao_face: AlignedFace,
) -> ([u8; 4], [<B::Light as Light>::Average; 4]) {
    // Load all 9 neighbors in the face-normal plane once.
    let get = |du: isize, dv: isize| -> &B {
        unsafe { data.get_unchecked((n_idx as isize + du + dv) as usize) }
    };

    let center = get(0, 0);
    let neg_u = get(-u_stride, 0);
    let pos_u = get(u_stride, 0);
    let neg_v = get(0, -v_stride);
    let pos_v = get(0, v_stride);
    let neg_u_neg_v = get(-u_stride, -v_stride);
    let pos_u_neg_v = get(u_stride, -v_stride);
    let pos_u_pos_v = get(u_stride, v_stride);
    let neg_u_pos_v = get(-u_stride, v_stride);

    // AO: does the material darken neighboring vertices?
    let ao = if B::Light::AO_ENABLED {
        let ao_neg_u = shape_ao_opaque(neg_u, ao_face);
        let ao_pos_u = shape_ao_opaque(pos_u, ao_face);
        let ao_neg_v = shape_ao_opaque(neg_v, ao_face);
        let ao_pos_v = shape_ao_opaque(pos_v, ao_face);

        let ao0 = if ao_neg_u && ao_neg_v {
            0
        } else {
            3 - ao_neg_u as u8 - ao_neg_v as u8 - shape_ao_opaque(neg_u_neg_v, ao_face) as u8
        };
        let ao1 = if ao_pos_u && ao_neg_v {
            0
        } else {
            3 - ao_pos_u as u8 - ao_neg_v as u8 - shape_ao_opaque(pos_u_neg_v, ao_face) as u8
        };
        let ao2 = if ao_pos_u && ao_pos_v {
            0
        } else {
            3 - ao_pos_u as u8 - ao_pos_v as u8 - shape_ao_opaque(pos_u_pos_v, ao_face) as u8
        };
        let ao3 = if ao_neg_u && ao_pos_v {
            0
        } else {
            3 - ao_neg_u as u8 - ao_pos_v as u8 - shape_ao_opaque(neg_u_pos_v, ao_face) as u8
        };

        [ao0, ao1, ao2, ao3]
    } else {
        [3; 4]
    };

    // Smooth light: each vertex averages light from the 4 surrounding
    // voxels unconditionally. Unlike AO, we do not exclude opaque
    // neighbors — their stored light values already reflect blockage,
    // and excluding them would cause different blocks sharing a vertex
    // to compute different averages (since one block's "side" neighbor
    // is another's "corner"), producing visible discontinuities.
    let light = if B::Light::LIGHT_ENABLED {
        let cl = center.light();
        [
            B::Light::average(&[cl, neg_u.light(), neg_v.light(), neg_u_neg_v.light()]),
            B::Light::average(&[cl, pos_u.light(), neg_v.light(), pos_u_neg_v.light()]),
            B::Light::average(&[cl, pos_u.light(), pos_v.light(), pos_u_pos_v.light()]),
            B::Light::average(&[cl, neg_u.light(), pos_v.light(), neg_u_pos_v.light()]),
        ]
    } else {
        Default::default()
    };

    (ao, light)
}

/// Moves per-vertex AO and light, computed at a cell face's four corners,
/// onto the corners of a rectangle inside that face.
///
/// Both are samples of a field defined at the corners of cells. Geometry
/// that does not fill its cell — a stair's tread and riser, the L its side
/// shows, the strip on a slab's — has vertices part-way across the face,
/// and those take the field's value where they actually stand, which is
/// the bilinear blend of the four corners.
///
/// Without this a stair took the whole cell's gradient and stretched it
/// over each of its halves, so a neighbor's darkening arrived at the middle
/// of the step rather than at the edge it touches, twice as steep as it
/// should be and showing plainly the moment anything stood near enough to
/// cast AO at all.
///
/// A rectangle that *is* the whole face is returned untouched, which is the
/// common case and keeps whole blocks exactly as they were.
#[inline]
fn resample_to_rect<B: Block>(
    ao: [u8; 4],
    light: [<B::Light as Light>::Average; 4],
    rect: &Rect16,
) -> ([u8; 4], [<B::Light as Light>::Average; 4]) {
    if rect.is_full() {
        return (ao, light);
    }
    let ft = FULL_THICKNESS as f32;
    // Bilinear weights at one corner of the rectangle, in the same
    // mask-local order the four samples are in.
    let weights = |u: u8, v: u8| {
        let (tu, tv) = (u as f32 / ft, v as f32 / ft);
        [
            (1.0 - tu) * (1.0 - tv),
            tu * (1.0 - tv),
            tu * tv,
            (1.0 - tu) * tv,
        ]
    };
    let corners = [
        weights(rect.u0, rect.v0),
        weights(rect.u1, rect.v0),
        weights(rect.u1, rect.v1),
        weights(rect.u0, rect.v1),
    ];
    let out_ao = if B::Light::AO_ENABLED {
        corners.map(|w| {
            let blended = ao[0] as f32 * w[0]
                + ao[1] as f32 * w[1]
                + ao[2] as f32 * w[2]
                + ao[3] as f32 * w[3];
            // AO is four levels and a vertex gets one of them, so the
            // blend has to land back on the ladder it came off.
            blended.round().clamp(0.0, 3.0) as u8
        })
    } else {
        ao
    };
    let out_light = if B::Light::LIGHT_ENABLED {
        corners.map(|w| B::Light::blend(&light, w))
    } else {
        light
    };
    (out_ao, out_light)
}

impl<L: Light> Quads<L> {
    /// Creates an empty `Quads` with no allocations.
    pub fn new() -> Self {
        Quads {
            faces: [vec![], vec![], vec![], vec![], vec![], vec![]],
            diagonals: [vec![], vec![]],
            fluid: [vec![], vec![], vec![], vec![], vec![], vec![]],
        }
    }

    /// Clears all face lists without freeing their backing allocations.
    pub fn reset(&mut self) {
        for face in &mut self.faces {
            face.clear();
        }
        for diag in &mut self.diagonals {
            diag.clear();
        }
        for face in &mut self.fluid {
            face.clear();
        }
    }

    /// Total number of quads across all faces (including diagonals and
    /// fluid overlays).
    pub fn total(&self) -> usize {
        self.faces.iter().map(|v| v.len()).sum::<usize>()
            + self.diagonals.iter().map(|v| v.len()).sum::<usize>()
            + self.fluid.iter().map(|v| v.len()).sum::<usize>()
    }

    /// Returns the quad list for the given [`Face`].
    ///
    /// This allows iterating all faces uniformly via [`Face::ALL`]:
    ///
    /// ```ignore
    /// for qf in Face::ALL {
    ///     for quad in quads.get(qf) {
    ///         let vp = quad.voxel_position(qf);
    ///         // ...
    ///     }
    /// }
    /// ```
    pub fn get(&self, face: Face) -> &[Quad<L>] {
        match face {
            Face::Aligned(f) => &self.faces[f.index()],
            Face::Diagonal(d) => &self.diagonals[d.index()],
        }
    }
}

impl<L: Light> Default for Quads<L> {
    fn default() -> Self {
        Self::new()
    }
}

/// Meshes a padded chunk, returning the generated quads.
///
/// When `greedy` is true, coplanar identical faces are merged into
/// larger quads (fewer draw calls, but hides per-block boundaries).
pub fn mesh_chunk<B: Block, S: ChunkShape>(
    chunk: &PaddedChunk<B, S>,
    greedy: bool,
) -> Quads<B::Light>
where
    [(); S::PADDED_VOLUME]:,
    [(); S::SIZE]:,
{
    let mut quads = Quads::new();
    mesh_chunk_into(chunk, greedy, &mut quads);
    quads
}

/// Returns (n_stride, u_stride, v_stride) as linear index steps into
/// the padded chunk array, matching the tangent convention in
/// [`face_tangents`].
#[inline]
fn face_strides(face: AlignedFace, padded: usize) -> (usize, usize, usize) {
    let p = padded;
    let p2 = padded * padded;
    match face {
        AlignedFace::PosX | AlignedFace::NegX => (1, p2, p), // normal=X, u=Z, v=Y
        AlignedFace::PosY | AlignedFace::NegY => (p, p2, 1), // normal=Y, u=Z, v=X
        AlignedFace::PosZ | AlignedFace::NegZ => (p2, 1, p), // normal=Z, u=X, v=Y
    }
}

/// Converts a [`MaskEntry`] into a [`Quad`].
///
/// - `normal_idx`, `u_idx`, `v_idx`: axis indices from
///   [`face_axis_indices`].
/// - `normal_block`, `u_block`, `v_block`: block-level position along
///   each axis, already including any padding offset.
/// - `width`, `height`: number of blocks merged along u/v (1 when not
///   greedy merging).
/// - `face`: the face being emitted, used for vertex order correction.
#[inline]
fn emit_quad<B: Block>(
    entry: &MaskEntry<B>,
    normal_idx: usize,
    u_idx: usize,
    v_idx: usize,
    normal_block: u32,
    u_block: u32,
    v_block: u32,
    width: u32,
    height: u32,
    face: AlignedFace,
) -> Quad<B::Light> {
    let ft32 = FULL_THICKNESS;
    let mut origin = [0u32; 3];
    origin[normal_idx] = normal_block * ft32 + entry.normal_pos as u32;
    origin[u_idx] = u_block * ft32 + entry.u_intra_offset as u32;
    origin[v_idx] = v_block * ft32 + entry.v_intra_offset as u32;

    // Reorder AO, light and fluid offsets from mask-local order
    // [umin/vmin, umax/vmin, umax/vmax, umin/vmax] to match the vertex
    // order from positions().
    let (corner_offsets, ao, light) = if face.tangent_cross_positive() {
        // positions(): [base, base+du, base+du+dv, base+dv]
        // = [umin/vmin, umax/vmin, umax/vmax, umin/vmax]
        (entry.corner_offsets, entry.ao, entry.light)
    } else {
        // positions(): [base, base+dv, base+dv+du, base+du]
        // = [umin/vmin, umin/vmax, umax/vmax, umax/vmin]
        (
            [
                entry.corner_offsets[0],
                entry.corner_offsets[3],
                entry.corner_offsets[2],
                entry.corner_offsets[1],
            ],
            [entry.ao[0], entry.ao[3], entry.ao[2], entry.ao[1]],
            [
                entry.light[0],
                entry.light[3],
                entry.light[2],
                entry.light[1],
            ],
        )
    };

    Quad {
        origin_padded: UVec3::new(origin[0], origin[1], origin[2]),
        size: UVec2::new(
            width * entry.u_intra_extent as u32,
            height * entry.v_intra_extent as u32,
        ),
        corner_offsets,
        ao,
        light,
    }
}

/// Emits diagonal quads for a cross-shaped block into `quads`.
///
/// `block_pos` is the block-level position (including padding) per axis.
/// `root_face` determines orientation: its axis is the merge axis, and
/// `merge_len` is the number of blocks merged along it.
/// `light_bottom` and `light_top` are the per-vertex light values for the
/// bottom and top vertices of the cross quad.
#[inline]
fn emit_cross_quads<B: Block>(
    quads: &mut Quads<B::Light>,
    block_pos: [u32; 3],
    root_face: AlignedFace,
    merge_len: u32,
    light_bottom: <B::Light as Light>::Average,
    light_top: <B::Light as Light>::Average,
) {
    let ft32 = FULL_THICKNESS;
    let merge_axis = root_face.axis().index();
    let (cross_a, cross_b) = cross_axes(root_face.axis());

    // size.x = one block wide in the crossing plane.
    // size.y = merge_len blocks along the merge axis.
    let mut origin = [0u32; 3];
    origin[cross_a] = block_pos[cross_a] * ft32;
    origin[cross_b] = block_pos[cross_b] * ft32;
    origin[merge_axis] = block_pos[merge_axis] * ft32;

    // Cross quad vertices: v0,v1 at bottom, v2,v3 at top.
    let ao = [3; 4];
    let light = [light_bottom, light_bottom, light_top, light_top];

    for diag in DiagonalFace::ALL {
        quads.diagonals[diag.index()].push(Quad {
            origin_padded: UVec3::new(origin[0], origin[1], origin[2]),
            size: UVec2::new(ft32, merge_len * ft32),
            corner_offsets: [0; 4],
            ao,
            light,
        });
    }
}

/// Emits one layer's mask into `out`, greedily merging runs of identical
/// entries when asked to, and clears it as it goes.
///
/// A fluid needs no special case here, because `corner_offsets` is part
/// of a mask entry's identity. Two cells that merge have equal offsets,
/// and the vertex between them is one vertex, so the offsets they each
/// give it must agree — which forces the run to be flat along the
/// direction it merges in. Standing water merges like stone; a slope
/// cannot merge at all, and never silently loses its crease.
#[allow(clippy::too_many_arguments)]
#[inline]
fn emit_mask<B: Block, S: ChunkShape>(
    mask: &mut [[Option<MaskEntry<B>>; S::SIZE]; S::SIZE],
    greedy: bool,
    out: &mut Vec<Quad<B::Light>>,
    normal_idx: usize,
    u_idx: usize,
    v_idx: usize,
    layer: usize,
    face: AlignedFace,
) where
    [(); S::SIZE]:,
{
    let ft = FULL_THICKNESS as u8;
    for v in 0..S::SIZE {
        let mut u = 0;
        while u < S::SIZE {
            let entry = match mask[v][u] {
                Some(e) => e,
                None => {
                    u += 1;
                    continue;
                }
            };

            let mut width = 1;
            let mut height = 1;

            if greedy {
                // Find widest run of identical entries along u.
                // Sub-block u extents (slabs, stairs) must not merge along u.
                if entry.u_intra_extent == ft {
                    while u + width < S::SIZE && mask[v][u + width] == Some(entry) {
                        width += 1;
                    }
                }

                // Extend the run along v.
                // Sub-block v extents (slabs, stairs) must not merge along v.
                if entry.v_intra_extent == ft {
                    'extend: while v + height < S::SIZE {
                        for du in 0..width {
                            if mask[v + height][u + du] != Some(entry) {
                                break 'extend;
                            }
                        }
                        height += 1;
                    }
                }
            }

            // Clear the merged region.
            for dv in 0..height {
                for du in 0..width {
                    mask[v + dv][u + du] = None;
                }
            }

            out.push(emit_quad(
                &entry,
                normal_idx,
                u_idx,
                v_idx,
                (PADDING + layer) as u32,
                (PADDING + u) as u32,
                (PADDING + v) as u32,
                width as u32,
                height as u32,
                face,
            ));
            u += width;
        }
    }
}

/// Meshes a single block with all faces exposed (no neighbor culling).
///
/// `light` is applied uniformly to all vertices; AO is disabled.
/// Useful for rendering held items or dropped block entities.
pub fn mesh_block<B: Block>(block: &B, light: <B::Light as Light>::Average) -> Quads<B::Light> {
    let mut quads = Quads::new();
    mesh_block_into(block, light, &mut quads);
    quads
}

/// Like [`mesh_block`], but reuses an existing [`Quads`] buffer.
pub fn mesh_block_into<B: Block>(
    block: &B,
    light: <B::Light as Light>::Average,
    quads: &mut Quads<B::Light>,
) {
    quads.reset();

    if !block.cull_mode().is_renderable() {
        return;
    }

    let avg = light;

    if let Shape::Cross(info) = block.shape() {
        let p = PADDING as u32;
        emit_cross_quads::<B>(quads, [p, p, p], info.face, 1, avg, avg);
        return;
    }

    for face in AlignedFace::ALL {
        let (normal_idx, u_idx, v_idx) = face_axis_indices(face);

        // A stair shows two quads on some faces; everything else one.
        let entries = match block.shape() {
            Shape::Stair(info) => stair_entries(block, info, face, u_idx, v_idx)
                .map(|part| part.map(|(entry, _)| entry)),
            _ => [mask_entry_for_shape(block, face, u_idx, v_idx), None],
        };
        for mut entry in entries.into_iter().flatten() {
            entry.light = [avg; 4];
            let quad = emit_quad(
                &entry,
                normal_idx,
                u_idx,
                v_idx,
                PADDING as u32,
                PADDING as u32,
                PADDING as u32,
                1,
                1,
                face,
            );
            quads.faces[face.index()].push(quad);
        }
    }
}

/// Like [`mesh_chunk`], but reuses an existing [`Quads`] buffer.
///
/// The buffer is [`reset`](Quads::reset) before meshing, so previous
/// contents are cleared but backing allocations are preserved.
pub fn mesh_chunk_into<B: Block, S: ChunkShape>(
    chunk: &PaddedChunk<B, S>,
    greedy: bool,
    quads: &mut Quads<B::Light>,
) where
    [(); S::PADDED_VOLUME]:,
    [(); S::SIZE]:,
{
    quads.reset();
    let ft = FULL_THICKNESS as u8;
    let data = &chunk.data;

    // Masks are hoisted outside the layer loop. The build phase overwrites
    // every cell unconditionally so previous values do not matter. Three
    // per layer: the shape's quad, the second quad a stair shows on some
    // faces, and the fluid standing in a solid's cell. The second two are
    // sparse in practice and cost a `None` per cell when unused.
    let mut mask: [[Option<MaskEntry<B>>; S::SIZE]; S::SIZE] = [[None; S::SIZE]; S::SIZE];
    let mut extra: [[Option<MaskEntry<B>>; S::SIZE]; S::SIZE] = [[None; S::SIZE]; S::SIZE];
    let mut overlay: [[Option<MaskEntry<B>>; S::SIZE]; S::SIZE] = [[None; S::SIZE]; S::SIZE];

    for face in AlignedFace::ALL {
        let (normal_idx, u_idx, v_idx) = face_axis_indices(face);
        let (n_stride, u_stride, v_stride) = face_strides(face, S::PADDED);
        let neighbor_stride: isize = if face.is_positive() {
            n_stride as isize
        } else {
            -(n_stride as isize)
        };
        let whole_normal_pos: u8 = if face.is_positive() { ft } else { 0 };

        for layer in 0..S::SIZE {
            let layer_base = (PADDING + layer) * n_stride + PADDING * u_stride + PADDING * v_stride;

            // Build the 2D masks for this layer.
            let mut v_base = layer_base;
            for v in 0..S::SIZE {
                let mut idx = v_base;
                for u in 0..S::SIZE {
                    debug_assert!(idx < data.len());
                    let n_idx = (idx as isize + neighbor_stride) as usize;
                    debug_assert!(n_idx < data.len());

                    // SAFETY: the padding ring guarantees all indices
                    // (including the neighbor one step along the normal)
                    // are within the PADDED_VOLUME array.
                    let (block, neighbor) =
                        unsafe { (data.get_unchecked(idx), data.get_unchecked(n_idx)) };

                    let shape = block.shape();
                    let is_facade = matches!(shape, Shape::Facade(_));

                    let mut second = None;
                    let mut entry = match shape {
                        _ if !block.cull_mode().is_renderable() => None,
                        Shape::WholeBlock => {
                            if is_culled_at_boundary(block, neighbor, face, &Rect16::FULL) {
                                None
                            } else {
                                // WholeBlock fast path: normal_pos is constant
                                // for the entire face, skip shape dispatch.
                                Some(MaskEntry {
                                    block: *block,
                                    normal_pos: whole_normal_pos,
                                    u_intra_offset: 0,
                                    u_intra_extent: ft,
                                    v_intra_offset: 0,
                                    v_intra_extent: ft,
                                    corner_offsets: [0; 4],
                                    ao: [3; 4],
                                    light: Default::default(),
                                })
                            }
                        }
                        // Cross blocks are handled in a separate pass.
                        Shape::Cross(_) => None,
                        // Facade quads are offset inward from their face,
                        // treated as never flush with the block boundary,
                        // so skip neighbor culling.
                        Shape::Facade(_) => mask_entry_for_shape(block, face, u_idx, v_idx),
                        Shape::Inset(_) => {
                            if face.axis() == Axis::Y {
                                // Top/bottom at boundary, normal culling.
                                if is_culled_at_boundary(block, neighbor, face, &Rect16::FULL) {
                                    None
                                } else {
                                    mask_entry_for_shape(block, face, u_idx, v_idx)
                                }
                            } else {
                                // Side faces are inset, no neighbor culling.
                                mask_entry_for_shape(block, face, u_idx, v_idx)
                            }
                        }
                        Shape::Slab(_) => {
                            compute_slab_mask_entry(block, neighbor, face, u_idx, v_idx)
                        }
                        Shape::Stair(info) => {
                            // Each of the two quads is culled on its own
                            // rectangle; the inset ones never are.
                            let keep = |part: Option<(MaskEntry<B>, bool)>| {
                                part.filter(|(e, at_boundary)| {
                                    !at_boundary
                                        || !is_culled_at_boundary(
                                            block,
                                            neighbor,
                                            face,
                                            &entry_rect(e),
                                        )
                                })
                                .map(|(e, _)| e)
                            };
                            let [first, next] = stair_entries(block, info, face, u_idx, v_idx);
                            second = keep(next);
                            keep(first)
                        }
                        Shape::Fluid(info) => {
                            debug_assert!(
                                B::FLUID_ENABLED,
                                "Shape::Fluid requires Block::FLUID_ENABLED"
                            );
                            if B::FLUID_ENABLED {
                                // SAFETY: `idx` is an inner cell, so the
                                // 3x3 of columns around it and one step
                                // along the fluid axis are inside the
                                // padding ring.
                                unsafe {
                                    compute_fluid_mask_entry(
                                        data,
                                        idx,
                                        block,
                                        neighbor,
                                        info,
                                        face,
                                        normal_idx,
                                        u_idx,
                                        v_idx,
                                        S::PADDED,
                                    )
                                }
                            } else {
                                None
                            }
                        }
                    };

                    // The fluid standing in a solid's cell. Not gated on
                    // the solid being renderable: a consumer may draw the
                    // solid some other way and still want its water.
                    let mut fluid_entry = None;
                    if B::FLUID_ENABLED && !matches!(shape, Shape::Fluid(_)) {
                        if let Some(info) = block.fluid() {
                            // SAFETY: as for `Shape::Fluid` above.
                            fluid_entry = unsafe {
                                compute_fluid_mask_entry(
                                    data,
                                    idx,
                                    block,
                                    neighbor,
                                    info,
                                    face,
                                    normal_idx,
                                    u_idx,
                                    v_idx,
                                    S::PADDED,
                                )
                            };
                        }
                    }

                    // Compute AO and smooth light for visible faces.
                    if B::Light::AO_ENABLED || B::Light::LIGHT_ENABLED {
                        let light_entry = |e: &mut MaskEntry<B>, overlay: bool| {
                            // Faces inset into the block sample AO/light
                            // at the block's own plane rather than the
                            // neighbor's, and check the inset direction
                            // for occlusion instead of the opposite,
                            // since neighbors at the same level don't
                            // protrude past the surface. A slab's inner
                            // face, a stair's tread and riser.
                            //
                            // The fluid standing in a solid's cell is none of
                            // that solid's geometry and must not be lit as if
                            // it were. A waterlogged ladder is a `Facade`, so
                            // without the `overlay` test its water was sampled
                            // at the ladder's own cell against the ladder's
                            // occluders, and the surface came out a different
                            // shade from the pond it is part of — a rectangle
                            // of slightly wrong blue, outlined against the
                            // water next door. Left to `inset`, it lights
                            // exactly as a plain fluid cell of the same height
                            // does, which is what it is.
                            let inset = e.normal_pos != whole_normal_pos;
                            let (sample_idx, ao_face) = if is_facade && !overlay {
                                (idx, face.opposite())
                            } else if inset {
                                (idx, face)
                            } else {
                                (n_idx, face.opposite())
                            };
                            let (ao, light) = compute_ao_light(
                                data,
                                sample_idx,
                                u_stride as isize,
                                v_stride as isize,
                                ao_face,
                            );
                            // Sampled at the cell's corners; this quad may
                            // only cover part of the cell.
                            let (ao, light) = resample_to_rect::<B>(ao, light, &entry_rect(e));
                            e.ao = ao;
                            e.light = light;
                        };
                        if let Some(ref mut e) = entry {
                            light_entry(e, false);
                        }
                        if let Some(ref mut e) = second {
                            light_entry(e, false);
                        }
                        if let Some(ref mut e) = fluid_entry {
                            light_entry(e, true);
                        }
                    }

                    mask[v][u] = entry;
                    extra[v][u] = second;
                    overlay[v][u] = fluid_entry;

                    idx += u_stride;
                }
                v_base += v_stride;
            }

            // Emit phase (with optional greedy merging).
            emit_mask::<B, S>(
                &mut mask,
                greedy,
                &mut quads.faces[face.index()],
                normal_idx,
                u_idx,
                v_idx,
                layer,
                face,
            );
            emit_mask::<B, S>(
                &mut extra,
                greedy,
                &mut quads.faces[face.index()],
                normal_idx,
                u_idx,
                v_idx,
                layer,
                face,
            );
            if B::FLUID_ENABLED {
                emit_mask::<B, S>(
                    &mut overlay,
                    greedy,
                    &mut quads.fluid[face.index()],
                    normal_idx,
                    u_idx,
                    v_idx,
                    layer,
                    face,
                );
            }
        }
    }

    // Cross-block pass: for each merge axis, scan columns along that
    // axis and merge consecutive identical cross blocks.
    let axis_strides = [1usize, S::PADDED, S::PADDED * S::PADDED];

    for merge_axis in 0..3usize {
        let (plane_a, plane_b) = match merge_axis {
            0 => (1, 2), // merge along X, iterate YZ
            1 => (0, 2), // merge along Y, iterate XZ
            _ => (0, 1), // merge along Z, iterate XY
        };
        let merge_stride = axis_strides[merge_axis];

        for pb in 0..S::SIZE {
            for pa in 0..S::SIZE {
                let mut pos = [0usize; 3];
                pos[plane_a] = pa + PADDING;
                pos[plane_b] = pb + PADDING;
                pos[merge_axis] = PADDING;
                let col_base = pos[0] + pos[1] * S::PADDED + pos[2] * S::PADDED * S::PADDED;

                let mut m = 0;
                while m < S::SIZE {
                    let idx = col_base + m * merge_stride;
                    let block = unsafe { data.get_unchecked(idx) };

                    let info = match block.shape() {
                        Shape::Cross(info)
                            if block.cull_mode().is_renderable()
                                && info.face.axis().index() == merge_axis =>
                        {
                            info
                        }
                        _ => {
                            m += 1;
                            continue;
                        }
                    };

                    let mut merge_len = 1u32;
                    if greedy {
                        while m + merge_len as usize <= S::SIZE - 1 {
                            let next_idx = col_base + (m + merge_len as usize) * merge_stride;
                            let next = unsafe { data.get_unchecked(next_idx) };
                            if next != block {
                                break;
                            }
                            merge_len += 1;
                        }
                    }

                    let mut block_pos = [0u32; 3];
                    block_pos[plane_a] = (PADDING + pa) as u32;
                    block_pos[plane_b] = (PADDING + pb) as u32;
                    block_pos[merge_axis] = (PADDING + m) as u32;

                    // Compute interpolated light for cross block endpoints.
                    let (light_bottom, light_top) = if B::Light::LIGHT_ENABLED {
                        let first_idx = idx;
                        let last_idx = col_base + (m + merge_len as usize - 1) * merge_stride;
                        // Bottom: average of first block and the block below it.
                        let below_idx = (first_idx as isize - merge_stride as isize) as usize;
                        let below = unsafe { data.get_unchecked(below_idx) };
                        let first = unsafe { data.get_unchecked(first_idx) };
                        let light_bottom = B::Light::average(&[first.light(), below.light()]);
                        // Top: average of last block and the block above it.
                        let above_idx = last_idx + merge_stride;
                        let above = unsafe { data.get_unchecked(above_idx) };
                        let last = unsafe { data.get_unchecked(last_idx) };
                        let light_top = B::Light::average(&[last.light(), above.light()]);
                        (light_bottom, light_top)
                    } else {
                        Default::default()
                    };

                    emit_cross_quads::<B>(
                        quads,
                        block_pos,
                        info.face,
                        merge_len,
                        light_bottom,
                        light_top,
                    );

                    m += merge_len as usize;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{CullMode, Shape};
    use crate::chunk::{ChunkShape, ChunkShape16, PaddedChunk16};
    use crate::face::AlignedFace;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum TestBlock {
        Air,
        Stone,
    }

    impl Block for TestBlock {
        type TransparentGroup = ();

        fn shape(&self) -> Shape {
            Shape::WholeBlock
        }
        fn cull_mode(&self) -> CullMode {
            match self {
                TestBlock::Air => CullMode::Empty,
                TestBlock::Stone => CullMode::Opaque,
            }
        }
    }

    #[test]
    fn block_faces_matches_greedy_mesh_for_single_block() {
        let mut chunk = PaddedChunk16::new_filled(TestBlock::Air);
        chunk.set(UVec3::ZERO, TestBlock::Stone);
        let from_chunk = mesh_chunk(&chunk, true);
        let from_block = mesh_block(&TestBlock::Stone, ());
        assert_eq!(from_chunk.total(), from_block.total());
        for face in AlignedFace::ALL {
            assert_eq!(
                from_chunk.faces[face.index()],
                from_block.faces[face.index()],
                "face {:?}",
                face
            );
        }
    }

    #[test]
    fn block_faces_air_produces_no_quads() {
        let q = mesh_block(&TestBlock::Air, ());
        assert_eq!(q.total(), 0);
    }

    #[test]
    fn single_block_quad_size_is_one_block() {
        let mut chunk = PaddedChunk16::new_filled(TestBlock::Air);
        chunk.set(UVec3::ZERO, TestBlock::Stone);
        let q = mesh_chunk(&chunk, true);
        for face in AlignedFace::ALL {
            let quad = &q.faces[face.index()][0];
            assert_eq!(quad.size, UVec2::new(16, 16), "face {:?}", face);
        }
    }

    #[test]
    fn full_chunk_quad_size_is_sixteen_blocks() {
        let mut chunk = PaddedChunk16::new_filled(TestBlock::Air);
        for x in 0..ChunkShape16::SIZE as u32 {
            for y in 0..ChunkShape16::SIZE as u32 {
                for z in 0..ChunkShape16::SIZE as u32 {
                    chunk.set(UVec3::new(x, y, z), TestBlock::Stone);
                }
            }
        }
        let q = mesh_chunk(&chunk, true);
        for face in AlignedFace::ALL {
            let quad = &q.faces[face.index()][0];
            assert_eq!(quad.size, UVec2::new(16 * 16, 16 * 16), "face {:?}", face);
        }
    }
}

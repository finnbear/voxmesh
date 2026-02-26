use glam::{UVec2, UVec3, Vec2, Vec3};

use crate::block::{Block, CrossInfo, CullMode, Shape, FULL_THICKNESS};
use crate::chunk::{PaddedChunk, CHUNK_SIZE, PADDED, PADDING};
use crate::face::{Axis, DiagonalFace, Face, QuadFace};
use crate::light::Light;

#[derive(Debug, Clone, PartialEq)]
pub struct Quad<L: Light = ()> {
    /// Position of the lowest-coordinate corner in 1/16ths of a block,
    /// in the padded 3D space of the chunk.
    origin_padded: UVec3,
    /// Size of the quad in 1/16ths of a block.
    size: UVec2,
    /// Per-vertex ambient occlusion (0=fully occluded, 3=fully lit).
    /// Vertex order matches [`positions`](Self::positions).
    pub ao: [u8; 4],
    /// Per-vertex averaged light values.
    /// Vertex order matches [`positions`](Self::positions).
    pub light: [L::Average; 4],
}

/// Returns the (u, v) tangent vectors for a face.
/// u corresponds to `size.x`, v corresponds to `size.y`.
/// Matches the block-mesh axis permutation convention (Xzy, Yzx, Zxy).
fn face_tangents(face: Face) -> (Vec3, Vec3) {
    match face {
        Face::PosX | Face::NegX => (Vec3::Z, Vec3::Y),
        Face::PosY | Face::NegY => (Vec3::Z, Vec3::X),
        Face::PosZ | Face::NegZ => (Vec3::X, Vec3::Y),
    }
}

impl<L: Light> Quad<L> {
    /// Returns the minimum voxel coordinate (excluding padding) of the
    /// block that produced this quad. Use this to look up the block type
    /// in a chunk or flat voxel array.
    ///
    /// `face` must match the face under which this quad was generated: an
    /// axis-aligned [`Face`] for quads from [`Quads::faces`], or a
    /// [`DiagonalFace`] for quads from [`Quads::diagonals`].
    pub fn voxel_position(&self, face: impl Into<QuadFace>) -> UVec3 {
        let ft = FULL_THICKNESS;
        let pad = PADDING as u32;
        match face.into() {
            QuadFace::Aligned(f) => {
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
            QuadFace::Diagonal(_) => UVec3::new(
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
    pub fn positions(&self, face: impl Into<QuadFace>, shape: Shape) -> [Vec3; 4] {
        match face.into() {
            QuadFace::Aligned(face) => {
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
                if face.tangent_cross_positive() {
                    [base, base + du, base + du + dv, base + dv]
                } else {
                    [base, base + dv, base + dv + du, base + du]
                }
            }
            QuadFace::Diagonal(diag) => {
                let info = match shape {
                    Shape::Cross(info) => info,
                    _ => CrossInfo {
                        face: Face::NegY,
                        stretch: 0,
                    },
                };
                let scale = 1.0 / FULL_THICKNESS as f32;
                let pad = PADDING as f32;

                // The root face axis is the merge/height axis.
                // The two perpendicular axes form the crossing plane.
                let merge_axis = info.face.axis().index();
                let (cross_a, cross_b) = cross_axes(info.face.axis());

                let origin = Vec3::new(
                    self.origin_padded.x as f32 * scale - pad,
                    self.origin_padded.y as f32 * scale - pad,
                    self.origin_padded.z as f32 * scale - pad,
                );

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
    pub fn texture_coordinates(
        &self,
        face: impl Into<QuadFace>,
        u_flip_face: Axis,
        flip_v: bool,
    ) -> [Vec2; 4] {
        match face.into() {
            QuadFace::Aligned(face) => {
                let scale = 1.0 / FULL_THICKNESS as f32;
                let u_size = self.size.x as f32 * scale;
                let v_size = self.size.y as f32 * scale;

                let flip_u = if face.is_positive() {
                    face.axis() == u_flip_face
                } else {
                    face.axis() != u_flip_face
                };

                let raw = if face.tangent_cross_positive() {
                    [
                        Vec2::new(0.0, 0.0),
                        Vec2::new(u_size, 0.0),
                        Vec2::new(u_size, v_size),
                        Vec2::new(0.0, v_size),
                    ]
                } else {
                    [
                        Vec2::new(0.0, 0.0),
                        Vec2::new(0.0, v_size),
                        Vec2::new(u_size, v_size),
                        Vec2::new(u_size, 0.0),
                    ]
                };

                raw.map(|uv| {
                    Vec2::new(
                        if flip_u { u_size - uv.x } else { uv.x },
                        if flip_v { v_size - uv.y } else { uv.y },
                    )
                })
            }
            QuadFace::Diagonal(_) => {
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

    /// Returns the 6 vertex indices with the anisotropy fix for AO.
    ///
    /// When the AO values form a saddle pattern (opposite corners have
    /// different sums), the triangle diagonal is flipped to avoid visual
    /// artifacts in AO shading.
    #[inline]
    pub fn indices_ao(&self, start: u32) -> [u32; 6] {
        let ao = &self.ao;
        if ao[0] as u16 + ao[2] as u16 >= ao[1] as u16 + ao[3] as u16 {
            [start, start + 1, start + 2, start, start + 2, start + 3]
        } else {
            [start, start + 1, start + 3, start + 1, start + 2, start + 3]
        }
    }
}

pub struct Quads<L: Light = ()> {
    pub faces: [Vec<Quad<L>>; 6],
    /// Diagonal quads for X-shaped billboard blocks, indexed by [`DiagonalFace`].
    pub diagonals: [Vec<Quad<L>>; 2],
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
    /// Per-vertex AO in mask-local order: [umin/vmin, umax/vmin, umax/vmax, umin/vmax].
    ao: [u8; 4],
    /// Per-vertex light in mask-local order.
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
fn face_axis_indices(face: Face) -> (usize, usize, usize) {
    match face {
        Face::PosX | Face::NegX => (0, 2, 1), // normal=X, u=Z, v=Y
        Face::PosY | Face::NegY => (1, 2, 0), // normal=Y, u=Z, v=X
        Face::PosZ | Face::NegZ => (2, 0, 1), // normal=Z, u=X, v=Y
    }
}

/// Whether the neighbor fully covers the block's face region on the
/// shared boundary. For whole-blocks this is trivial. For slabs, checks
/// whether the neighbor occupies at least the same sub-region along the
/// slab axis.
fn neighbor_covers_face_region<B: Block>(block: &B, neighbor: &B, face: Face) -> bool {
    match neighbor.shape() {
        Shape::WholeBlock => true,
        // Cross and facade blocks never cover any face region.
        Shape::Cross(_) | Shape::Facade(_) => false,
        // Inset blocks cover top/bottom (flush) but not sides (inset).
        Shape::Inset(_) => face.axis() == Axis::Y,
        Shape::Slab(n_info) => {
            // The neighbor slab is flush against our face only if its
            // slab face equals our face's opposite.
            if n_info.face == face.opposite() {
                return true;
            }
            // For side faces: if the block is also a slab on the same axis
            // and the neighbor covers at least the block's extent, it culls.
            if let Shape::Slab(b_info) = block.shape() {
                if b_info.face.axis() == n_info.face.axis() && face.axis() != b_info.face.axis() {
                    // Both slabs share the same axis. The neighbor covers
                    // our region if its thickness >= ours on the same side.
                    return b_info.face == n_info.face && n_info.thickness >= b_info.thickness;
                }
            }
            false
        }
    }
}

/// Whether the current block's face is culled by the given neighbor.
/// Only valid for faces at the block boundary (flush or side).
fn is_culled_at_boundary<B: Block>(block: &B, neighbor: &B, face: Face) -> bool {
    if !neighbor_covers_face_region(block, neighbor, face) {
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

/// Compute the mask entry for a block/face based purely on shape,
/// ignoring neighbor culling. Returns `None` for faces that never
/// emit geometry (cross blocks, non-matching facades, etc.).
#[inline]
fn mask_entry_for_shape<B: Block>(
    block: &B,
    face: Face,
    u_idx: usize,
    v_idx: usize,
) -> Option<MaskEntry<B>> {
    let ft = FULL_THICKNESS as u8;
    match block.shape() {
        // Cross blocks have no axis-aligned faces.
        Shape::Cross(_) => return None,
        // Facade emits one quad on its own face, offset 1/16 inward.
        Shape::Facade(facade_face) => {
            if face != facade_face {
                return None;
            }
            let normal_pos = if face.is_positive() { ft - 1 } else { 1 };
            return Some(MaskEntry {
                block: *block,
                normal_pos,
                u_intra_offset: 0,
                u_intra_extent: ft,
                v_intra_offset: 0,
                v_intra_extent: ft,
                ao: [3; 4],
                light: Default::default(),
            });
        }
        Shape::WholeBlock | Shape::Inset(_) => {
            let normal_pos = if let Shape::Inset(n) = block.shape() {
                // Side faces are inset, top/bottom are flush.
                if face.axis() == Axis::Y {
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
                }
            } else {
                if face.is_positive() {
                    ft
                } else {
                    0
                }
            };
            Some(MaskEntry {
                block: *block,
                normal_pos,
                u_intra_offset: 0,
                u_intra_extent: ft,
                v_intra_offset: 0,
                v_intra_extent: ft,
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
    face: Face,
    u_idx: usize,
    v_idx: usize,
) -> Option<MaskEntry<B>> {
    let info = match block.shape() {
        Shape::Slab(info) => info,
        Shape::WholeBlock | Shape::Cross(_) | Shape::Facade(_) | Shape::Inset(_) => unreachable!(),
    };

    // Flush face along the slab's own axis: check neighbor culling.
    if face.axis() == info.face.axis() && face == info.face {
        if is_culled_at_boundary(block, neighbor, face) {
            return None;
        }
    } else if face.axis() != info.face.axis() {
        // Side faces: normal neighbor culling.
        if is_culled_at_boundary(block, neighbor, face) {
            return None;
        }
    }

    mask_entry_for_shape(block, face, u_idx, v_idx)
}

/// Computes per-vertex AO and smooth light for a face cell.
///
/// `data` is the padded chunk array. `n_idx` is the linear index of the
/// voxel one step along the face normal from the current block.
/// `u_stride` and `v_stride` are the linear index steps along the face's
/// tangent axes.
///
/// Returns `(ao, light)` arrays in mask-local vertex order:
/// `[umin/vmin, umax/vmin, umax/vmax, umin/vmax]`.
#[inline]
fn compute_ao_light<B: Block>(
    data: &[B],
    n_idx: usize,
    u_stride: isize,
    v_stride: isize,
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

    let s_neg_u = neg_u.ao_opaque();
    let s_pos_u = pos_u.ao_opaque();
    let s_neg_v = neg_v.ao_opaque();
    let s_pos_v = pos_v.ao_opaque();

    // Vertex 0: (u_min, v_min) — neighbors: neg_u, neg_v, neg_u_neg_v
    let ao0 = if s_neg_u && s_neg_v {
        0
    } else {
        3 - s_neg_u as u8 - s_neg_v as u8 - neg_u_neg_v.ao_opaque() as u8
    };

    // Vertex 1: (u_max, v_min) — neighbors: pos_u, neg_v, pos_u_neg_v
    let ao1 = if s_pos_u && s_neg_v {
        0
    } else {
        3 - s_pos_u as u8 - s_neg_v as u8 - pos_u_neg_v.ao_opaque() as u8
    };

    // Vertex 2: (u_max, v_max) — neighbors: pos_u, pos_v, pos_u_pos_v
    let ao2 = if s_pos_u && s_pos_v {
        0
    } else {
        3 - s_pos_u as u8 - s_pos_v as u8 - pos_u_pos_v.ao_opaque() as u8
    };

    // Vertex 3: (u_min, v_max) — neighbors: neg_u, pos_v, neg_u_pos_v
    let ao3 = if s_neg_u && s_pos_v {
        0
    } else {
        3 - s_neg_u as u8 - s_pos_v as u8 - neg_u_pos_v.ao_opaque() as u8
    };

    let ao = [ao0, ao1, ao2, ao3];

    // Smooth light: each vertex averages light from up to 4 voxels.
    let vertex_light = |s1_opaque: bool,
                        s2_opaque: bool,
                        center_l: B::Light,
                        s1_l: B::Light,
                        s2_l: B::Light,
                        corner_l: B::Light|
     -> <B::Light as Light>::Average {
        if s1_opaque && s2_opaque {
            B::Light::average(&[center_l])
        } else if s1_opaque {
            B::Light::average(&[center_l, s2_l])
        } else if s2_opaque {
            B::Light::average(&[center_l, s1_l])
        } else {
            B::Light::average(&[center_l, s1_l, s2_l, corner_l])
        }
    };

    let cl = center.light();
    let light = [
        vertex_light(
            s_neg_u,
            s_neg_v,
            cl,
            neg_u.light(),
            neg_v.light(),
            neg_u_neg_v.light(),
        ),
        vertex_light(
            s_pos_u,
            s_neg_v,
            cl,
            pos_u.light(),
            neg_v.light(),
            pos_u_neg_v.light(),
        ),
        vertex_light(
            s_pos_u,
            s_pos_v,
            cl,
            pos_u.light(),
            pos_v.light(),
            pos_u_pos_v.light(),
        ),
        vertex_light(
            s_neg_u,
            s_pos_v,
            cl,
            neg_u.light(),
            pos_v.light(),
            neg_u_pos_v.light(),
        ),
    ];

    (ao, light)
}

impl<L: Light> Quads<L> {
    /// Creates an empty `Quads` with no allocations.
    pub fn new() -> Self {
        Quads {
            faces: [vec![], vec![], vec![], vec![], vec![], vec![]],
            diagonals: [vec![], vec![]],
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
    }

    /// Total number of quads across all faces (including diagonals).
    pub fn total(&self) -> usize {
        self.faces.iter().map(|v| v.len()).sum::<usize>()
            + self.diagonals.iter().map(|v| v.len()).sum::<usize>()
    }

    /// Returns the quad list for the given [`QuadFace`].
    ///
    /// This allows iterating all faces uniformly via [`QuadFace::ALL`]:
    ///
    /// ```ignore
    /// for qf in QuadFace::ALL {
    ///     for quad in quads.get(qf) {
    ///         let vp = quad.voxel_position(qf);
    ///         // ...
    ///     }
    /// }
    /// ```
    pub fn get(&self, face: QuadFace) -> &[Quad<L>] {
        match face {
            QuadFace::Aligned(f) => &self.faces[f.index()],
            QuadFace::Diagonal(d) => &self.diagonals[d.index()],
        }
    }
}

impl<L: Light> Default for Quads<L> {
    fn default() -> Self {
        Self::new()
    }
}

pub fn greedy_mesh<B: Block>(chunk: &PaddedChunk<B>) -> Quads<B::Light> {
    let mut quads = Quads::new();
    greedy_mesh_into(chunk, &mut quads);
    quads
}

/// Returns (n_stride, u_stride, v_stride) as linear index steps into
/// the padded chunk array, matching the tangent convention in
/// [`face_tangents`].
#[inline]
fn face_strides(face: Face) -> (usize, usize, usize) {
    const P: usize = PADDED;
    const P2: usize = PADDED * PADDED;
    match face {
        Face::PosX | Face::NegX => (1, P2, P), // normal=X, u=Z, v=Y
        Face::PosY | Face::NegY => (P, P2, 1), // normal=Y, u=Z, v=X
        Face::PosZ | Face::NegZ => (P2, 1, P), // normal=Z, u=X, v=Y
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
    face: Face,
) -> Quad<B::Light> {
    let ft32 = FULL_THICKNESS;
    let mut origin = [0u32; 3];
    origin[normal_idx] = normal_block * ft32 + entry.normal_pos as u32;
    origin[u_idx] = u_block * ft32 + entry.u_intra_offset as u32;
    origin[v_idx] = v_block * ft32 + entry.v_intra_offset as u32;

    // Reorder AO and light from mask-local order [umin/vmin, umax/vmin,
    // umax/vmax, umin/vmax] to match the vertex order from positions().
    let (ao, light) = if face.tangent_cross_positive() {
        // positions(): [base, base+du, base+du+dv, base+dv]
        // = [umin/vmin, umax/vmin, umax/vmax, umin/vmax]
        (entry.ao, entry.light)
    } else {
        // positions(): [base, base+dv, base+dv+du, base+du]
        // = [umin/vmin, umin/vmax, umax/vmax, umax/vmin]
        (
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
    root_face: Face,
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
            ao,
            light,
        });
    }
}

/// Produces quads for a single block at the origin with all faces
/// exposed (no neighbor culling). Useful for rendering held items or
/// dropped block entities.
///
/// The resulting [`Quad`] positions span [0, 1] (or the appropriate
/// sub-range for slabs), the same coordinate space as a block at
/// (0,0,0) in a chunk.
pub fn block_faces<B: Block>(block: &B, light: <B::Light as Light>::Average) -> Quads<B::Light> {
    let mut quads = Quads::new();
    block_faces_into(block, light, &mut quads);
    quads
}

/// Like [`block_faces`], but reuses an existing [`Quads`] buffer.
///
/// `light` is applied uniformly to all vertices. AO is disabled.
pub fn block_faces_into<B: Block>(
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

    for face in Face::ALL {
        let (normal_idx, u_idx, v_idx) = face_axis_indices(face);

        if let Some(mut entry) = mask_entry_for_shape(block, face, u_idx, v_idx) {
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

/// Like [`greedy_mesh`] but reuses an existing [`Quads`] buffer.
///
/// The buffer is [`reset`](Quads::reset) before meshing, so previous
/// contents are cleared but backing allocations are preserved.
pub fn greedy_mesh_into<B: Block>(chunk: &PaddedChunk<B>, quads: &mut Quads<B::Light>) {
    quads.reset();
    let ft = FULL_THICKNESS as u8;
    let data = &chunk.data;

    // Mask is hoisted outside the layer loop. The build phase overwrites
    // every cell unconditionally so previous values do not matter.
    let mut mask: [[Option<MaskEntry<B>>; CHUNK_SIZE]; CHUNK_SIZE] =
        [[None; CHUNK_SIZE]; CHUNK_SIZE];

    for face in Face::ALL {
        let (normal_idx, u_idx, v_idx) = face_axis_indices(face);
        let (n_stride, u_stride, v_stride) = face_strides(face);
        let neighbor_stride: isize = if face.is_positive() {
            n_stride as isize
        } else {
            -(n_stride as isize)
        };
        let whole_normal_pos: u8 = if face.is_positive() { ft } else { 0 };

        for layer in 0..CHUNK_SIZE {
            let layer_base = (PADDING + layer) * n_stride + PADDING * u_stride + PADDING * v_stride;

            // Build the 2D mask for this layer.
            let mut v_base = layer_base;
            for v in 0..CHUNK_SIZE {
                let mut idx = v_base;
                for u in 0..CHUNK_SIZE {
                    debug_assert!(idx < data.len());
                    let n_idx = (idx as isize + neighbor_stride) as usize;
                    debug_assert!(n_idx < data.len());

                    // SAFETY: the padding ring guarantees all indices
                    // (including the neighbor one step along the normal)
                    // are within the PADDED_VOLUME array.
                    let (block, neighbor) =
                        unsafe { (data.get_unchecked(idx), data.get_unchecked(n_idx)) };

                    let mut entry = if !block.cull_mode().is_renderable() {
                        None
                    } else if matches!(block.shape(), Shape::WholeBlock) {
                        if is_culled_at_boundary(block, neighbor, face) {
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
                                ao: [3; 4],
                                light: Default::default(),
                            })
                        }
                    } else if matches!(block.shape(), Shape::Cross(_)) {
                        // Cross blocks are handled in a separate pass.
                        None
                    } else if matches!(block.shape(), Shape::Facade(_)) {
                        // Facade quads are offset 1/16 inward, never at
                        // the block boundary, so skip neighbor culling.
                        mask_entry_for_shape(block, face, u_idx, v_idx)
                    } else if matches!(block.shape(), Shape::Inset(_)) {
                        if face.axis() == Axis::Y {
                            // Top/bottom at boundary, normal culling.
                            if is_culled_at_boundary(block, neighbor, face) {
                                None
                            } else {
                                mask_entry_for_shape(block, face, u_idx, v_idx)
                            }
                        } else {
                            // Side faces are inset, no neighbor culling.
                            mask_entry_for_shape(block, face, u_idx, v_idx)
                        }
                    } else {
                        compute_slab_mask_entry(block, neighbor, face, u_idx, v_idx)
                    };

                    // Compute AO and smooth light for visible faces.
                    if B::Light::ENABLED {
                        if let Some(ref mut e) = entry {
                            let (ao, light) =
                                compute_ao_light(data, n_idx, u_stride as isize, v_stride as isize);
                            e.ao = ao;
                            e.light = light;
                        }
                    }

                    mask[v][u] = entry;

                    idx += u_stride;
                }
                v_base += v_stride;
            }

            // Greedy merge phase.
            for v in 0..CHUNK_SIZE {
                let mut u = 0;
                while u < CHUNK_SIZE {
                    let entry = match mask[v][u] {
                        Some(e) => e,
                        None => {
                            u += 1;
                            continue;
                        }
                    };

                    // Find widest run of identical entries along u.
                    // Sub-block u extents (slabs) must not merge along u.
                    let mut width = 1;
                    if entry.u_intra_extent == ft {
                        while u + width < CHUNK_SIZE && mask[v][u + width] == Some(entry) {
                            width += 1;
                        }
                    }

                    // Extend the run along v.
                    // Sub-block v extents (slabs) must not merge along v.
                    let mut height = 1;
                    if entry.v_intra_extent == ft {
                        'extend: while v + height < CHUNK_SIZE {
                            for du in 0..width {
                                if mask[v + height][u + du] != Some(entry) {
                                    break 'extend;
                                }
                            }
                            height += 1;
                        }
                    }

                    // Clear the merged region.
                    for dv in 0..height {
                        for du in 0..width {
                            mask[v + dv][u + du] = None;
                        }
                    }

                    // Emit the quad.
                    let quad = emit_quad(
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
                    );

                    quads.faces[face.index()].push(quad);
                    u += width;
                }
            }
        }
    }

    // Cross-block pass: for each merge axis, scan columns along that
    // axis and merge consecutive identical cross blocks.
    let axis_strides = [1usize, PADDED, PADDED * PADDED];

    for merge_axis in 0..3usize {
        let (plane_a, plane_b) = match merge_axis {
            0 => (1, 2), // merge along X, iterate YZ
            1 => (0, 2), // merge along Y, iterate XZ
            _ => (0, 1), // merge along Z, iterate XY
        };
        let merge_stride = axis_strides[merge_axis];

        for pb in 0..CHUNK_SIZE {
            for pa in 0..CHUNK_SIZE {
                let mut pos = [0usize; 3];
                pos[plane_a] = pa + PADDING;
                pos[plane_b] = pb + PADDING;
                pos[merge_axis] = PADDING;
                let col_base = pos[0] + pos[1] * PADDED + pos[2] * PADDED * PADDED;

                let mut m = 0;
                while m < CHUNK_SIZE {
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

                    // Merge along the merge axis while the block is identical.
                    let mut merge_len = 1u32;
                    while m + merge_len as usize <= CHUNK_SIZE - 1 {
                        let next_idx = col_base + (m + merge_len as usize) * merge_stride;
                        let next = unsafe { data.get_unchecked(next_idx) };
                        if next != block {
                            break;
                        }
                        merge_len += 1;
                    }

                    let mut block_pos = [0u32; 3];
                    block_pos[plane_a] = (PADDING + pa) as u32;
                    block_pos[plane_b] = (PADDING + pb) as u32;
                    block_pos[merge_axis] = (PADDING + m) as u32;

                    // Compute interpolated light for cross block endpoints.
                    let (light_bottom, light_top) = if B::Light::ENABLED {
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
    use crate::face::Face;

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
        let mut chunk = PaddedChunk::new_filled(TestBlock::Air);
        chunk.set(UVec3::ZERO, TestBlock::Stone);
        let from_chunk = greedy_mesh(&chunk);
        let from_block = block_faces(&TestBlock::Stone, ());
        assert_eq!(from_chunk.total(), from_block.total());
        for face in Face::ALL {
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
        let q = block_faces(&TestBlock::Air, ());
        assert_eq!(q.total(), 0);
    }

    #[test]
    fn single_block_quad_size_is_one_block() {
        let mut chunk = PaddedChunk::new_filled(TestBlock::Air);
        chunk.set(UVec3::ZERO, TestBlock::Stone);
        let q = greedy_mesh(&chunk);
        for face in Face::ALL {
            let quad = &q.faces[face.index()][0];
            assert_eq!(quad.size, UVec2::new(16, 16), "face {:?}", face);
        }
    }

    #[test]
    fn full_chunk_quad_size_is_sixteen_blocks() {
        let mut chunk = PaddedChunk::new_filled(TestBlock::Air);
        for x in 0..CHUNK_SIZE as u32 {
            for y in 0..CHUNK_SIZE as u32 {
                for z in 0..CHUNK_SIZE as u32 {
                    chunk.set(UVec3::new(x, y, z), TestBlock::Stone);
                }
            }
        }
        let q = greedy_mesh(&chunk);
        for face in Face::ALL {
            let quad = &q.faces[face.index()][0];
            assert_eq!(quad.size, UVec2::new(16 * 16, 16 * 16), "face {:?}", face);
        }
    }
}

use glam::{UVec2, UVec3, Vec2, Vec3};

use crate::block::{Block, CrossStretch, CullMode, Shape, FULL_THICKNESS};
use crate::chunk::{PaddedChunk, CHUNK_SIZE, PADDED, PADDING};
use crate::face::{Axis, DiagonalFace, Face, QuadFace};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Quad {
    /// Position of the lowest-coordinate corner, in 1/16's of a block, in the padded 3D space of the chunk.
    origin_padded: UVec3,
    /// Size of the quad in 1/16's of a block.
    size: UVec2,
}

/// Returns the (u_direction, v_direction) tangent vectors for a face.
/// u corresponds to size.x, v corresponds to size.y.
/// Matches the block-mesh axis permutation convention (Xzy, Yzx, Zxy).
fn face_tangents(face: Face) -> (Vec3, Vec3) {
    match face {
        Face::PosX | Face::NegX => (Vec3::Z, Vec3::Y),
        Face::PosY | Face::NegY => (Vec3::Z, Vec3::X),
        Face::PosZ | Face::NegZ => (Vec3::X, Vec3::Y),
    }
}

impl Quad {
    /// Returns the minimum voxel coordinate (not including padding) of the
    /// block that produced this quad. Use this to look up the block type in a
    /// chunk or flat voxel array.
    ///
    /// `face` must match the face under which this quad was generated (an
    /// axis-aligned [`Face`] for quads from [`Quads::faces`], or a
    /// [`DiagonalFace`] for quads from [`Quads::diagonals`]).
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
                        // Positive faces sit at the far edge; step back.
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
    /// (viewed from outside).
    ///
    /// `stretch` is the horizontal stretch in 1/16ths for diagonal
    /// ([`Shape::Cross`]) faces — 0 for sugar cane (corners on the block
    /// diagonal), positive to push edges toward the block corners (cobwebs).
    /// It is ignored for axis-aligned faces.
    pub fn positions(&self, face: impl Into<QuadFace>, stretch: CrossStretch) -> [Vec3; 4] {
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
                // [base, base+du, base+du+dv, base+dv] is CCW when u×v aligns with
                // the outward normal. When u×v opposes it, swap du/dv offsets.
                let cross_sign = u_dir.cross(v_dir).dot(face.normal().as_vec3());
                if cross_sign > 0.0 {
                    [base, base + du, base + du + dv, base + dv]
                } else {
                    [base, base + dv, base + dv + du, base + du]
                }
            }
            QuadFace::Diagonal(diag) => {
                let scale = 1.0 / FULL_THICKNESS as f32;
                let pad = PADDING as f32;

                let base_y = self.origin_padded.y as f32 * scale - pad;
                let height = self.size.y as f32 * scale;

                let bx = self.origin_padded.x as f32 * scale - pad;
                let bz = self.origin_padded.z as f32 * scale - pad;
                let cx = bx + 0.5;
                let cz = bz + 0.5;

                let half_diag = 0.5 + stretch as f32 * scale;

                let dir = diag.direction();
                let dx = dir.x * half_diag;
                let dz = dir.z * half_diag;

                let p0 = Vec3::new(cx - dx, base_y, cz - dz);
                let p1 = Vec3::new(cx + dx, base_y, cz + dz);
                let p2 = Vec3::new(cx + dx, base_y + height, cz + dz);
                let p3 = Vec3::new(cx - dx, base_y + height, cz - dz);

                [p0, p1, p2, p3]
            }
        }
    }

    /// Returns the 4 texture coordinates for this quad.
    ///
    /// `u_flip_face` and `flip_v` control UV mirroring for axis-aligned faces
    /// and are ignored for diagonal faces.
    pub fn texture_coordinates(
        &self,
        face: impl Into<QuadFace>,
        u_flip_face: Axis,
        flip_v: bool,
    ) -> [Vec2; 4] {
        match face.into() {
            QuadFace::Aligned(face) => {
                let u_size = self.size.x as f32 / FULL_THICKNESS as f32;
                let v_size = self.size.y as f32 / FULL_THICKNESS as f32;

                let flip_u = if face.is_positive() {
                    face.axis() == u_flip_face
                } else {
                    face.axis() != u_flip_face
                };

                let (u_dir, v_dir) = face_tangents(face);
                let cross_sign = u_dir.cross(v_dir).dot(face.normal().as_vec3());

                let raw = if cross_sign > 0.0 {
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
                let v_size = self.size.y as f32 / FULL_THICKNESS as f32;
                [
                    Vec2::new(0.0, 0.0),
                    Vec2::new(1.0, 0.0),
                    Vec2::new(1.0, v_size),
                    Vec2::new(0.0, v_size),
                ]
            }
        }
    }

    /// Returns the 6 vertex indices for this quad (two triangles), suitable for
    /// indexed drawing.
    ///
    /// `start` is the index of the first vertex of this quad in the vertex
    /// buffer. The returned indices reference vertices in the order produced by
    /// [`positions`](Self::positions) or
    /// [`diagonal_positions`](Self::diagonal_positions), which is always
    /// counter-clockwise when viewed from outside.
    ///
    /// The winding is compatible with block-mesh-rs's `quad_mesh_indices`.
    #[inline]
    pub fn indices(start: u32) -> [u32; 6] {
        [start, start + 1, start + 2, start, start + 2, start + 3]
    }
}

pub struct Quads {
    pub faces: [Vec<Quad>; 6],
    /// Diagonal quads for X-shaped billboard blocks, indexed by [`DiagonalFace`].
    pub diagonals: [Vec<Quad>; 2],
}

// ---------------------------------------------------------------------------
// Greedy meshing internals
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
struct MaskEntry<B: Block> {
    block: B,
    /// Face surface position along the normal axis, in 1/16ths from the block's
    /// min-normal coordinate. Whole-block positive face = 16, negative = 0.
    normal_pos: u8,
    /// Quad start within the block along u, in 1/16ths.
    u_intra_offset: u8,
    /// Quad extent within one block cell along u, in 1/16ths.
    u_intra_extent: u8,
    /// Quad start within the block along v, in 1/16ths.
    v_intra_offset: u8,
    /// Quad extent within one block cell along v, in 1/16ths.
    v_intra_extent: u8,
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

/// Whether the neighbor fully covers the block's face region on the shared
/// boundary. For whole-blocks this is simple; for slabs we must check whether
/// the neighbor occupies at least the same sub-region as the block along the
/// slab axis.
fn neighbor_covers_face_region<B: Block>(block: &B, neighbor: &B, face: Face) -> bool {
    match neighbor.shape() {
        Shape::WholeBlock => true,
        // Cross and facade blocks never cover any face region.
        Shape::Cross(_) | Shape::Facade(_) => false,
        Shape::Slab(n_info) => {
            // The neighbor slab is flush against our face on this boundary
            // only if its slab face equals our face's opposite.
            if n_info.face == face.opposite() {
                return true;
            }
            // For side faces: if the block is also a slab with the same axis
            // and the neighbor covers at least the block's extent, it culls.
            if let Shape::Slab(b_info) = block.shape() {
                if b_info.face.axis() == n_info.face.axis() && face.axis() != b_info.face.axis() {
                    // Both slabs share the same axis; the neighbor covers our
                    // region if its thickness is >= ours on the same side.
                    return b_info.face == n_info.face && n_info.thickness >= b_info.thickness;
                }
            }
            false
        }
    }
}

/// Whether the current block's face is culled by the given neighbor.
/// Only call this for faces that sit at the block boundary (flush / side).
fn is_culled_at_boundary<B: Block>(block: &B, neighbor: &B, face: Face) -> bool {
    if !neighbor_covers_face_region(block, neighbor, face) {
        return false;
    }
    match neighbor.cull_mode() {
        CullMode::Opaque => true,
        CullMode::TransparentMerged | CullMode::Empty => block == neighbor,
        CullMode::TransparentUnmerged => false,
    }
}

/// Compute the mask entry for a block/face based purely on shape, ignoring
/// neighbor culling. Returns `None` only for the inner (non-flush) face of a
/// slab along its own axis.
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
        // Facade emits exactly one quad on its own face, offset 1/16 inward.
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
            });
        }
        Shape::WholeBlock => {
            let normal_pos = if face.is_positive() { ft } else { 0 };
            Some(MaskEntry {
                block: *block,
                normal_pos,
                u_intra_offset: 0,
                u_intra_extent: ft,
                v_intra_offset: 0,
                v_intra_extent: ft,
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
                // The inner face of a slab (opposite its flush face) is never
                // at the block boundary, so it always emits geometry. The
                // flush face may be culled by a neighbor, but that is handled
                // by the caller.
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
                })
            }
        }
    }
}

/// Compute the mask entry for a slab block/face combination, or `None` if the
/// face is not visible. Only called when `block.shape()` is `Slab`.
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
        Shape::WholeBlock | Shape::Cross(_) | Shape::Facade(_) => unreachable!(),
    };

    // For the flush face of the slab along its own axis, check neighbor culling.
    if face.axis() == info.face.axis() && face == info.face {
        if is_culled_at_boundary(block, neighbor, face) {
            return None;
        }
    } else if face.axis() != info.face.axis() {
        // Side faces: check neighbor culling.
        if is_culled_at_boundary(block, neighbor, face) {
            return None;
        }
    }

    mask_entry_for_shape(block, face, u_idx, v_idx)
}

impl Quads {
    /// Create an empty `Quads` with no allocations.
    pub fn new() -> Self {
        Quads {
            faces: [vec![], vec![], vec![], vec![], vec![], vec![]],
            diagonals: [vec![], vec![]],
        }
    }

    /// Clear all face lists without freeing their backing allocations.
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
    ///         // …
    ///     }
    /// }
    /// ```
    pub fn get(&self, face: QuadFace) -> &[Quad] {
        match face {
            QuadFace::Aligned(f) => &self.faces[f.index()],
            QuadFace::Diagonal(d) => &self.diagonals[d.index()],
        }
    }
}

impl Default for Quads {
    fn default() -> Self {
        Self::new()
    }
}

pub fn greedy_mesh<B: Block>(chunk: &PaddedChunk<B>) -> Quads {
    let mut quads = Quads::new();
    greedy_mesh_into(chunk, &mut quads);
    quads
}

/// Returns (n_stride, u_stride, v_stride) as linear index steps into the
/// padded chunk array, matching the tangent convention in [`face_tangents`].
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

/// Convert a [`MaskEntry`] into a [`Quad`].
///
/// - `normal_idx`, `u_idx`, `v_idx`: axis indices from [`face_axis_indices`].
/// - `normal_block`, `u_block`, `v_block`: block-level position along each
///   axis, already including any padding offset.
/// - `width`, `height`: number of blocks merged along u / v (1 when not
///   doing greedy merging).
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
) -> Quad {
    let ft32 = FULL_THICKNESS;
    let mut origin = [0u32; 3];
    origin[normal_idx] = normal_block * ft32 + entry.normal_pos as u32;
    origin[u_idx] = u_block * ft32 + entry.u_intra_offset as u32;
    origin[v_idx] = v_block * ft32 + entry.v_intra_offset as u32;

    Quad {
        origin_padded: UVec3::new(origin[0], origin[1], origin[2]),
        size: UVec2::new(
            width * entry.u_intra_extent as u32,
            height * entry.v_intra_extent as u32,
        ),
    }
}

/// Emit diagonal quads for a cross-shaped block into `quads`.
///
/// `x_block` and `z_block` are the block-level XZ position (including
/// padding). `y_block` is the base Y position. `height` is the number of
/// blocks merged vertically.
#[inline]
fn emit_cross_quads(quads: &mut Quads, x_block: u32, y_block: u32, z_block: u32, height: u32) {
    let ft32 = FULL_THICKNESS;
    for diag in DiagonalFace::ALL {
        quads.diagonals[diag.index()].push(Quad {
            origin_padded: UVec3::new(x_block * ft32, y_block * ft32, z_block * ft32),
            size: UVec2::new(ft32, height * ft32),
        });
    }
}

/// Produce quads for a single block placed at the origin with all faces
/// exposed (no neighbor culling). Useful for rendering held items or dropped
/// block entities.
///
/// The resulting [`Quad`] positions span `[0, 1]` (or the appropriate
/// sub-range for slabs), the same coordinate space as a block at `(0,0,0)` in
/// a chunk.
pub fn block_faces<B: Block>(block: &B) -> Quads {
    let mut quads = Quads::new();
    block_faces_into(block, &mut quads);
    quads
}

/// Like [`block_faces`], but reuses an existing [`Quads`] buffer.
pub fn block_faces_into<B: Block>(block: &B, quads: &mut Quads) {
    quads.reset();

    if !block.cull_mode().is_renderable() {
        return;
    }

    if matches!(block.shape(), Shape::Cross(_)) {
        emit_cross_quads(quads, PADDING as u32, PADDING as u32, PADDING as u32, 1);
        return;
    }

    for face in Face::ALL {
        let (normal_idx, u_idx, v_idx) = face_axis_indices(face);

        if let Some(entry) = mask_entry_for_shape(block, face, u_idx, v_idx) {
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
            );
            quads.faces[face.index()].push(quad);
        }
    }
}

/// Like [`greedy_mesh`], but reuses an existing [`Quads`] buffer.
///
/// The buffer is [`reset`](Quads::reset) before meshing, so previous contents
/// are cleared but backing allocations are preserved across calls.
pub fn greedy_mesh_into<B: Block>(chunk: &PaddedChunk<B>, quads: &mut Quads) {
    quads.reset();
    let ft = FULL_THICKNESS as u8;
    let data = &chunk.data;

    // Mask is hoisted outside the layer loop — the build phase overwrites every
    // cell unconditionally so the previous layer's values don't matter.
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

                    // SAFETY: The PADDING ring guarantees all indices
                    // (including the neighbor one step along the normal) are
                    // within the PADDED_VOLUME array.
                    let (block, neighbor) =
                        unsafe { (data.get_unchecked(idx), data.get_unchecked(n_idx)) };

                    mask[v][u] = if !block.cull_mode().is_renderable() {
                        None
                    } else if matches!(block.shape(), Shape::WholeBlock) {
                        if is_culled_at_boundary(block, neighbor, face) {
                            None
                        } else {
                            // WholeBlock fast path: normal_pos is constant for
                            // the entire face, avoid full shape dispatch.
                            Some(MaskEntry {
                                block: *block,
                                normal_pos: whole_normal_pos,
                                u_intra_offset: 0,
                                u_intra_extent: ft,
                                v_intra_offset: 0,
                                v_intra_extent: ft,
                            })
                        }
                    } else if matches!(block.shape(), Shape::Cross(_)) {
                        // Cross blocks are handled in a separate pass.
                        None
                    } else if matches!(block.shape(), Shape::Facade(_)) {
                        // Facade quads are offset 1/16 inward, never at the
                        // block boundary, so skip neighbor culling entirely.
                        mask_entry_for_shape(block, face, u_idx, v_idx)
                    } else {
                        compute_slab_mask_entry(block, neighbor, face, u_idx, v_idx)
                    };

                    idx += u_stride;
                }
                v_base += v_stride;
            }

            // Greedy merge.
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
                    );

                    quads.faces[face.index()].push(quad);
                    u += width;
                }
            }
        }
    }

    // Cross-block pass: scan Y columns and merge vertically.
    let y_stride = PADDED; // index step for +1 Y
    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let col_base = crate::chunk::padded_idx(x + PADDING, PADDING, z + PADDING);
            let mut y = 0;
            while y < CHUNK_SIZE {
                let idx = col_base + y * y_stride;
                let block = unsafe { data.get_unchecked(idx) };

                let stretch = match block.shape() {
                    Shape::Cross(s) if block.cull_mode().is_renderable() => s,
                    _ => {
                        y += 1;
                        continue;
                    }
                };

                // Merge upward while the block is identical.
                let mut height = 1u32;
                while y + height as usize <= CHUNK_SIZE - 1 {
                    let above_idx = col_base + (y + height as usize) * y_stride;
                    let above = unsafe { data.get_unchecked(above_idx) };
                    if above != block {
                        break;
                    }
                    height += 1;
                }
                let _ = stretch; // stretch is stored on the block, not in the quad

                emit_cross_quads(
                    quads,
                    (PADDING + x) as u32,
                    (PADDING + y) as u32,
                    (PADDING + z) as u32,
                    height,
                );

                y += height as usize;
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
        chunk.set(0, 0, 0, TestBlock::Stone);
        let from_chunk = greedy_mesh(&chunk);
        let from_block = block_faces(&TestBlock::Stone);
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
        let q = block_faces(&TestBlock::Air);
        assert_eq!(q.total(), 0);
    }

    #[test]
    fn single_block_quad_size_is_one_block() {
        let mut chunk = PaddedChunk::new_filled(TestBlock::Air);
        chunk.set(0, 0, 0, TestBlock::Stone);
        let q = greedy_mesh(&chunk);
        for face in Face::ALL {
            let quad = &q.faces[face.index()][0];
            assert_eq!(quad.size, UVec2::new(16, 16), "face {:?}", face);
        }
    }

    #[test]
    fn full_chunk_quad_size_is_sixteen_blocks() {
        let mut chunk = PaddedChunk::new_filled(TestBlock::Air);
        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    chunk.set(x, y, z, TestBlock::Stone);
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

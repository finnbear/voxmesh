use glam::{UVec2, UVec3, Vec2, Vec3};

use crate::block::{Block, CullMode, FULL_THICKNESS, Shape};
use crate::chunk::{CHUNK_SIZE, PADDED, PADDING, PaddedChunk};
use crate::face::{Axis, Face};

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
    pub fn positions(&self, face: Face) -> [Vec3; 4] {
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

    pub fn texture_coordinates(&self, face: Face, u_flip_face: Axis, flip_v: bool) -> [Vec2; 4] {
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
}

pub struct Quads {
    pub faces: [Vec<Quad>; 6],
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

/// Whether a neighbor block fully covers the block boundary on the given
/// `neighbor_face` (the face of the neighbor that touches our block).
fn neighbor_covers_boundary<B: Block>(neighbor: &B, neighbor_face: Face) -> bool {
    match neighbor.shape() {
        Shape::WholeBlock => true,
        Shape::Slab(info) => info.face == neighbor_face,
    }
}

/// Whether the current block's face is culled by the given neighbor.
/// Only call this for faces that sit at the block boundary (flush / side).
fn is_culled_at_boundary<B: Block>(block: &B, neighbor: &B, face: Face) -> bool {
    if !neighbor_covers_boundary(neighbor, face.opposite()) {
        return false;
    }
    match neighbor.cull_mode() {
        CullMode::Opaque => true,
        CullMode::TransparentMerged | CullMode::Empty => block == neighbor,
        CullMode::TransparentUnmerged => false,
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
    let ft = FULL_THICKNESS as u8;
    let info = match block.shape() {
        Shape::Slab(info) => info,
        Shape::WholeBlock => unreachable!(),
    };

    let slab_axis_idx = info.face.axis().index();
    let thickness = info.thickness as u8;

    let (slab_min, slab_max) = if info.face.is_positive() {
        (ft - thickness, ft)
    } else {
        (0, thickness)
    };

    if face.axis() == info.face.axis() {
        if face == info.face {
            if is_culled_at_boundary(block, neighbor, face) {
                return None;
            }
        }
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
        if is_culled_at_boundary(block, neighbor, face) {
            return None;
        }
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

impl Quads {
    /// Create an empty `Quads` with no allocations.
    pub fn new() -> Self {
        Quads {
            faces: [vec![], vec![], vec![], vec![], vec![], vec![]],
        }
    }

    /// Clear all face lists without freeing their backing allocations.
    pub fn reset(&mut self) {
        for face in &mut self.faces {
            face.clear();
        }
    }

    /// Total number of quads across all faces.
    pub fn total(&self) -> usize {
        self.faces.iter().map(|v| v.len()).sum()
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

                    // Inlined WholeBlock fast path — avoids the full
                    // compute_slab_mask_entry call for the common case.
                    mask[v][u] = if !block.cull_mode().is_renderable() {
                        None
                    } else if matches!(block.shape(), Shape::WholeBlock) {
                        if is_culled_at_boundary(block, neighbor, face) {
                            None
                        } else {
                            Some(MaskEntry {
                                block: *block,
                                normal_pos: whole_normal_pos,
                                u_intra_offset: 0,
                                u_intra_extent: ft,
                                v_intra_offset: 0,
                                v_intra_extent: ft,
                            })
                        }
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
                    let ft32 = FULL_THICKNESS;
                    let mut origin = [0u32; 3];
                    origin[normal_idx] = (PADDING + layer) as u32 * ft32 + entry.normal_pos as u32;
                    origin[u_idx] = (PADDING + u) as u32 * ft32 + entry.u_intra_offset as u32;
                    origin[v_idx] = (PADDING + v) as u32 * ft32 + entry.v_intra_offset as u32;

                    let quad = Quad {
                        origin_padded: UVec3::new(origin[0], origin[1], origin[2]),
                        size: UVec2::new(
                            width as u32 * entry.u_intra_extent as u32,
                            height as u32 * entry.v_intra_extent as u32,
                        ),
                    };

                    quads.faces[face.index()].push(quad);
                    u += width;
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

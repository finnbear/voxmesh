mod common;

use glam::Vec3;

use common::TestBlock;
use voxmesh::*;

use block_mesh::{
    greedy_quads, GreedyQuadsBuffer, MergeVoxel, Voxel, VoxelVisibility, RIGHT_HANDED_Y_UP_CONFIG,
};
use ndshape::{ConstShape, ConstShape3u32};

// block-mesh voxel type matching our TestBlock for opaque/empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BmVoxel {
    Air,
    Stone,
}

impl Voxel for BmVoxel {
    fn get_visibility(&self) -> VoxelVisibility {
        match self {
            BmVoxel::Air => VoxelVisibility::Empty,
            BmVoxel::Stone => VoxelVisibility::Opaque,
        }
    }
}

impl MergeVoxel for BmVoxel {
    type MergeValue = Self;
    fn merge_value(&self) -> Self::MergeValue {
        *self
    }
}

type PaddedShape = ConstShape3u32<18, 18, 18>;

/// Maps a block-mesh face index to a voxmesh `AlignedFace`.
/// block-mesh RIGHT_HANDED_Y_UP_CONFIG order: -X, -Y, -Z, +X, +Y, +Z.
fn bm_face_index_to_face(i: usize) -> AlignedFace {
    match i {
        0 => AlignedFace::NegX,
        1 => AlignedFace::NegY,
        2 => AlignedFace::NegZ,
        3 => AlignedFace::PosX,
        4 => AlignedFace::PosY,
        5 => AlignedFace::PosZ,
        _ => unreachable!(),
    }
}

/// Compares indices for a single opaque block at (0,0,0) between
/// voxmesh and block-mesh. Both should produce triangles with the same
/// winding when vertices are matched by position.
#[test]
fn single_block_indices_match_block_mesh() {
    let config = &RIGHT_HANDED_Y_UP_CONFIG;

    // Run block-mesh.
    let mut bm_voxels = [BmVoxel::Air; PaddedShape::SIZE as usize];
    let idx = PaddedShape::linearize([1, 1, 1]) as usize;
    bm_voxels[idx] = BmVoxel::Stone;

    let mut buffer = GreedyQuadsBuffer::new(bm_voxels.len());
    greedy_quads(
        &bm_voxels,
        &PaddedShape {},
        [0; 3],
        [17; 3],
        &config.faces,
        &mut buffer,
    );

    // Run voxmesh.
    let vm_quads = common::mesh_single(TestBlock::Stone);

    // Compare per face.
    for (bm_face_idx, bm_face_quads) in buffer.quads.groups.iter().enumerate() {
        let face = bm_face_index_to_face(bm_face_idx);
        let oriented_face = &config.faces[bm_face_idx];

        let bm_quad = &bm_face_quads[0];
        let vm_quad = &vm_quads.faces[face.index()][0];

        // block-mesh positions (subtract 1 for padding offset).
        let bm_positions: [Vec3; 4] = oriented_face
            .quad_mesh_positions(bm_quad, 1.0)
            .map(|p| Vec3::new(p[0] - 1.0, p[1] - 1.0, p[2] - 1.0));

        // voxmesh positions.
        let vm_positions = vm_quad.positions(face, Shape::WholeBlock);

        // Map each block-mesh vertex index to the matching voxmesh
        // vertex index by position.
        let bm_to_vm: [usize; 4] = std::array::from_fn(|bm_i| {
            vm_positions
                .iter()
                .position(|vp| (*vp - bm_positions[bm_i]).length() < 1e-6)
                .unwrap_or_else(|| {
                    panic!(
                        "face {face:?}: block-mesh vertex {bm_i} at {:?} \
                         not found in voxmesh positions {vm_positions:?}",
                        bm_positions[bm_i]
                    )
                })
        });

        // block-mesh indices (rebased to 0).
        let bm_indices = oriented_face.quad_mesh_indices(0);

        // voxmesh indices (rebased to 0).
        let vm_indices = Quad::<()>::indices(0);

        // Remap block-mesh indices into voxmesh vertex space.
        let bm_indices_in_vm: [u32; 6] = bm_indices.map(|i| bm_to_vm[i as usize] as u32);

        // Both index sets should form triangles with the same winding.
        for tri in 0..2 {
            let base = tri * 3;
            let bm_tri = [
                vm_positions[bm_indices_in_vm[base] as usize],
                vm_positions[bm_indices_in_vm[base + 1] as usize],
                vm_positions[bm_indices_in_vm[base + 2] as usize],
            ];
            let vm_tri = [
                vm_positions[vm_indices[base] as usize],
                vm_positions[vm_indices[base + 1] as usize],
                vm_positions[vm_indices[base + 2] as usize],
            ];

            let bm_normal = (bm_tri[1] - bm_tri[0]).cross(bm_tri[2] - bm_tri[0]);
            let vm_normal = (vm_tri[1] - vm_tri[0]).cross(vm_tri[2] - vm_tri[0]);

            assert!(
                bm_normal.dot(vm_normal) > 0.0,
                "face {face:?}, triangle {tri}: winding mismatch.\n\
                 block-mesh tri (in vm space): {bm_tri:?} normal={bm_normal:?}\n\
                 voxmesh tri: {vm_tri:?} normal={vm_normal:?}",
            );
        }
    }
}

/// Compares texture coordinates for a single opaque block at (0,0,0)
/// between voxmesh and block-mesh. UVs are matched by vertex position
/// so that ordering differences don't cause false failures.
#[test]
fn single_block_tex_coords_match_block_mesh() {
    let config = &RIGHT_HANDED_Y_UP_CONFIG;
    let flip_v = false;

    // Run block-mesh.
    let mut bm_voxels = [BmVoxel::Air; PaddedShape::SIZE as usize];
    // Place stone at padded (1,1,1) = unpadded (0,0,0).
    let idx = PaddedShape::linearize([1, 1, 1]) as usize;
    bm_voxels[idx] = BmVoxel::Stone;

    let mut buffer = GreedyQuadsBuffer::new(bm_voxels.len());
    greedy_quads(
        &bm_voxels,
        &PaddedShape {},
        [0; 3],
        [17; 3],
        &config.faces,
        &mut buffer,
    );

    // Run voxmesh.
    let vm_quads = common::mesh_single(TestBlock::Stone);

    // Compare per face.
    for (bm_face_idx, bm_face_quads) in buffer.quads.groups.iter().enumerate() {
        let face = bm_face_index_to_face(bm_face_idx);
        let oriented_face = &config.faces[bm_face_idx];

        assert_eq!(
            bm_face_quads.len(),
            1,
            "block-mesh: expected 1 quad on face {face:?}",
        );
        let vm_face_quads = &vm_quads.faces[face.index()];
        assert_eq!(
            vm_face_quads.len(),
            1,
            "voxmesh: expected 1 quad on face {face:?}",
        );

        let bm_quad = &bm_face_quads[0];
        let vm_quad = &vm_face_quads[0];

        // block-mesh positions in padded space, subtract 1 for chunk-local.
        let bm_positions: [[f32; 3]; 4] = oriented_face
            .quad_mesh_positions(bm_quad, 1.0)
            .map(|p| [p[0] - 1.0, p[1] - 1.0, p[2] - 1.0]);
        let bm_uvs: [[f32; 2]; 4] = oriented_face.tex_coords(config.u_flip_face, flip_v, bm_quad);

        // voxmesh positions and UVs.
        let vm_positions = vm_quad.positions(face, Shape::WholeBlock);
        let vm_uvs = vm_quad.texture_coordinates(face, Axis::X, flip_v);

        // Match vertices by position, then compare UVs.
        for (bm_i, bm_pos) in bm_positions.iter().enumerate() {
            let vm_i = vm_positions
                .iter()
                .position(|vp| {
                    (vp.x - bm_pos[0]).abs() < 1e-6
                        && (vp.y - bm_pos[1]).abs() < 1e-6
                        && (vp.z - bm_pos[2]).abs() < 1e-6
                })
                .unwrap_or_else(|| {
                    panic!(
                        "face {face:?}: block-mesh vertex {bm_i} at {bm_pos:?} \
                         not found in voxmesh positions {vm_positions:?}"
                    )
                });

            let bm_uv = bm_uvs[bm_i];
            let vm_uv = vm_uvs[vm_i];
            assert!(
                (vm_uv.x - bm_uv[0]).abs() < 1e-6 && (vm_uv.y - bm_uv[1]).abs() < 1e-6,
                "face {face:?}: UV mismatch at position {bm_pos:?}: \
                 block-mesh={bm_uv:?}, voxmesh={vm_uv:?}",
            );
        }
    }
}

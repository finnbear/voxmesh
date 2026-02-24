mod common;

use common::TestBlock;
use voxmesh::*;

use block_mesh::{
    GreedyQuadsBuffer, MergeVoxel, RIGHT_HANDED_Y_UP_CONFIG, Voxel, VoxelVisibility, greedy_quads,
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

/// Map a block-mesh face index to a voxmesh Face.
/// block-mesh RIGHT_HANDED_Y_UP_CONFIG order: [-X, -Y, -Z, +X, +Y, +Z]
fn bm_face_index_to_face(i: usize) -> Face {
    match i {
        0 => Face::NegX,
        1 => Face::NegY,
        2 => Face::NegZ,
        3 => Face::PosX,
        4 => Face::PosY,
        5 => Face::PosZ,
        _ => unreachable!(),
    }
}

/// Compare texture coordinates for a single opaque block at (0,0,0)
/// between voxmesh and block-mesh. UVs are matched by vertex world-position
/// so that vertex ordering differences don't cause false failures.
#[test]
fn single_block_tex_coords_match_block_mesh() {
    let config = &RIGHT_HANDED_Y_UP_CONFIG;
    let flip_v = false;

    // -- Run block-mesh ---------------------------------------------------
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

    // -- Run voxmesh ------------------------------------------------------
    let vm_quads = common::mesh_single(TestBlock::Stone);

    // -- Compare per face -------------------------------------------------
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

        // block-mesh positions (voxel_size = 1.0). These are in the padded
        // coordinate space, so subtract 1 to get chunk-local coords.
        let bm_positions: [[f32; 3]; 4] = oriented_face
            .quad_mesh_positions(bm_quad, 1.0)
            .map(|p| [p[0] - 1.0, p[1] - 1.0, p[2] - 1.0]);
        let bm_uvs: [[f32; 2]; 4] = oriented_face.tex_coords(config.u_flip_face, flip_v, bm_quad);

        // voxmesh positions and UVs.
        let vm_positions = vm_quad.positions(face);
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

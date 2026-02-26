mod common;

use common::*;
use voxmesh::AlignedFace;

#[test]
fn positive_face_winding_is_ccw() {
    let q = mesh_single(TestBlock::Stone);
    for face in [AlignedFace::PosX, AlignedFace::PosY, AlignedFace::PosZ] {
        assert_ccw_winding(&q, face);
    }
}

#[test]
fn negative_face_winding_is_ccw() {
    let q = mesh_single(TestBlock::Stone);
    for face in [AlignedFace::NegX, AlignedFace::NegY, AlignedFace::NegZ] {
        assert_ccw_winding(&q, face);
    }
}

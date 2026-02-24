mod common;

use common::*;
use voxmesh::Face;

#[test]
fn positive_face_winding_is_ccw() {
    let q = mesh_single(TestBlock::Stone);
    for face in [Face::PosX, Face::PosY, Face::PosZ] {
        assert_ccw_winding(&q, face);
    }
}

#[test]
fn negative_face_winding_is_ccw() {
    let q = mesh_single(TestBlock::Stone);
    for face in [Face::NegX, Face::NegY, Face::NegZ] {
        assert_ccw_winding(&q, face);
    }
}

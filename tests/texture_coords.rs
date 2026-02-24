mod common;

use common::*;
use voxmesh::*;

#[test]
fn single_block_uvs_span_zero_to_one() {
    let q = mesh_single(TestBlock::Stone);

    for face in Face::ALL {
        let quad = &q.faces[face.index()][0];
        let uvs = quad.texture_coordinates(face, Axis::X, false);
        let u_min = uvs.iter().map(|v| v.x).fold(f32::INFINITY, f32::min);
        let u_max = uvs.iter().map(|v| v.x).fold(f32::NEG_INFINITY, f32::max);
        let v_min = uvs.iter().map(|v| v.y).fold(f32::INFINITY, f32::min);
        let v_max = uvs.iter().map(|v| v.y).fold(f32::NEG_INFINITY, f32::max);
        assert!(u_min.abs() < 1e-6, "face {:?} u_min", face);
        assert!((u_max - 1.0).abs() < 1e-6, "face {:?} u_max", face);
        assert!(v_min.abs() < 1e-6, "face {:?} v_min", face);
        assert!((v_max - 1.0).abs() < 1e-6, "face {:?} v_max", face);
    }
}

#[test]
fn merged_quad_uvs_scale_with_size() {
    // PosY tangents: u=Z, v=X. 3 blocks along Z = 3 in u direction.
    let q = mesh_with(&[
        (0, 0, 0, TestBlock::Stone),
        (0, 0, 1, TestBlock::Stone),
        (0, 0, 2, TestBlock::Stone),
    ]);

    assert_eq!(face_count(&q, Face::PosY), 1);
    let quad = &q.faces[Face::PosY.index()][0];
    let uvs = quad.texture_coordinates(Face::PosY, Axis::Y, false);
    let u_max = uvs.iter().map(|v| v.x).fold(f32::NEG_INFINITY, f32::max);
    assert!(
        (u_max - 3.0).abs() < 1e-6,
        "PosY merged quad u_max should be 3.0, got {u_max}"
    );
}

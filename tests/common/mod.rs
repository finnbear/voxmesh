#![allow(dead_code)]

use voxmesh::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestBlock {
    Air,
    Stone,
    Dirt,
    Glass,
    Leaves,
    UpperSlab, // PosY slab, thickness 8
    LowerSlab, // NegY slab, thickness 8
    SugarCane, // Cross(NegY, 0) — diagonal billboard, no stretch
    Cobweb,    // Cross(NegY, 4) — diagonal billboard, stretched
    Ladder,    // Facade(PosX) — flat face on +X side
    Rail,      // Facade(NegY) — flat face on bottom
    Cactus,    // Inset(1) — horizontal faces inset by 1/16
    Chain,     // Cross(PosY, 0) — hanging cross, merges along Y
    Vine,      // Cross(PosX, 0) — wall-mounted cross, merges along X
}

impl Block for TestBlock {
    type TransparentGroup = ();

    fn shape(&self) -> Shape {
        match self {
            TestBlock::UpperSlab => Shape::Slab(SlabInfo {
                face: Face::PosY,
                thickness: 8,
            }),
            TestBlock::LowerSlab => Shape::Slab(SlabInfo {
                face: Face::NegY,
                thickness: 8,
            }),
            TestBlock::SugarCane => Shape::Cross(CrossInfo {
                face: Face::NegY,
                stretch: 0,
            }),
            TestBlock::Cobweb => Shape::Cross(CrossInfo {
                face: Face::NegY,
                stretch: 4,
            }),
            TestBlock::Chain => Shape::Cross(CrossInfo {
                face: Face::PosY,
                stretch: 0,
            }),
            TestBlock::Vine => Shape::Cross(CrossInfo {
                face: Face::PosX,
                stretch: 0,
            }),
            TestBlock::Ladder => Shape::Facade(Face::PosX),
            TestBlock::Rail => Shape::Facade(Face::NegY),
            TestBlock::Cactus => Shape::Inset(1),
            _ => Shape::WholeBlock,
        }
    }

    fn cull_mode(&self) -> CullMode {
        match self {
            TestBlock::Air => CullMode::Empty,
            TestBlock::Glass => CullMode::TransparentMerged(()),
            TestBlock::Leaves
            | TestBlock::SugarCane
            | TestBlock::Cobweb
            | TestBlock::Chain
            | TestBlock::Vine
            | TestBlock::Ladder
            | TestBlock::Rail => CullMode::TransparentUnmerged,
            _ => CullMode::Opaque, // Stone, Dirt, Cactus
        }
    }
}

/// Mesh a chunk containing only the given block placements.
/// Returns the `Quads` produced by `greedy_mesh`.
pub fn mesh_with(blocks: &[(usize, usize, usize, TestBlock)]) -> Quads {
    let mut chunk = PaddedChunk::new_filled(TestBlock::Air);
    for &(x, y, z, block) in blocks {
        chunk.set(x, y, z, block);
    }
    greedy_mesh(&chunk)
}

/// Mesh a single block at (0,0,0).
pub fn mesh_single(block: TestBlock) -> Quads {
    mesh_with(&[(0, 0, 0, block)])
}

/// Number of quads on a specific face.
pub fn face_count(q: &Quads, face: Face) -> usize {
    q.faces[face.index()].len()
}

/// Get vertex positions of the first quad on a given face.
pub fn first_face_positions(q: &Quads, face: Face) -> [glam::Vec3; 4] {
    q.faces[face.index()][0].positions(face, 0, Face::NegY)
}

/// Assert all vertices of the first quad on `face` have `axis_val` on the
/// given component (0=x, 1=y, 2=z).
pub fn assert_face_on_plane(q: &Quads, face: Face, axis: usize, expected: f32) {
    let verts = first_face_positions(q, face);
    let label = ["x", "y", "z"][axis];
    for v in &verts {
        let val = [v.x, v.y, v.z][axis];
        assert!(
            (val - expected).abs() < 1e-6,
            "face {face:?}: vertex {label} should be {expected}, got {val}",
        );
    }
}

/// Return (min, max) of a component (0=x, 1=y, 2=z) across all vertices
/// of the first quad on `face`.
pub fn face_vertex_range(q: &Quads, face: Face, axis: usize) -> (f32, f32) {
    let verts = first_face_positions(q, face);
    let vals: Vec<f32> = verts.iter().map(|v| [v.x, v.y, v.z][axis]).collect();
    (
        vals.iter().copied().fold(f32::INFINITY, f32::min),
        vals.iter().copied().fold(f32::NEG_INFINITY, f32::max),
    )
}

/// Assert the winding of the first quad on `face` is CCW when viewed from
/// outside (i.e. triangle (v0,v1,v2) normal aligns with face normal).
pub fn assert_ccw_winding(q: &Quads, face: Face) {
    let v = first_face_positions(q, face);
    let normal = face.normal().as_vec3();
    let cross = (v[1] - v[0]).cross(v[2] - v[0]);
    assert!(
        cross.dot(normal) > 0.0,
        "face {face:?}: winding should be CCW from outside. cross={cross:?}, normal={normal:?}",
    );
}

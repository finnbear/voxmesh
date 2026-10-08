//! Ramps: a wedge whose slope rises from the floor at the front to the far
//! side at the back, in any orientation.
//!
//! The slope is the face opposite the floor with its front corners lowered
//! onto the floor, each side is a whole face with one corner folded down
//! into a triangle, and the front is only an edge.

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

mod common;

use common::*;
use glam::{UVec2, UVec3, Vec2, Vec3};
use voxmesh::*;

const EPSILON: f32 = 1e-6;

fn info_of(block: TestBlock) -> RampInfo {
    match block.shape() {
        Shape::Ramp(info) => info,
        other => panic!("{block:?} is {other:?}, not a ramp"),
    }
}

/// The quads on `face` that belong to the cell at `at`.
fn quads_at(q: &Quads, face: AlignedFace, at: UVec3) -> Vec<&Quad> {
    q.faces[face.index()]
        .iter()
        .filter(|quad| quad.voxel_position(face) == at)
        .collect()
}

/// The distinct points among `points`.
fn distinct(points: &[Vec3]) -> Vec<Vec3> {
    let mut out: Vec<Vec3> = Vec::new();
    for p in points {
        if !out.iter().any(|o| o.distance(*p) < EPSILON) {
            out.push(*p);
        }
    }
    out
}

/// Twice the area-weighted normal of a polygon (Newell's method), which a
/// repeated corner leaves alone — so it reads a folded quad as the triangle
/// it is.
fn winding_normal(verts: &[Vec3; 4]) -> Vec3 {
    (0..4).fold(Vec3::ZERO, |n, i| n + verts[i].cross(verts[(i + 1) % 4]))
}

/// A ramp in whatever orientation the test asks for, or air.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cell {
    Air,
    Wedge(RampInfo),
}

impl Block for Cell {
    type TransparentGroup = ();

    fn shape(&self) -> Shape {
        match self {
            Cell::Air => Shape::WholeBlock,
            Cell::Wedge(info) => Shape::Ramp(*info),
        }
    }

    fn cull_mode(&self) -> CullMode {
        match self {
            Cell::Air => CullMode::Empty,
            Cell::Wedge(_) => CullMode::Opaque,
        }
    }
}

/// A lone ramp meshed as a chunk.
fn mesh_wedge(info: RampInfo) -> Quads {
    let mut chunk = PaddedChunk16::new_filled(Cell::Air);
    chunk.set(UVec3::ZERO, Cell::Wedge(info));
    mesh_chunk(&chunk, true)
}

#[test]
fn a_ramp_has_five_quads_and_no_front() {
    for block in [
        TestBlock::Ramp,
        TestBlock::HungRamp,
        TestBlock::WallRamp,
        TestBlock::SideRamp,
    ] {
        let q = mesh_single(block);
        assert_eq!(q.total(), 5, "{block:?}");
        assert_eq!(face_count(&q, info_of(block).front()), 0, "{block:?}");
    }
}

#[test]
fn a_lone_block_meshes_like_a_lone_cell() {
    for block in [
        TestBlock::Ramp,
        TestBlock::HungRamp,
        TestBlock::WallRamp,
        TestBlock::SideRamp,
    ] {
        let from_chunk = mesh_single(block);
        let from_block = mesh_block(&block, ());
        for face in AlignedFace::ALL {
            assert_eq!(
                from_chunk.faces[face.index()],
                from_block.faces[face.index()],
                "{block:?} {face:?}"
            );
        }
    }
}

#[test]
fn the_slope_falls_from_the_back_top_to_the_front_floor() {
    let shape = TestBlock::Ramp.shape();
    let q = mesh_single(TestBlock::Ramp);
    let slope = &q.faces[AlignedFace::PosY.index()];
    assert_eq!(slope.len(), 1);
    let verts = slope[0].positions(AlignedFace::PosY, shape);
    assert_eq!(distinct(&verts).len(), 4, "a whole rectangle");
    for v in verts {
        // Rising toward -Z: the back (z = 0) at the top, the front (z = 1)
        // on the floor, and straight between.
        assert!((v.y - (1.0 - v.z)).abs() < EPSILON, "{v:?}");
        assert!(v.x.abs() < EPSILON || (v.x - 1.0).abs() < EPSILON, "{v:?}");
    }
    let normal = Face::Aligned(AlignedFace::PosY).normal(shape);
    let s = std::f32::consts::FRAC_1_SQRT_2;
    assert!(
        normal.distance(Vec3::new(0.0, s, s)) < EPSILON,
        "{normal:?}"
    );
    assert!(
        winding_normal(&verts).dot(normal) > 0.0,
        "wound counter-clockwise seen from where the slope faces"
    );
    // Every other face of a ramp keeps its own normal.
    assert_eq!(Face::Aligned(AlignedFace::NegY).normal(shape), Vec3::NEG_Y);
}

#[test]
fn each_side_is_a_right_triangle() {
    let shape = TestBlock::Ramp.shape();
    let q = mesh_single(TestBlock::Ramp);
    for face in [AlignedFace::PosX, AlignedFace::NegX] {
        let quads = &q.faces[face.index()];
        assert_eq!(quads.len(), 1, "{face:?}");
        let verts = quads[0].positions(face, shape);
        let x = if face.is_positive() { 1.0 } else { 0.0 };
        let corners = distinct(&verts);
        assert_eq!(corners.len(), 3, "{face:?} {verts:?}");
        for want in [
            Vec3::new(x, 0.0, 0.0),
            Vec3::new(x, 1.0, 0.0),
            Vec3::new(x, 0.0, 1.0),
        ] {
            assert!(
                corners.iter().any(|c| c.distance(want) < EPSILON),
                "{face:?} lacks {want:?}: {corners:?}"
            );
        }
        assert!(
            winding_normal(&verts).dot(face.normal().as_vec3()) > 0.0,
            "{face:?} faces out"
        );
    }
}

#[test]
fn the_slope_stretches_its_texture_and_the_sides_crop_theirs() {
    let shape = TestBlock::Ramp.shape();
    let q = mesh_single(TestBlock::Ramp);

    // The whole texture once, stretched over the slope's length like a
    // rail ramp's.
    let slope = &q.faces[AlignedFace::PosY.index()][0];
    let uvs = slope.texture_coordinates(AlignedFace::PosY, shape, Axis::X, false);
    let lo = uvs
        .iter()
        .copied()
        .fold(Vec2::splat(f32::INFINITY), Vec2::min);
    let hi = uvs
        .iter()
        .copied()
        .fold(Vec2::splat(f32::NEG_INFINITY), Vec2::max);
    assert!(lo.distance(Vec2::ZERO) < EPSILON, "{uvs:?}");
    assert!(hi.distance(Vec2::ONE) < EPSILON, "{uvs:?}");

    // Where two corners of a side meet, so do their texture coordinates:
    // the triangle is cut out of the texture rather than squeezed into it.
    for face in [AlignedFace::PosX, AlignedFace::NegX] {
        let quad = &q.faces[face.index()][0];
        let verts = quad.positions(face, shape);
        let uvs = quad.texture_coordinates(face, shape, Axis::X, false);
        for i in 0..4 {
            for j in 0..4 {
                if verts[i].distance(verts[j]) < EPSILON {
                    assert!(uvs[i].distance(uvs[j]) < EPSILON, "{face:?} {uvs:?}");
                }
            }
            for c in uvs[i].to_array() {
                assert!(c.abs() < EPSILON || (c - 1.0).abs() < EPSILON, "{uvs:?}");
            }
        }
    }
}

#[test]
fn a_row_of_ramps_shares_one_slope_and_hides_its_inner_sides() {
    // Rising toward -Z, so a row across the ramp runs along X.
    let q = mesh_with(&[
        (0, 0, 0, TestBlock::Ramp),
        (1, 0, 0, TestBlock::Ramp),
        (2, 0, 0, TestBlock::Ramp),
        (3, 0, 0, TestBlock::Ramp),
    ]);
    let slopes = &q.faces[AlignedFace::PosY.index()];
    assert_eq!(slopes.len(), 1, "one slope across the row");
    // Along Z (u) one ramp's run, along X (v) four ramps' width.
    assert_eq!(slopes[0].size, UVec2::new(16, 64));
    for v in slopes[0].positions(AlignedFace::PosY, TestBlock::Ramp.shape()) {
        assert!((v.y - (1.0 - v.z)).abs() < EPSILON, "{v:?}");
    }
    assert_eq!(face_count(&q, AlignedFace::PosX), 1);
    assert_eq!(face_count(&q, AlignedFace::NegX), 1);
    assert_eq!(face_count(&q, AlignedFace::NegY), 1);
    assert_eq!(face_count(&q, AlignedFace::NegZ), 1);
    assert_eq!(face_count(&q, AlignedFace::PosZ), 0);
}

#[test]
fn ramps_in_a_line_along_their_run_stay_a_sawtooth() {
    let q = mesh_with(&[
        (0, 0, 0, TestBlock::Ramp),
        (0, 0, 1, TestBlock::Ramp),
        (0, 0, 2, TestBlock::Ramp),
    ]);
    let slopes = &q.faces[AlignedFace::PosY.index()];
    assert_eq!(slopes.len(), 3, "not one long slope");
    for slope in slopes {
        assert_eq!(slope.size, UVec2::new(16, 16));
    }
    // The triangles on one plane don't merge either.
    assert_eq!(face_count(&q, AlignedFace::PosX), 3);
    // Each ramp's back stands up out of the low front of the one behind it.
    assert_eq!(face_count(&q, AlignedFace::NegZ), 3);
}

#[test]
fn a_flight_of_ramps_is_one_unbroken_slope() {
    // Each ramp one up and one back from the last.
    let q = mesh_with(&[
        (0, 0, 2, TestBlock::Ramp),
        (0, 1, 1, TestBlock::Ramp),
        (0, 2, 0, TestBlock::Ramp),
    ]);
    let slopes = &q.faces[AlignedFace::PosY.index()];
    assert_eq!(slopes.len(), 3);
    for slope in slopes {
        for v in slope.positions(AlignedFace::PosY, TestBlock::Ramp.shape()) {
            assert!((v.y + v.z - 3.0).abs() < EPSILON, "{v:?} is off the plane");
        }
    }
}

#[test]
fn what_covers_the_triangle_hides_it() {
    for (neighbor, hidden) in [
        (TestBlock::Stone, true),
        // The same stair's L contains the triangle.
        (TestBlock::Stair, true),
        // The same ramp's triangle is the triangle.
        (TestBlock::Ramp, true),
        (TestBlock::LowerSlab, false),
        (TestBlock::Corbel, false),
        (TestBlock::SideRamp, false),
        (TestBlock::Glass, false),
    ] {
        let q = mesh_with(&[(0, 0, 0, TestBlock::Ramp), (1, 0, 0, neighbor)]);
        assert_eq!(
            quads_at(&q, AlignedFace::PosX, UVec3::ZERO).len(),
            if hidden { 0 } else { 1 },
            "{neighbor:?}"
        );
    }
    // The triangle never hides a whole face beside it.
    let q = mesh_with(&[(0, 0, 0, TestBlock::Ramp), (1, 0, 0, TestBlock::Stone)]);
    assert_eq!(
        quads_at(&q, AlignedFace::NegX, UVec3::new(1, 0, 0)).len(),
        1
    );
}

#[test]
fn a_ramp_seals_its_floor_and_back_and_nothing_else() {
    // On stone: the ramp's floor and the stone's top hide each other.
    let q = mesh_with(&[(0, 0, 0, TestBlock::Stone), (0, 1, 0, TestBlock::Ramp)]);
    assert_eq!(quads_at(&q, AlignedFace::PosY, UVec3::ZERO).len(), 0);
    assert_eq!(
        quads_at(&q, AlignedFace::NegY, UVec3::new(0, 1, 0)).len(),
        0
    );
    // Backed onto a wall: the back and the wall hide each other.
    let q = mesh_with(&[(0, 0, 0, TestBlock::Stone), (0, 0, 1, TestBlock::Ramp)]);
    assert_eq!(
        quads_at(&q, AlignedFace::NegZ, UVec3::new(0, 0, 1)).len(),
        0
    );
    assert_eq!(quads_at(&q, AlignedFace::PosZ, UVec3::ZERO).len(), 0);
    // A block in front shows its face: the front is only an edge.
    let q = mesh_with(&[(0, 0, 0, TestBlock::Ramp), (0, 0, 1, TestBlock::Stone)]);
    assert_eq!(
        quads_at(&q, AlignedFace::NegZ, UVec3::new(0, 0, 1)).len(),
        1
    );
    // A block on top shows its underside, and the slope under it is drawn.
    let q = mesh_with(&[(0, 0, 0, TestBlock::Ramp), (0, 1, 0, TestBlock::Stone)]);
    assert_eq!(
        quads_at(&q, AlignedFace::NegY, UVec3::new(0, 1, 0)).len(),
        1
    );
    assert_eq!(quads_at(&q, AlignedFace::PosY, UVec3::ZERO).len(), 1);
}

#[test]
fn every_orientation_is_the_same_wedge() {
    for floor in AlignedFace::ALL {
        for back in AlignedFace::ALL {
            if floor.axis() == back.axis() {
                continue;
            }
            let info = RampInfo { floor, back };
            let block = Cell::Wedge(info);
            let shape = block.shape();
            let q = mesh_block(&block, ());
            assert_eq!(q.total(), 5, "{info:?}");
            let from_chunk = mesh_wedge(info);
            for face in AlignedFace::ALL {
                assert_eq!(
                    from_chunk.faces[face.index()],
                    q.faces[face.index()],
                    "{info:?} {face:?}"
                );
            }

            let floor_axis = floor.axis().index();
            let back_axis = back.axis().index();
            // How far above the floor, and how far back from the front.
            let height = |v: Vec3| {
                if floor.is_positive() {
                    1.0 - v[floor_axis]
                } else {
                    v[floor_axis]
                }
            };
            let depth = |v: Vec3| {
                if back.is_positive() {
                    v[back_axis]
                } else {
                    1.0 - v[back_axis]
                }
            };

            for face in AlignedFace::ALL {
                for quad in &q.faces[face.index()] {
                    let verts = quad.positions(face, shape);
                    for v in verts {
                        assert!(
                            v.cmpge(Vec3::splat(-EPSILON)).all()
                                && v.cmple(Vec3::splat(1.0 + EPSILON)).all(),
                            "{info:?} {face:?} {v:?} leaves the cell"
                        );
                        // Nothing rises above the slope.
                        assert!(height(v) <= depth(v) + EPSILON, "{info:?} {face:?} {v:?}");
                    }
                    let normal = Face::Aligned(face).normal(shape);
                    assert!(
                        winding_normal(&verts).dot(normal) > 0.0,
                        "{info:?} {face:?} faces in"
                    );
                    if face == info.slope_face() {
                        assert_eq!(distinct(&verts).len(), 4, "{info:?}");
                        for v in verts {
                            assert!((height(v) - depth(v)).abs() < EPSILON, "{info:?} {v:?}");
                        }
                        for i in 0..4 {
                            let edge = verts[(i + 1) % 4] - verts[i];
                            assert!(edge.dot(normal).abs() < EPSILON, "{info:?}");
                        }
                        // Away from the wedge's middle: a third of the
                        // way up from the floor, two thirds back from the
                        // front.
                        let center = verts.iter().copied().sum::<Vec3>() * 0.25;
                        let third = 1.0 / 3.0;
                        let mut middle = Vec3::splat(0.5);
                        middle[floor_axis] = if floor.is_positive() {
                            1.0 - third
                        } else {
                            third
                        };
                        middle[back_axis] = if back.is_positive() {
                            1.0 - third
                        } else {
                            third
                        };
                        assert!((center - middle).dot(normal) > 0.0, "{info:?}");
                    } else if face.axis() == info.side_axis() {
                        assert_eq!(distinct(&verts).len(), 3, "{info:?} {face:?}");
                    } else {
                        assert!(face == floor || face == back, "{info:?} {face:?}");
                        assert_eq!(distinct(&verts).len(), 4, "{info:?}");
                    }
                }
            }
        }
    }
}

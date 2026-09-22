//! A fluid standing in a solid's cell: [`Block::fluid`] answering `Some`
//! for a block whose shape is a slab or a stair. The solid meshes as it
//! always did; the fluid meshes into [`Quads::fluid`], hidden by the
//! solid's own full faces and joined to the water next door.

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use glam::UVec3;
use voxmesh::*;

const HEIGHTS: [Thickness; 4] = [16, 12, 8, 4];
const WATER: u8 = 0;
const LADDER: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum B {
    Air,
    Stone,
    /// Water at a strength: `0` a source, each step out one shallower.
    Water(u8),
    LowerSlab,
    /// A lower slab with water standing in its cell.
    WetLowerSlab,
    /// A floor stair rising toward -Z, with water standing in its cell.
    WetStair,
    /// A ladder hung on the -Z wall, with water standing in its cell.
    ///
    /// The only waterlogged block here that is **transparent** — its
    /// texture has holes in it, so it merges as its own group rather than
    /// reading as opaque. That is the whole reason it exists: an opaque
    /// solid in a wet cell hides a face by sealing it, and every question
    /// about culling water against a wet neighbor was being answered by
    /// the seal rather than by the water.
    WetLadder,
}

impl Block for B {
    type TransparentGroup = u8;

    const FLUID_ENABLED: bool = true;

    fn shape(&self) -> Shape {
        match self {
            B::Water(strength) => Shape::Fluid(FluidInfo {
                face: AlignedFace::PosY,
                height: HEIGHTS[(*strength as usize).min(3)],
                id: WATER,
            }),
            B::LowerSlab | B::WetLowerSlab => Shape::Slab(SlabInfo {
                face: AlignedFace::NegY,
                thickness: 8,
            }),
            B::WetStair => Shape::Stair(StairInfo {
                floor: AlignedFace::NegY,
                back: AlignedFace::NegZ,
            }),
            B::WetLadder => Shape::Facade(FacadeInfo {
                face: AlignedFace::NegZ,
                offset: 1,
            }),
            _ => Shape::WholeBlock,
        }
    }

    fn fluid(&self) -> Option<FluidInfo> {
        match self {
            B::Water(_) => match self.shape() {
                Shape::Fluid(info) => Some(info),
                _ => unreachable!(),
            },
            // Binary: a waterlogged cell is a source.
            B::WetLowerSlab | B::WetStair | B::WetLadder => Some(FluidInfo {
                face: AlignedFace::PosY,
                height: FULL_THICKNESS,
                id: WATER,
            }),
            _ => None,
        }
    }

    fn cull_mode(&self) -> CullMode<u8> {
        match self {
            B::Air => CullMode::Empty,
            B::Water(_) => CullMode::TransparentMerged(WATER),
            B::WetLadder => CullMode::TransparentMerged(LADDER),
            _ => CullMode::Opaque,
        }
    }
}

fn mesh(blocks: &[(u32, u32, u32, B)]) -> Quads {
    let mut chunk = PaddedChunk16::new_filled(B::Air);
    for &(x, y, z, block) in blocks {
        chunk.set(UVec3::new(x, y, z), block);
    }
    mesh_chunk(&chunk, true)
}

fn solid_total(q: &Quads) -> usize {
    q.faces.iter().map(|v| v.len()).sum()
}

fn fluid_total(q: &Quads) -> usize {
    q.fluid.iter().map(|v| v.len()).sum()
}

/// The solid quads belonging to the cell at `at`, across every face.
fn solid_quads_at(q: &Quads, at: UVec3) -> usize {
    AlignedFace::ALL
        .into_iter()
        .map(|face| {
            q.faces[face.index()]
                .iter()
                .filter(|quad| quad.voxel_position(face) == at)
                .count()
        })
        .sum()
}

#[test]
fn a_dry_solid_has_no_fluid_quads() {
    let q = mesh(&[(0, 0, 0, B::LowerSlab)]);
    assert_eq!(solid_total(&q), 6);
    assert_eq!(fluid_total(&q), 0);
}

#[test]
fn a_plain_fluid_stays_in_faces_not_in_fluid() {
    // The overlay is for fluids in *other* shapes' cells; a fluid cell's
    // own quads go where they always went.
    let q = mesh(&[(0, 0, 0, B::Water(0))]);
    assert_eq!(solid_total(&q), 6);
    assert_eq!(fluid_total(&q), 0);
}

#[test]
fn a_wet_slab_meshes_as_the_slab_and_the_water_it_stands_in() {
    let q = mesh(&[(0, 0, 0, B::WetLowerSlab)]);
    // The slab as ever.
    assert_eq!(solid_total(&q), 6);
    // The water: its surface and four sides. Its underside is the slab's
    // own floor, which is full, so there is nothing to draw there.
    assert_eq!(fluid_total(&q), 5);
    assert_eq!(q.fluid[AlignedFace::NegY.index()].len(), 0);
    assert_eq!(q.fluid[AlignedFace::PosY.index()].len(), 1);
    // A source: the surface is at the top of the cell.
    let top = &q.fluid[AlignedFace::PosY.index()][0];
    let shape = Shape::Fluid(B::WetLowerSlab.fluid().unwrap());
    for v in top.positions(AlignedFace::PosY, shape) {
        assert!((v.y - 1.0).abs() < 1e-6, "{v:?}");
    }
}

#[test]
fn a_wet_stair_hides_its_water_behind_its_floor_and_back() {
    let q = mesh(&[(0, 0, 0, B::WetStair)]);
    assert_eq!(solid_total(&q), 10);
    // Surface, front, and the two sides; not the floor or the back.
    assert_eq!(fluid_total(&q), 4);
    assert_eq!(q.fluid[AlignedFace::NegY.index()].len(), 0);
    assert_eq!(q.fluid[AlignedFace::NegZ.index()].len(), 0);
}

#[test]
fn water_beside_a_wet_slab_joins_it() {
    let q = mesh(&[(0, 0, 0, B::WetLowerSlab), (1, 0, 0, B::Water(0))]);
    // The slab's stone still shows through the water beside it: the
    // water does not seal a solid's face.
    assert_eq!(solid_total(&q), 6 + 5);
    // The two bodies of water share a face and neither draws it.
    assert_eq!(fluid_total(&q), 5 - 1);
    assert_eq!(q.fluid[AlignedFace::PosX.index()].len(), 0);
    let water_negx = q.faces[AlignedFace::NegX.index()]
        .iter()
        .filter(|quad| quad.voxel_position(AlignedFace::NegX) == UVec3::new(1, 0, 0))
        .count();
    assert_eq!(water_negx, 0, "the water's face against the wet cell");
}

/// The transparent twin of [`water_beside_a_wet_slab_joins_it`], and the
/// regression this pair exists to hold down.
///
/// A slab's stone is opaque, so the pond beside it was culled by the
/// "an opaque neighbor seals the face" rule and the real question was never
/// put. A ladder's is not. The pond's own quads go to
/// `is_culled_at_boundary`, which ends by asking whether the two *blocks*
/// merge — `TransparentMerged(WATER)` against `TransparentMerged(LADDER)`,
/// which they do not — so a pane of water was drawn down the middle of one
/// body of water, visible in game as a surface hanging beside every
/// waterlogged ladder.
#[test]
fn water_beside_a_wet_ladder_joins_it() {
    let q = mesh(&[(0, 0, 0, B::WetLadder), (1, 0, 0, B::Water(0))]);
    // Neither side draws the shared face. The wet cell's overlay...
    assert_eq!(
        q.fluid[AlignedFace::PosX.index()].len(),
        0,
        "the wet cell's water, facing the pond"
    );
    // ...nor the pond's own quad, which is the half that was wrong.
    let water_negx = q.faces[AlignedFace::NegX.index()]
        .iter()
        .filter(|quad| quad.voxel_position(AlignedFace::NegX) == UVec3::new(1, 0, 0))
        .count();
    assert_eq!(water_negx, 0, "the pond's face against the wet cell");
    // And the ladder still hangs there. Culling water against water must
    // not reach the block sharing the cell: water seals nothing of a
    // facade, so the rungs show through it exactly as they do in air.
    let alone = mesh(&[(0, 0, 0, B::WetLadder)]);
    assert_eq!(
        solid_quads_at(&q, UVec3::ZERO),
        solid_quads_at(&alone, UVec3::ZERO),
        "the ladder's own quads, with water beside it and without"
    );
}

/// A wet ladder's surface is shaded as water, not as the ladder.
///
/// The other half of one pond reading as two. With the faces between them
/// culled, what was left was the *shade*: the overlay was lit through the
/// `Facade` branch — the ladder's own cell, against the ladder's occluders —
/// so the wall it hangs on darkened the water above it, and the surface came
/// out a rectangle of slightly wrong blue outlined against the pond next door.
///
/// A ladder hangs on a wall, and a wall is exactly the occluder that makes the
/// two disagree, so the wall is what this builds. It sits *under* the
/// waterline, where it has no business shading anything on top of it.
#[test]
fn a_wet_ladders_surface_is_shaded_like_the_water_beside_it() {
    let q = mesh(&[
        // The wall, the ladder on its +Z face, and open water alongside.
        (1, 0, 0, B::Stone),
        (1, 0, 1, B::WetLadder),
        (1, 0, 2, B::Water(0)),
    ]);
    let wet = &q.fluid[AlignedFace::PosY.index()];
    assert_eq!(wet.len(), 1, "the wet cell's surface");
    let pond = q.faces[AlignedFace::PosY.index()]
        .iter()
        .find(|quad| quad.voxel_position(AlignedFace::PosY) == UVec3::new(1, 0, 2))
        .expect("the pond's surface");
    assert_eq!(
        wet[0].ao, pond.ao,
        "the wet cell's surface is shaded differently from the water it is part of"
    );
}

/// Two waterlogged ladders are one body of water, not two.
///
/// The same rule where *both* sides of the shared face are overlays, which
/// the overlay arm already handled — so this is the guard that hoisting the
/// same-fluid test out of that arm left it doing its old job.
#[test]
fn two_wet_ladders_join_their_water() {
    let q = mesh(&[(0, 0, 0, B::WetLadder), (1, 0, 0, B::WetLadder)]);
    assert_eq!(q.fluid[AlignedFace::PosX.index()].len(), 0);
    assert_eq!(q.fluid[AlignedFace::NegX.index()].len(), 0);
    // Merged into one surface over the pair, as two wet slabs are.
    assert_eq!(q.fluid[AlignedFace::PosY.index()].len(), 1);
}

#[test]
fn a_wet_cell_lifts_the_surface_of_flowing_water_beside_it() {
    let q = mesh(&[(0, 0, 0, B::WetLowerSlab), (1, 0, 0, B::Water(1))]);
    let shape = Shape::Fluid(B::Water(1).fluid().unwrap());
    let top = q.faces[AlignedFace::PosY.index()]
        .iter()
        .find(|quad| quad.voxel_position(AlignedFace::PosY) == UVec3::new(1, 0, 0))
        .expect("the flowing water's surface");
    for v in top.positions(AlignedFace::PosY, shape) {
        let want = if (v.x - 1.0).abs() < 1e-6 {
            // The edge shared with the full column.
            1.0
        } else {
            12.0 / 16.0
        };
        assert!((v.y - want).abs() < 1e-6, "{v:?} should be at {want}");
    }
}

#[test]
fn stone_over_a_wet_slab_hides_the_surface() {
    let q = mesh(&[(0, 0, 0, B::WetLowerSlab), (0, 1, 0, B::Stone)]);
    // The surface is gone; the stone's underside stays, since a lower slab
    // seals nothing of the top of its cell.
    assert_eq!(fluid_total(&q), 4);
    assert_eq!(q.fluid[AlignedFace::PosY.index()].len(), 0);
    assert_eq!(solid_total(&q), 6 + 6);
}

#[test]
fn stone_beside_a_wet_slab_hides_that_side_of_the_water() {
    let q = mesh(&[(0, 0, 0, B::WetLowerSlab), (1, 0, 0, B::Stone)]);
    assert_eq!(q.fluid[AlignedFace::PosX.index()].len(), 0);
    assert_eq!(fluid_total(&q), 4);
}

#[test]
fn two_wet_slabs_merge_their_surface() {
    let q = mesh(&[(0, 0, 0, B::WetLowerSlab), (1, 0, 0, B::WetLowerSlab)]);
    assert_eq!(q.fluid[AlignedFace::PosY.index()].len(), 1);
    assert_eq!(q.fluid[AlignedFace::PosX.index()].len(), 1);
    assert_eq!(q.fluid[AlignedFace::NegX.index()].len(), 1);
}

/// The `(min, max)` of a quad's vertices.
fn extent(quad: &Quad, face: AlignedFace, shape: Shape) -> (glam::Vec3, glam::Vec3) {
    quad.positions(face, shape).iter().fold(
        (
            glam::Vec3::splat(f32::INFINITY),
            glam::Vec3::splat(f32::NEG_INFINITY),
        ),
        |(lo, hi), v| (lo.min(*v), hi.max(*v)),
    )
}

#[test]
fn the_water_in_a_slab_is_clipped_to_the_open_half() {
    // Never a full-cell quad sharing the slab's own plane: the side faces
    // are the strip above the slab and nothing more.
    let q = mesh(&[(0, 0, 0, B::WetLowerSlab)]);
    let shape = Shape::Fluid(B::WetLowerSlab.fluid().unwrap());
    for face in [
        AlignedFace::PosX,
        AlignedFace::NegX,
        AlignedFace::PosZ,
        AlignedFace::NegZ,
    ] {
        let quads = &q.fluid[face.index()];
        assert_eq!(quads.len(), 1, "{face:?}");
        let (lo, hi) = extent(&quads[0], face, shape);
        assert!(
            (lo.y - 0.5).abs() < 1e-6,
            "{face:?} starts at the slab's top: {lo:?}"
        );
        assert!((hi.y - 1.0).abs() < 1e-6, "{face:?} {hi:?}");
    }
}

#[test]
fn the_water_in_a_stair_is_clipped_around_the_step() {
    let q = mesh(&[(0, 0, 0, B::WetStair)]);
    let shape = Shape::Fluid(B::WetStair.fluid().unwrap());
    // The surface shows only in front of the step (the step is on the -Z
    // half, flush with the back).
    let top = &q.fluid[AlignedFace::PosY.index()];
    assert_eq!(top.len(), 1);
    let (lo, hi) = extent(&top[0], AlignedFace::PosY, shape);
    assert!(
        (lo.z - 0.5).abs() < 1e-6 && (hi.z - 1.0).abs() < 1e-6,
        "{lo:?} {hi:?}"
    );
    // Each side shows the one open quadrant: above the slab, in front of
    // the step.
    for face in [AlignedFace::PosX, AlignedFace::NegX] {
        let quads = &q.fluid[face.index()];
        assert_eq!(quads.len(), 1, "{face:?}");
        let (lo, hi) = extent(&quads[0], face, shape);
        assert!(
            (lo.y - 0.5).abs() < 1e-6 && (hi.y - 1.0).abs() < 1e-6,
            "{face:?} {lo:?} {hi:?}"
        );
        assert!(
            (lo.z - 0.5).abs() < 1e-6 && (hi.z - 1.0).abs() < 1e-6,
            "{face:?} {lo:?} {hi:?}"
        );
    }
    // The front, opposite the back, is open above the slab.
    let front = &q.fluid[AlignedFace::PosZ.index()];
    assert_eq!(front.len(), 1);
    let (lo, hi) = extent(&front[0], AlignedFace::PosZ, shape);
    assert!(
        (lo.y - 0.5).abs() < 1e-6 && (hi.y - 1.0).abs() < 1e-6,
        "{lo:?} {hi:?}"
    );
}

#[test]
fn a_lone_block_has_no_overlay() {
    // `mesh_block` draws a held item; the fluid it stood in is not part of
    // the item.
    let q = mesh_block(&B::WetLowerSlab, ());
    assert_eq!(solid_total(&q), 6);
    assert_eq!(fluid_total(&q), 0);
}

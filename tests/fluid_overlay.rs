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
            B::WetLowerSlab | B::WetStair => Some(FluidInfo {
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

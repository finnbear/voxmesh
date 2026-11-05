//! Meshing real chunks: 100 taken from a game as its world loads, evenly
//! spaced over every chunk it meshed at full detail (remeshes included), so
//! they weigh each kind of terrain as a game does.
//!
//! `chunks.bin` holds them as the mesher sees them and no more: each distinct
//! block's answers to [`Block`] — shape, fluid, cull mode, AO opacity — and
//! each cell's block and light. They were extracted from voxmaxa (seed 1,
//! render distance 32) by its chunk benchmark's `--extract-voxmesh`.
//!
//! Every run first checks the output against [`EXPECTED`], a hash of every
//! quad, so an optimization can't change what is meshed unnoticed.

#![feature(test)]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

extern crate test;

use std::sync::OnceLock;
use test::Bencher;
use voxmesh::*;

/// The 100 chunks' [`checksum`]s, hashed in order. The same as voxmaxa's own
/// types mesh the chunks to, which is what shows the extraction kept
/// everything the mesher reads.
const EXPECTED: u64 = 0x751a8d1ec3728820;

/// One distinct block's answers to [`Block`].
struct Kind {
    shape: Shape,
    fluid: Option<FluidInfo>,
    cull_mode: CullMode<u16>,
    ao_opaque: bool,
}

/// Every [`Kind`] in `chunks.bin`, which a [`Voxel`] indexes.
static KINDS: OnceLock<Vec<Kind>> = OnceLock::new();

/// Two light channels of 0..=10 in a byte — block light high, sky light low —
/// with [`Self::OPAQUE`] for a solid cell, which casts none: the light the
/// chunks were extracted with. Averaging skips opaque cells, which is what
/// decides which neighboring faces light the same and so merge.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct DualLight(u8);

impl DualLight {
    const OPAQUE: u8 = u8::MAX;
    const MAX: f32 = 10.0;
}

impl Light for DualLight {
    type Average = [f32; 2];

    #[inline]
    fn average(values: &[Self]) -> [f32; 2] {
        let (mut count, mut block, mut sky) = (0u8, 0u8, 0u8);
        for value in values {
            if value.0 == Self::OPAQUE {
                continue;
            }
            block += value.0 >> 4;
            sky += value.0 & 0b1111;
            count += 1;
        }
        let scale = count as f32 * Self::MAX;
        [block as f32 / scale, sky as f32 / scale]
    }

    #[inline]
    fn blend(corners: &[[f32; 2]; 4], weights: [f32; 4]) -> [f32; 2] {
        let mut out = [0.0; 2];
        for (corner, weight) in corners.iter().zip(weights) {
            out[0] += corner[0] * weight;
            out[1] += corner[1] * weight;
        }
        out
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Voxel {
    kind: u16,
    light: DualLight,
}

impl Voxel {
    #[inline]
    fn kind(&self) -> &'static Kind {
        &KINDS.get().unwrap()[self.kind as usize]
    }
}

impl Block for Voxel {
    type TransparentGroup = u16;
    type Light = DualLight;
    const FLUID_ENABLED: bool = true;

    #[inline]
    fn shape(&self) -> Shape {
        self.kind().shape
    }

    #[inline]
    fn fluid(&self) -> Option<FluidInfo> {
        self.kind().fluid
    }

    #[inline]
    fn cull_mode(&self) -> CullMode<u16> {
        self.kind().cull_mode
    }

    #[inline]
    fn ao_opaque(&self) -> bool {
        self.kind().ao_opaque
    }

    #[inline]
    fn light(&self) -> DualLight {
        self.light
    }
}

/// Reads `chunks.bin` (see voxmaxa's `extract_voxmesh_chunks` for the
/// format), filling [`KINDS`].
fn chunks() -> Vec<Box<PaddedChunk16<Voxel>>> {
    let bytes: &[u8] = include_bytes!("chunks.bin");
    let mut at = 0;
    let mut take = |n: usize| {
        let taken = &bytes[at..at + n];
        at += n;
        taken
    };
    assert_eq!(take(16), b"VOXMESHBENCH0001");
    let u32_at = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap());
    let u16_at = |b: &[u8]| u16::from_le_bytes(b.try_into().unwrap());
    let face = |i: u8| {
        let face = AlignedFace::ALL[i as usize];
        assert_eq!(face.index(), i as usize);
        face
    };

    let kind_count = u32_at(take(4));
    let mut kinds = Vec::new();
    for _ in 0..kind_count {
        let [tag, a, b, c]: [u8; 4] = take(4).try_into().unwrap();
        let shape = match tag {
            0 => Shape::WholeBlock,
            1 => Shape::Slab(SlabInfo {
                face: face(a),
                thickness: b as Thickness,
            }),
            2 => Shape::Stair(StairInfo {
                floor: face(a),
                back: face(b),
            }),
            3 => Shape::Ramp(RampInfo {
                floor: face(a),
                back: face(b),
            }),
            4 => Shape::Cross(CrossInfo {
                face: face(a),
                stretch: b as u32,
            }),
            5 => Shape::Facade(FacadeInfo {
                face: face(a),
                offset: b,
            }),
            6 => Shape::Inset(a as Thickness),
            7 => Shape::Fluid(FluidInfo {
                face: face(a),
                height: b as Thickness,
                id: c,
            }),
            _ => panic!("unknown shape {tag}"),
        };
        let [has_fluid, a, b, c]: [u8; 4] = take(4).try_into().unwrap();
        let fluid = (has_fluid != 0).then(|| FluidInfo {
            face: face(a),
            height: b as Thickness,
            id: c,
        });
        let tag = take(1)[0];
        let group = u16_at(take(2));
        let cull_mode = match tag {
            0 => CullMode::Opaque,
            1 => CullMode::TransparentMerged(group),
            2 => CullMode::TransparentUnmerged,
            3 => CullMode::Empty,
            _ => panic!("unknown cull mode {tag}"),
        };
        let ao_opaque = take(1)[0] != 0;
        kinds.push(Kind {
            shape,
            fluid,
            cull_mode,
            ao_opaque,
        });
    }
    let _ = KINDS.set(kinds);

    let chunk_count = u32_at(take(4));
    let mut chunks = Vec::new();
    for _ in 0..chunk_count {
        let mut chunk = Box::new(PaddedChunk16::new_filled(Voxel {
            kind: 0,
            light: DualLight(0),
        }));
        let runs = u32_at(take(4));
        let mut cell = 0;
        for _ in 0..runs {
            let len = u16_at(take(2)) as usize;
            let kind = u16_at(take(2));
            let light = DualLight(take(1)[0]);
            chunk.data[cell..cell + len].fill(Voxel { kind, light });
            cell += len;
        }
        assert_eq!(cell, chunk.data.len());
        chunks.push(chunk);
    }
    assert_eq!(at, bytes.len());
    chunks
}

const FNV_OFFSET: u64 = 0xcbf29ce484222325;

/// FNV-1a over `bytes`, continuing from `hash`.
fn fnv(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |hash, &byte| {
        (hash ^ byte as u64).wrapping_mul(0x100000001b3)
    })
}

/// FNV-1a over everything in one chunk's `quads`.
fn checksum(quads: &Quads<DualLight>) -> u64 {
    let mut hash = FNV_OFFSET;
    let mut word = |word: u32| hash = fnv(hash, &word.to_le_bytes());
    for list in quads
        .faces
        .iter()
        .chain(&quads.diagonals)
        .chain(&quads.fluid)
    {
        word(list.len() as u32);
        for quad in list {
            for c in quad.origin_padded.to_array() {
                word(c);
            }
            word(quad.size.x);
            word(quad.size.y);
            word(u32::from_le_bytes(quad.corner_offsets.map(|o| o as u8)));
            word(u32::from_le_bytes(quad.ao));
            for [block, sky] in quad.light {
                word(block.to_bits());
                word(sky.to_bits());
            }
        }
    }
    hash
}

#[bench]
fn real_chunks(b: &mut Bencher) {
    let chunks = chunks();
    let mut quads = Quads::new();

    let mut hash = FNV_OFFSET;
    let mut total = 0;
    for chunk in &chunks {
        mesh_chunk_into(chunk.as_ref(), true, &mut quads);
        hash = fnv(hash, &checksum(&quads).to_le_bytes());
        total += quads.total();
    }
    assert_eq!(
        hash, EXPECTED,
        "output changed: {hash:#018x} ({total} quads), not {EXPECTED:#018x}"
    );

    b.iter(|| {
        for chunk in &chunks {
            mesh_chunk_into(test::black_box(chunk.as_ref()), true, &mut quads);
            test::black_box(&quads);
        }
    });
}

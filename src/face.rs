use glam::{IVec3, Vec3};

use crate::block::{CrossInfo, Shape};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    #[inline]
    pub fn index(self) -> usize {
        self as u8 as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Face {
    PosX,
    NegX,
    PosY,
    NegY,
    PosZ,
    NegZ,
}

impl Face {
    #[inline]
    pub fn index(self) -> usize {
        self as u8 as usize
    }

    #[inline]
    pub fn opposite(self) -> Face {
        match self {
            Face::PosX => Face::NegX,
            Face::NegX => Face::PosX,
            Face::PosY => Face::NegY,
            Face::NegY => Face::PosY,
            Face::PosZ => Face::NegZ,
            Face::NegZ => Face::PosZ,
        }
    }

    #[inline]
    pub fn axis(self) -> Axis {
        match self {
            Face::PosX | Face::NegX => Axis::X,
            Face::PosY | Face::NegY => Axis::Y,
            Face::PosZ | Face::NegZ => Axis::Z,
        }
    }

    #[inline]
    pub fn is_positive(self) -> bool {
        matches!(self, Face::PosX | Face::PosY | Face::PosZ)
    }

    #[inline]
    pub fn normal(self) -> IVec3 {
        match self {
            Face::PosX => IVec3::new(1, 0, 0),
            Face::NegX => IVec3::new(-1, 0, 0),
            Face::PosY => IVec3::new(0, 1, 0),
            Face::NegY => IVec3::new(0, -1, 0),
            Face::PosZ => IVec3::new(0, 0, 1),
            Face::NegZ => IVec3::new(0, 0, -1),
        }
    }

    /// Whether the tangent cross product `u×v` (from [`face_tangents`]) aligns
    /// with the outward normal. When `true` the natural vertex order
    /// `[base, base+du, base+du+dv, base+dv]` is already CCW; when `false`
    /// the offsets must be swapped.
    #[inline]
    pub fn tangent_cross_positive(self) -> bool {
        // u×v·normal > 0 for NegX, PosY, PosZ; < 0 for PosX, NegY, NegZ.
        self.is_positive() != (self.axis() == Axis::X)
    }

    pub const ALL: [Face; 6] = [
        Face::PosX,
        Face::NegX,
        Face::PosY,
        Face::NegY,
        Face::PosZ,
        Face::NegZ,
    ];
}

/// Any face that a quad can belong to: either an axis-aligned [`Face`] or a
/// [`DiagonalFace`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuadFace {
    Aligned(Face),
    Diagonal(DiagonalFace),
}

impl QuadFace {
    pub const ALL: [QuadFace; 8] = [
        QuadFace::Aligned(Face::PosX),
        QuadFace::Aligned(Face::NegX),
        QuadFace::Aligned(Face::PosY),
        QuadFace::Aligned(Face::NegY),
        QuadFace::Aligned(Face::PosZ),
        QuadFace::Aligned(Face::NegZ),
        QuadFace::Diagonal(DiagonalFace::A),
        QuadFace::Diagonal(DiagonalFace::B),
    ];

    /// Whether this face requires two-sided (backface-culling-disabled)
    /// rendering.
    #[inline]
    pub fn is_diagonal(self) -> bool {
        matches!(self, QuadFace::Diagonal(_))
    }

    /// Returns the normalized outward normal for this face.
    ///
    /// For axis-aligned faces this is the unit axis vector. For diagonal
    /// faces the normal depends on the root face axis (which determines the
    /// crossing plane). The sign is arbitrary since diagonal quads are
    /// two-sided.
    ///
    /// `shape` is only used for diagonal faces; for aligned faces it is
    /// ignored.
    #[inline]
    pub fn normal(self, shape: Shape) -> Vec3 {
        match self {
            QuadFace::Aligned(f) => f.normal().as_vec3(),
            QuadFace::Diagonal(d) => {
                let info = match shape {
                    Shape::Cross(info) => info,
                    _ => CrossInfo {
                        face: Face::NegY,
                        stretch: 0,
                    },
                };
                let dir = d.direction(); // (±s, 0, ±s) already normalized
                let (cross_a, cross_b) = match info.face.axis() {
                    Axis::X => (1, 2), // crossing in YZ
                    Axis::Y => (0, 2), // crossing in XZ
                    Axis::Z => (0, 1), // crossing in XY
                };
                let merge_axis = info.face.axis().index();
                // The normal is perpendicular to the diagonal in the crossing
                // plane — a 90° rotation of (dir.x, dir.z) mapped onto the
                // crossing axes. Already unit-length since dir is normalized.
                let mut n = [0.0f32; 3];
                n[cross_a] = -dir.z;
                n[cross_b] = dir.x;
                n[merge_axis] = 0.0;
                Vec3::from_array(n)
            }
        }
    }
}

impl From<Face> for QuadFace {
    #[inline]
    fn from(f: Face) -> Self {
        QuadFace::Aligned(f)
    }
}

impl From<DiagonalFace> for QuadFace {
    #[inline]
    fn from(d: DiagonalFace) -> Self {
        QuadFace::Diagonal(d)
    }
}

/// One of the two diagonal planes in an X-shaped billboard.
///
/// Viewed from above (looking down -Y), the two planes form an X:
/// - `A`: runs from the (minX, minZ) corner to (maxX, maxZ) — the `+X +Z` diagonal.
/// - `B`: runs from the (maxX, minZ) corner to (minX, maxZ) — the `-X +Z` diagonal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DiagonalFace {
    A,
    B,
}

impl DiagonalFace {
    pub const ALL: [DiagonalFace; 2] = [DiagonalFace::A, DiagonalFace::B];

    #[inline]
    pub fn index(self) -> usize {
        self as u8 as usize
    }

    /// Returns the normalized horizontal direction vector for this diagonal
    /// (in the XZ plane).
    #[inline]
    pub fn direction(self) -> Vec3 {
        let s = std::f32::consts::FRAC_1_SQRT_2;
        match self {
            DiagonalFace::A => Vec3::new(s, 0.0, s),
            DiagonalFace::B => Vec3::new(-s, 0.0, s),
        }
    }
}

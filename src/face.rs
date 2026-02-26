use glam::{IVec3, Vec3};

use crate::block::{CrossInfo, Shape};

/// One of the three spatial axes.
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

/// One of the six axis-aligned cube faces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AlignedFace {
    PosX,
    NegX,
    PosY,
    NegY,
    PosZ,
    NegZ,
}

impl AlignedFace {
    #[inline]
    pub fn index(self) -> usize {
        self as u8 as usize
    }

    #[inline]
    pub fn opposite(self) -> AlignedFace {
        match self {
            AlignedFace::PosX => AlignedFace::NegX,
            AlignedFace::NegX => AlignedFace::PosX,
            AlignedFace::PosY => AlignedFace::NegY,
            AlignedFace::NegY => AlignedFace::PosY,
            AlignedFace::PosZ => AlignedFace::NegZ,
            AlignedFace::NegZ => AlignedFace::PosZ,
        }
    }

    #[inline]
    pub fn axis(self) -> Axis {
        match self {
            AlignedFace::PosX | AlignedFace::NegX => Axis::X,
            AlignedFace::PosY | AlignedFace::NegY => Axis::Y,
            AlignedFace::PosZ | AlignedFace::NegZ => Axis::Z,
        }
    }

    #[inline]
    pub fn is_positive(self) -> bool {
        matches!(
            self,
            AlignedFace::PosX | AlignedFace::PosY | AlignedFace::PosZ
        )
    }

    #[inline]
    pub fn normal(self) -> IVec3 {
        match self {
            AlignedFace::PosX => IVec3::new(1, 0, 0),
            AlignedFace::NegX => IVec3::new(-1, 0, 0),
            AlignedFace::PosY => IVec3::new(0, 1, 0),
            AlignedFace::NegY => IVec3::new(0, -1, 0),
            AlignedFace::PosZ => IVec3::new(0, 0, 1),
            AlignedFace::NegZ => IVec3::new(0, 0, -1),
        }
    }

    /// Whether the tangent cross product `u x v`
    /// aligns with the outward normal. When true, the vertex order
    /// `[base, base+du, base+du+dv, base+dv]` is already CCW. When false,
    /// the offsets must be swapped.
    #[inline]
    pub fn tangent_cross_positive(self) -> bool {
        // (u x v) . normal > 0 for NegX, PosY, PosZ; < 0 for PosX, NegY, NegZ.
        self.is_positive() != (self.axis() == Axis::X)
    }

    pub const ALL: [AlignedFace; 6] = [
        AlignedFace::PosX,
        AlignedFace::NegX,
        AlignedFace::PosY,
        AlignedFace::NegY,
        AlignedFace::PosZ,
        AlignedFace::NegZ,
    ];
}

/// A face that a quad can belong to: either an axis-aligned
/// [`AlignedFace`] or a [`DiagonalFace`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Face {
    Aligned(AlignedFace),
    Diagonal(DiagonalFace),
}

impl Face {
    pub const ALL: [Face; 8] = [
        Face::Aligned(AlignedFace::PosX),
        Face::Aligned(AlignedFace::NegX),
        Face::Aligned(AlignedFace::PosY),
        Face::Aligned(AlignedFace::NegY),
        Face::Aligned(AlignedFace::PosZ),
        Face::Aligned(AlignedFace::NegZ),
        Face::Diagonal(DiagonalFace::A),
        Face::Diagonal(DiagonalFace::B),
    ];

    /// Whether this face requires two-sided rendering (no backface culling).
    #[inline]
    pub fn is_diagonal(self) -> bool {
        matches!(self, Face::Diagonal(_))
    }

    /// Returns the unit outward normal for this face.
    ///
    /// For axis-aligned faces, returns the unit axis vector. For diagonal
    /// faces, the normal depends on the root face axis (which determines
    /// the crossing plane). The sign is arbitrary since diagonal quads are
    /// two-sided.
    ///
    /// `shape` is only used for diagonal faces and ignored otherwise.
    #[inline]
    pub fn normal(self, shape: Shape) -> Vec3 {
        match self {
            Face::Aligned(f) => f.normal().as_vec3(),
            Face::Diagonal(d) => {
                let info = match shape {
                    Shape::Cross(info) => info,
                    _ => CrossInfo {
                        face: AlignedFace::NegY,
                        stretch: 0,
                    },
                };
                let dir = d.direction();
                let (cross_a, cross_b) = match info.face.axis() {
                    Axis::X => (1, 2),
                    Axis::Y => (0, 2),
                    Axis::Z => (0, 1),
                };
                let merge_axis = info.face.axis().index();
                // 90-degree rotation of the diagonal direction in the crossing
                // plane. Already unit-length since `dir` is normalized.
                let mut n = [0.0f32; 3];
                n[cross_a] = -dir.z;
                n[cross_b] = dir.x;
                n[merge_axis] = 0.0;
                Vec3::from_array(n)
            }
        }
    }
}

impl From<AlignedFace> for Face {
    #[inline]
    fn from(f: AlignedFace) -> Self {
        Face::Aligned(f)
    }
}

impl From<DiagonalFace> for Face {
    #[inline]
    fn from(d: DiagonalFace) -> Self {
        Face::Diagonal(d)
    }
}

/// One of the two diagonal planes in an X-shaped billboard.
///
/// Viewed from above (looking down -Y), the two planes form an X:
/// - `A`: runs from (minX, minZ) to (maxX, maxZ), the +X +Z diagonal.
/// - `B`: runs from (maxX, minZ) to (minX, maxZ), the -X +Z diagonal.
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

    /// Returns the unit horizontal direction vector for this diagonal
    /// in the XZ plane.
    #[inline]
    pub fn direction(self) -> Vec3 {
        let s = std::f32::consts::FRAC_1_SQRT_2;
        match self {
            DiagonalFace::A => Vec3::new(s, 0.0, s),
            DiagonalFace::B => Vec3::new(-s, 0.0, s),
        }
    }
}

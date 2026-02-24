use glam::IVec3;

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

    pub const ALL: [Face; 6] = [
        Face::PosX,
        Face::NegX,
        Face::PosY,
        Face::NegY,
        Face::PosZ,
        Face::NegZ,
    ];
}

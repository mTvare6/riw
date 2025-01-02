use crate::*;

#[derive(Clone)]
pub struct Interval{
    pub min:Float,
    pub max:Float,
}

impl Add<Float> for &Interval{
    type Output = Interval;
    fn add(self, rhs: Float) -> Self::Output {
        let min = self.min + rhs;
        let max = self.max + rhs;
        Self::Output{min, max}
    }
}

impl Interval{
    pub const NONE: Self = Self{min:INFINITY, max:NEG_INFINITY};
    pub const ALL: Self = Self{min:NEG_INFINITY, max:INFINITY};
    pub fn len(&self) -> Float{
        self.max - self.min
    }
    pub fn contains(&self, x:Float) -> bool{
        self.min <= x && x <= self.max
    }
    pub fn surrounds(&self, x:Float) -> bool{
        self.min < x && x < self.max
    }
    pub fn clamp(&self, x:Float) -> Float{
        x.clamp(self.min, self.max)
    }
    pub fn expanded(&self, d:Float) -> Interval{
        Interval{min:self.min-d, max:self.max+d}
    }
    pub fn ordered(x:Float, y:Float) -> Self{
        let (min, max) = if x>=y {(y, x)} else {(x, y)};
        Self{min, max}
    }
    pub fn enclosing(a: &Self, b:&Self) -> Self{
        let min = a.min.min(b.min);
        let max = a.max.max(b.max);
        Interval{min, max}
    }
}


pub struct Ray{
    pub orig:Point,
    pub dir:Point,
}

impl Ray{
    pub fn at(&self, t:Float) -> Point{
        self.orig + t*self.dir
    }
}


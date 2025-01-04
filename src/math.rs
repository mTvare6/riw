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
    pub const IN_SCENE : Self = Self{min: 0.001, max: INFINITY};
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
    pub dir:Vector,
}

impl Ray{
    pub fn at(&self, t:Float) -> Point{
        self.orig + t*self.dir
    }
    pub fn new(orig: Point, dir: Vector) -> Self{
        Self{orig, dir}
    }
}

pub struct ONB{
    pub u : Vector,
    pub v : Vector,
    pub w : Vector,
}
impl ONB{
    pub fn new(n: &Vector) -> Self{
        let w = n.normalize();
        let a = if w.x.abs() > 0.9 { Vector::Y } else { Vector::X };
        let v = w.cross(a).normalize();
        let u = w.cross(v);
        Self{u,v,w}
    }
    pub fn transform(&self, v: &Vector) -> Vector{
        self.u * v.x + self.v * v.y + self.w * v.z
    }
}

use crate::*;
#[derive(Clone)]
pub struct AABB{
    pub x: Interval,
    pub y: Interval,
    pub z: Interval
}

impl Index<usize> for AABB {
    type Output = Interval;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Index out of bounds: {}", index),
        }
    }
}

impl Add<&Vector> for &AABB{
    type Output = AABB;
    fn add(self, offset: &Vector) -> Self::Output{
        let x = &self.x + offset.x;
        let y = &self.y + offset.y;
        let z = &self.z + offset.z;
        Self::Output{x, y, z}
    }
} 

impl AABB{
    pub const NONE: Self = Self{
        x:Interval::NONE,
        y:Interval::NONE,
        z:Interval::NONE,
    };
    pub fn enclosing_point(a: &Vector, b: &Vector) -> Self{
        let x = Interval::ordered(a.x, b.x);
        let y = Interval::ordered(a.y, b.y);
        let z = Interval::ordered(a.z, b.z);
        Self{x, y, z}.padded()
    }
    pub fn enclosing_volume(a: &Self, b: &Self) -> Self{
        let x = Interval::enclosing(&a.x, &b.x);
        let y = Interval::enclosing(&a.y, &b.y);
        let z = Interval::enclosing(&a.z, &b.z);
        Self{x, y, z}.padded()
    }
    pub fn hit(&self, ray: &Ray, ray_t: &Interval) -> bool{
        for axis in 0..3{
            let interval = &self[axis];
            let dinv = ray.dir[axis].recip();
            let from = ray.orig[axis];

            let t0 = (interval.min - from) * dinv;
            let t1 = (interval.max - from) * dinv;

            let (t0, t1) = if t0<=t1 {(t0, t1)} else {(t1, t0)};
            let t_min = ray_t.min.max(t0);
            let t_max = ray_t.max.min(t1);
            if t_max <= t_min {return false;}
        }
        true
    }
    pub fn longest_axis(&self) -> usize {
        if self.x.len() > self.y.len() && self.x.len() > self.z.len() {
            0
        } else if self.y.len() > self.z.len() {
            1
        } else {
            2
        }
    }
    fn padded(self) -> Self{
        const DELTA: Float = 0.0001;
        let x = if self.x.len() < DELTA { self.x.expanded(DELTA) } else {self.x};
        let y = if self.y.len() < DELTA { self.y.expanded(DELTA) } else {self.y};
        let z = if self.z.len() < DELTA { self.z.expanded(DELTA) } else {self.z};
        Self{x, y, z}
    }
}

use crate::*;

pub trait PDF{
    fn pdf(&self, dir: &Vector) -> Float;
    fn icd(&self) -> Vector;
}

pub struct SpherePDF{}

impl SpherePDF{
    pub fn new() -> Self{
        Self{}
    }
}

impl PDF for SpherePDF{
    fn pdf(&self, dir: &Vector) -> Float{
        0.25 * FRAC_1_PI
    }
    fn icd(&self) -> Vector {
        Vector::random_unit_vector()
    }
}

pub struct CosinePDF{
    onb: ONB
}

impl PDF for CosinePDF{
    fn pdf(&self, dir: &Vector) -> Float {
        let cos = dir.dot(self.onb.w);
        0f64.max(cos*FRAC_1_PI)
    }
    fn icd(&self) -> Vector {
        self.onb.transform(&Vector::random_cosine_z())
    }
}

impl CosinePDF{
    pub fn new(dir: &Vector) -> Self{
        let onb = ONB::new(dir);
        Self{onb}
    }
}

pub struct HittablePDF<'a>{
    objects: &'a dyn Hittable,
    orig: Vector
}

impl<'a> PDF for HittablePDF<'a> {
    fn pdf(&self, dir: &Vector) -> Float {
        self.objects.pdf(&self.orig, dir)
    }
    fn icd(&self) -> Vector {
        self.objects.dir_to_random_point_to_hit(&self.orig)
    }
}

impl<'a> HittablePDF<'a>{
    pub fn new(objects: &'a dyn Hittable, orig: Vector)->Self{
        Self{objects,orig}
    }
}

pub struct MixedPDF<'a> {
    left: &'a dyn PDF,
    right: &'a dyn PDF,
    t: Float,
}

impl<'a> PDF for MixedPDF<'a> {
    fn pdf(&self, dir: &Vector) -> Float {
        self.t * self.right.pdf(dir) + (1.-self.t) * self.left.pdf(dir)
    }

    fn icd(&self) -> Vector {
        if rand() < self.t {self.right.icd()} else {self.left.icd()}
    }
}

impl<'a> MixedPDF<'a> {
    pub fn new(left: &'a dyn PDF, right: &'a dyn PDF, t: Float) -> Self {
        Self { left, right, t }
    }
}

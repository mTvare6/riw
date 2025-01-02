use crate::*;
#[derive(Clone)]
pub struct HitRecord{
    pub p: Point,
    pub n: Vector,
    pub t: Float,
    pub mat: Rc<dyn Material>,
    pub front: bool,
    pub uv: UV,
}

impl HitRecord{
    pub fn set_face_normal(&mut self, ray: &Ray, outward_normal: Vector) -> &Self{
        self.front = outward_normal.dot(ray.dir) < 0.;
        self.n =   if self.front {outward_normal} else {-outward_normal};
        self
    }
}

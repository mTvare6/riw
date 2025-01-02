use crate::*;
pub struct Translate{
    offset: Vector,
    object: Rc<dyn Hittable>,
    bbox: AABB
}

impl Translate{
    pub fn new(object: Rc<dyn Hittable>, offset: Vector) -> Self{
        let bbox = object.bounding_box() + &offset;
        Self{object, offset, bbox}
    }
}

impl Hittable for Translate{
    fn bounding_box(&self) -> &AABB {
        &self.bbox
    }
    fn hit(&self, ray: &Ray, t:&Interval) -> Option<HitRecord> {
        let offset_ray = Ray{orig:ray.orig-self.offset, dir:ray.dir};
        self.object.hit(&offset_ray, t).map(|r| HitRecord{p:r.p+self.offset, ..r})
    }
}


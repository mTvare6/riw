use crate::*;
impl Hittable for RotateY{
    fn bounding_box(&self) -> &AABB {
        &self.bbox
    }
    fn hit(&self, ray: &Ray, t:&Interval) -> Option<HitRecord> {
        let orig = Point::new(self.cos*ray.orig.x - self.sin*ray.orig.z, ray.orig.y, self.sin*ray.orig.x + self.cos*ray.orig.z);
        let dir = Point::new(self.cos*ray.dir.x - self.sin*ray.dir.z, ray.dir.y, self.sin*ray.dir.x + self.cos*ray.dir.z);
        let rotated_ray = Ray{orig, dir};
        self.object.hit(&rotated_ray, t).map(|record| {
            let p = Point::new(self.cos*record.p.x + self.sin*record.p.z, record.p.y, -self.sin*record.p.x + self.cos*record.p.z);
            let n = Point::new(self.cos*record.n.x + self.sin*record.n.z, record.n.y, -self.sin*record.n.x + self.cos*record.n.z);
            HitRecord{p, n, ..record}
        })
    }
}
impl RotateY{
    pub fn new(object: Rc<dyn Hittable>, t: Float) -> Self{
        let sin = t.sin();
        let cos = t.cos();
        let bbox = object.bounding_box();

        let (min, max) = (UV::splat(INFINITY), UV::splat(NEG_INFINITY));

        let corners = (0..2).flat_map(|i| (0..2).flat_map(move |j| (0..2).map(move |k| {
            let x = if i==0 {bbox.x.min} else {bbox.x.max};
            let y = if i==0 {bbox.y.min} else {bbox.y.max};
            let z = if i==0 {bbox.z.min} else {bbox.z.max};
            let xn = cos*x + sin*z;
            let zn = -sin*x + cos*z;
            UV::new(xn, zn)
        })));
        let (min, max) = corners.fold( (min, max), |(min, max), p| (min.min(p), max.max(p))  );
        let (min, max) = (Vector{y:bbox.y.min, x:min.x, z:min.y}, Vector{y:bbox.y.max, x:max.x, z:max.y});
        let bbox = AABB::enclosing_point(&min, &max);
        Self{sin, cos, bbox, object}
    }
}

pub struct RotateY{
    sin: Float,
    cos: Float,
    bbox: AABB,
    object: Rc<dyn Hittable>
}


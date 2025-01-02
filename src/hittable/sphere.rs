use crate::*;
pub struct Sphere{
    center: Point,
    radius: Float,
    mat: Rc<dyn Material>,
    bbox: AABB,
}

impl Hittable for Sphere{
    fn hit(&self, ray: &Ray, t:&Interval) -> Option<HitRecord>{
        let oc = self.center - ray.orig;
        let a = ray.dir.length_squared();
        let h = ray.dir.dot(oc);
        let c = oc.length_squared() - self.radius*self.radius;
        let discriminant = h*h - a*c;
        if discriminant < 0.0 {
            None
        } else{
            let sqrtd = discriminant.sqrt();
            let mut root = (h - sqrtd) / a;
            if  !t.surrounds(root){
                root = (h + sqrtd) / a;
                if  !t.surrounds(root){
                    return None;
                }
            }
            let p = ray.at(root);
            let outward_normal = (p - self.center) / self.radius;
            let mat = self.mat.clone();
            return Some(HitRecord{t:root, p, mat, n:Point::ZERO, front:false, uv:Sphere::get_sphere_uv(&outward_normal)}.set_face_normal(ray, outward_normal).clone());
        }

    }
    fn bounding_box(&self) -> &AABB {
        &self.bbox
    }

}
impl Sphere{
    pub fn new(center: Point, radius: Float, mat: Rc<dyn Material>) -> Self{
        let r = Vector::ONE*radius;
        let bbox = AABB::enclosing_point(&(center - r), &(center + r));
        Sphere{center, radius, mat, bbox}
    }
    fn get_sphere_uv(p: &Point) -> UV{
        let t = (-p.y).acos();
        let f = (-p.z).atan2(p.x) + PI;
        let u = f/(2.*PI);
        let v = t/PI;
        UV{x:u, y:v}
    }
}




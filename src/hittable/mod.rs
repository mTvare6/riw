use crate::*;
pub mod sphere;
pub mod quad;
pub mod hittable_list;
pub mod rotate;
pub mod translate;
pub mod constant_medium;

pub use sphere::*;
pub use quad::*;
pub use hittable_list::*;
pub use rotate::*;
pub use translate::*;
pub use constant_medium::*;


pub trait Hittable{
    fn hit(&self, ray: &Ray, t:&Interval) -> Option<HitRecord>;
    fn bounding_box(&self) -> &AABB;
    fn pdf(&self, orig: &Point, dir: &Vector) -> Float{
        0.0
    }
    fn dir_to_random_point_to_hit(&self, orig: &Point) -> Vector{
        Point::X
    }
}


use crate::*;
pub struct HittableList{
    pub objects: Vec<Rc<dyn Hittable>>,
    pub bbox: AABB,
}

impl HittableList{
    pub fn new() -> Self{
        HittableList{objects:Vec::new(), bbox:AABB::NONE}
    }
    pub fn add(&mut self, object: Rc<dyn Hittable>) {
        self.bbox = AABB::enclosing_volume(&self.bbox, object.bounding_box());
        self.objects.push(object);
    }
}

impl Hittable for HittableList{
    fn hit(&self, ray: &Ray, t:&Interval) -> Option<HitRecord>{
        let mut t_least = t.max;
        // &Vec<T> borrws whereas
        // Vec<T> consumes and should fall, but since stays, invalid code
        /*
            let t = vec![1, 2, 3];
            for h in t{
                println!("{}", h);
            }

            for h in t{
                println!("{}", h);
            }
        */
        /*
        * the syntactic sugar boils down to using IntoIterator implementation and IntoIterator for
        * Vec<T> is fundamentally different from that for &Vec<T> where each element is &T
        * instead of T.
        * So in Vec<T>, each T is dropped after {} whereas in &Vec<T> the reference is dropped
        *
        * If it weren't dropped, say A has a Box<T> and looping through A we push the elements to
        * B, here when finally going out of scope, double free can occur.
        */

        // Only closer objects are permitted after each iteration
        self.objects.iter()
            .filter_map(|object| {
                object.hit(ray, &mut Interval { min: t.min, max: t_least }).map(
                    |record_tmp| {
                        t_least = record_tmp.t;
                        record_tmp
                    }
                )}).last()
    }
    fn bounding_box(&self) -> &AABB {
        &self.bbox
    }

}

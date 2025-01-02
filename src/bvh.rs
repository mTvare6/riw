use crate::*;
pub struct BVHNode{
    left: Rc<dyn Hittable>,
    right: Rc<dyn Hittable>,
    bbox: AABB
}

impl Hittable for BVHNode{
    fn hit(&self, ray: &Ray, t:&Interval) -> Option<HitRecord>{
        if !self.bbox.hit(ray, t){
            None
        } else {
            self.left.hit(ray, t).map_or_else(
                || self.right.hit(ray, t),
                |record| {
                    self.right.hit(ray, &Interval {
                        min: t.min,
                        max: record.t,
                    }).or(Some(record))
                },
            )
        }
    }
    fn bounding_box(&self) -> &AABB {
        &self.bbox
    }

}

impl BVHNode{
    pub fn from_hittable_list(list: &mut HittableList) -> Self{
        let size = list.objects.len();
        Self::from_vec(&mut list.objects, 0, size)
    }
    fn from_vec(objects: &mut Vec<Rc<dyn Hittable>>, start: usize, end:usize) -> Self{
        let mut bbox = AABB::NONE;
        for i in start..end{
            bbox = AABB::enclosing_volume(&bbox, objects[i].bounding_box());
        }
        let axis = bbox.longest_axis();
        let compatator = |a: &Rc<dyn Hittable>, b: &Rc<dyn Hittable>| {
            a.bounding_box()[axis].min.partial_cmp(&b.bounding_box()[axis].min).unwrap()
        };

        let span = end - start;

        let (left, right) = match span {
            1 => {
                (objects[start].clone(), objects[start].clone())
            },
            2 => {
                (objects[start].clone(), objects[start+1].clone())
            },
            _ => {
                objects[start..end].sort_by(compatator);
                let mid = start + span/2;
                let left = Rc::new(BVHNode::from_vec(objects, start, mid));
                let right = Rc::new(BVHNode::from_vec(objects, mid, end));
                (left as Rc<dyn Hittable>, right as Rc<dyn Hittable>)
            }
        };
        let bbox = AABB::enclosing_volume(left.bounding_box(), right.bounding_box());
        Self{left, right, bbox}
    }
}



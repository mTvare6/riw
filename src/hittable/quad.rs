use crate::*;
pub fn make_bounding_cube(a: Point, b: Point, mat: Rc<dyn Material>) -> Rc<HittableList> {
    let mut sides:HittableList  = HittableList::new();
    let min = Point::new(
        a.x.min(b.x),
        a.y.min(b.y),
        a.z.min(b.z)
    );
    let max = Point::new(
        a.x.max(b.x),
        a.y.max(b.y),
        a.z.max(b.z)
    );

    let dx = Vector::new(max.x - min.x, 0.0, 0.0);
    let dy = Vector::new(0.0, max.y - min.y, 0.0);
    let dz = Vector::new(0.0, 0.0, max.z - min.z);

    sides.add(Rc::new(Quad::new(Point::new(min.x, min.y, max.z), dx, dy, mat.clone()))); // front
    sides.add(Rc::new(Quad::new(Point::new(max.x, min.y, max.z), -dz, dy, mat.clone()))); // right
    sides.add(Rc::new(Quad::new(Point::new(max.x, min.y, min.z), -dx, dy, mat.clone()))); // back
    sides.add(Rc::new(Quad::new(Point::new(min.x, min.y, min.z), dz, dy, mat.clone()))); // left
    sides.add(Rc::new(Quad::new(Point::new(min.x, max.y, max.z), dx, -dz, mat.clone()))); // top
    sides.add(Rc::new(Quad::new(Point::new(min.x, min.y, min.z), dx, dz, mat)));  // bottom
    Rc::new(sides)
}


#[derive(Clone)]
pub struct Quad{
    q: Point,
    u: Point,
    v: Point,
    n: Vector,
    w: Vector,
    d: Float,
    area: Float,
    mat: Rc<dyn Material>,
    bbox: AABB,
}

impl Quad{
    pub fn new(q: Vector, u:Vector, v:Vector, mat:Rc<dyn Material>)->Self{
        let a = AABB::enclosing_point(&q, &(q+u+v));
        let b = AABB::enclosing_point(&(q+u), &(q+v));
        let bbox = AABB::enclosing_volume(&a, &b);
        let normal = u.cross(v);
        let w = normal/normal.length_squared();
        let n = normal.normalize();
        let d = n.dot(q);
        let area = normal.length();
        Self{q, u, v, mat, bbox, n, d, w, area}
    }
    fn is_interior(a: Float, b: Float) -> Option<UV> {
        const UNIT: Interval = Interval{min:0.0, max:1.0};
        if !UNIT.contains(a) || !UNIT.contains(b) {
            None
        } else {
            Some(UV{x:a,y:b})
        }
    }
}

impl Hittable for Quad{
    fn hit(&self, ray: &Ray, t:&Interval) -> Option<HitRecord> {
        let denom = ray.dir.dot(self.n);
        if denom.abs() < 1e-8 {
            None
        } else {
            let root = (self.d - self.n.dot(ray.orig))/denom;
            if !t.contains(root){
                None
            } else {
                let p = ray.at(root);
                let planar_p = p - self.q;
                let alpha = self.w.dot(planar_p.cross(self.v));
                let beta = self.w.dot(self.u.cross(planar_p));
                let mat = self.mat.clone();
                Quad::is_interior(alpha, beta).map(|uv| HitRecord{t:root, p, mat, n:Point::ZERO, front:false, uv}.set_face_normal(ray, self.n).clone())
            }
        }
    }
    fn bounding_box(&self) -> &AABB {
        &self.bbox
    }
    fn pdf(&self, orig: &Point, dir: &Vector) -> Float {
        self.hit(&Ray{orig:*orig, dir:*dir}, &Interval::IN_SCENE).map_or(0.0, |record| {
            let dist_square = record.t * record.t * dir.length_squared();
            let cos = dir.dot(record.n).abs() / dir.length();
            dist_square / (cos*self.area)
        })
    }
    fn dir_to_random_point_to_hit(&self, orig: &Point) -> Vector {
        let p = self.q + (self.u*rand()) + (self.v*rand());
        p-orig
    }
}


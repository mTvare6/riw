#![allow(unused_variables)]
#![allow(dead_code)]
use glam::{DVec3,DVec2};
use std::rc::Rc;
use std::f64::{INFINITY,NEG_INFINITY};
use std::f64::consts::PI;
use rand::prelude::*;
use rand::distributions::Uniform;
use std::cell::RefCell;
use std::sync::Arc;
use std::ops::{Index, Add};
use std::array::from_fn;
use rayon::prelude::*;

struct UnsafeSync<T>(T);
unsafe impl<T> Sync for UnsafeSync<T> {}
unsafe impl<T> Send for UnsafeSync<T> {}

type Color = DVec3;
type Point = DVec3;
type Vector = DVec3;
type UV = DVec2;
type Float = f64;

const BLUE: Color = Color{x:0.7, y:0.8, z:1.0};

#[repr(C)]
pub struct ConfigValue {
    marker: [u8; 8],
    value: f64,
}

#[link_section = ".data"]
#[no_mangle]
static mut CONFIGS: [ConfigValue; 5] = [
    ConfigValue { marker: *b"COEF_F1_", value: 400.0 }, // width
    ConfigValue { marker: *b"COEF_F2_", value: 1. }, // aspect ratio
    ConfigValue { marker: *b"COEF_F3_", value: 100.0 }, // pixel density
    ConfigValue { marker: *b"COEF_F4_", value: 50.0 }, // max bounce
    ConfigValue { marker: *b"COEF_F5_", value: 4.0 }, // 
];

thread_local! {
    static PIXEL_RNG: RefCell<(ThreadRng, Uniform<Float>)> = RefCell::new((
        thread_rng(),
        Uniform::new(-0.5, 0.5)
    ));
}

thread_local! {
    static SQUARE_RNG: RefCell<(ThreadRng, Uniform<Float>)> = RefCell::new((
        thread_rng(),
        Uniform::new(-1.0, 1.0)
    ));
}

fn linear_to_gamma(linear: Float) -> Float{
    if linear>0.0 {linear.sqrt()} else {0.0}
}

fn rand() -> Float{
    SQUARE_RNG.with(|rng| {
        let (rng, distribution) = &mut *rng.borrow_mut();
        distribution.sample(rng).abs()
    })
}

trait Ext {
    fn write(&self);
}
impl Ext for Color {
    fn write(&self) {
        const COLORSPACE:Interval = Interval{min:0.000, max:0.999};
        let ir = (256. * COLORSPACE.clamp(linear_to_gamma(self.x))) as u8;
        let ig = (256. * COLORSPACE.clamp(linear_to_gamma(self.y))) as u8;
        let ib = (256. * COLORSPACE.clamp(linear_to_gamma(self.z))) as u8;
        println!("{ir} {ig} {ib}")
    }
}

trait Utils {
    fn random_unit_vector()->Self;
    fn random_on_pixel()->Self;
    fn random_on_disk()->Self;
    fn random_on_hemisphere(normal: &Vector)->Self;
    fn random()->Self;
    fn near_zero(&self) -> bool;
    fn reflect(&self, n: &Self) -> Self;
}

impl Utils for Vector {
    fn random_unit_vector()->Self{
        loop{
            let p = Self::random();
            let lsq = p.length_squared();
            if 1e-160 < lsq && lsq <= 1.0 {
                return p/lsq.sqrt();
            }
        }
    }
    fn random_on_hemisphere(normal: &Vector)->Self {
        let uv = Self::random_unit_vector();
        if normal.dot(uv) > 0.0 {uv} else {-uv}
    }
    fn random_on_pixel()->Point{
            PIXEL_RNG.with(|rng| {
                let (rng, distribution) = &mut *rng.borrow_mut();
                Point {
                    x: distribution.sample(rng),
                    y: distribution.sample(rng),
                    z: 0.0
                }
            })
    }
    fn random_on_disk()->Point{
        loop{
            let p = SQUARE_RNG.with(|rng| {
                let (rng, distribution) = &mut *rng.borrow_mut();
                Point {
                    x: distribution.sample(rng),
                    y: distribution.sample(rng),
                    z: 0.0
                }
            });
            if p.length_squared() < 1.0{
                return p;
            }
        }
    }
    fn random()->Self{
            SQUARE_RNG.with(|rng| {
                let (rng, distribution) = &mut *rng.borrow_mut();
                Self {
                    x: distribution.sample(rng),
                    y: distribution.sample(rng),
                    z: distribution.sample(rng),
                }
            })
    }
    fn near_zero(&self) -> bool{
        let t = 1e-8;
        (self.x.abs() < t) &&
        (self.y.abs() < t) &&
        (self.z.abs() < t)
    }

    fn reflect(&self, n: &Self) -> Self{
        self - 2.0*self.dot(*n)*n
    }
}

struct Ray{
    orig:Point,
    dir:Point,
}

impl Ray{
    fn at(&self, t:Float) -> Point{
        self.orig + t*self.dir
    }
}

#[derive(Clone)]
struct HitRecord{
    p: Point,
    n: Vector,
    mat: Rc<dyn Material>,
    t: Float,
    front: bool,
    uv: UV,
}

impl HitRecord{
    fn set_face_normal(&mut self, ray: &Ray, outward_normal: Vector) -> &Self{
        self.front = outward_normal.dot(ray.dir) < 0.;
        self.n =   if self.front {outward_normal} else {-outward_normal};
        self
    }
}

trait Hittable{
    fn hit(&self, ray: &Ray, t:&Interval) -> Option<HitRecord>;
    fn bounding_box(&self) -> &AABB;
}

struct Sphere{
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

struct Translate{
    offset: Vector,
    object: Rc<dyn Hittable>,
    bbox: AABB
}

impl Translate{
    fn new(object: Rc<dyn Hittable>, offset: Vector) -> Self{
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
    fn new(object: Rc<dyn Hittable>, t: Float) -> Self{
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

struct RotateY{
    sin: Float,
    cos: Float,
    bbox: AABB,
    object: Rc<dyn Hittable>
}


struct Quad{
    q: Point,
    u: Point,
    v: Point,
    n: Vector,
    w: Vector,
    d: Float,
    mat: Rc<dyn Material>,
    bbox: AABB,
}

impl Quad{
    fn new(q: Vector, u:Vector, v:Vector, mat:Rc<dyn Material>)->Self{
        let a = AABB::enclosing_point(&q, &(q+u+v));
        let b = AABB::enclosing_point(&(q+u), &(q+v));
        let bbox = AABB::enclosing_volume(&a, &b);
        let normal = u.cross(v);
        let w = normal/normal.length_squared();
        let n = normal.normalize();
        let d = n.dot(q);
        Self{q, u, v, mat, bbox, n, d, w}
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
}

struct ConstantMedium{
    boundary: Rc<dyn Hittable>,
    neg_inv_density: Float,
    phase_function: Rc<dyn Material>
}

impl ConstantMedium{
    fn from_color(boundary: Rc<dyn Hittable>, density: Float, albedo: Color) -> Self{
        let neg_inv_density = -density.recip();
        let phase_function = Rc::new(Isotropic::from_color(albedo));
        Self{
            boundary,
            neg_inv_density,
            phase_function,
        }
    }
    fn from_texture(boundary: Rc<dyn Hittable>, density: Float, tex: Rc<dyn Texture>) -> Self{
        let neg_inv_density = -density.recip();
        let phase_function = Rc::new(Isotropic::from_texture(tex));
        Self{
            boundary,
            neg_inv_density,
            phase_function,
        }
    }
}

impl Hittable for ConstantMedium{
    fn hit(&self, ray: &Ray, t:&Interval) -> Option<HitRecord> {
        self.boundary.hit(ray, &Interval::ALL).and_then(|record_all| {
            self.boundary.hit(ray, &Interval{min:record_all.t+0.0001, max:INFINITY}).and_then(|record_box|{
                let t_prev = t.min.max(record_all.t);
                let t_next = t.max.min(record_box.t);
                if t_prev >= t_next {
                    None
                } else {
                    let t_prev = t_prev.max(0.0);
                    let ray_length = ray.dir.length();
                    let distance_inside_boundary = (t_next-t_prev)*ray_length;
                    let hit_distance = self.neg_inv_density * rand().ln();
                    if hit_distance > distance_inside_boundary{
                        None
                    } else {
                        let t = t_prev + hit_distance/ray_length;
                        let p = ray.at(t);
                        let n = Vector::X;
                        let front = true;
                        let mat = self.phase_function.clone();
                        Some(HitRecord{p, n, mat, t, front, uv:UV::ZERO})
                    }
                }
            })
        })
    }
    fn bounding_box(&self) -> &AABB {
        &self.boundary.bounding_box()
    }
}

struct HittableList{
    objects: Vec<Rc<dyn Hittable>>,
    bbox: AABB,
}

impl HittableList{
    fn new() -> Self{
        HittableList{objects:Vec::new(), bbox:AABB::NONE}
    }
    fn add(&mut self, object: Rc<dyn Hittable>) {
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

impl Sphere{
    fn new(center: Point, radius: Float, mat: Rc<dyn Material>) -> Self{
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

#[derive(Clone)]
struct Interval{
    min:Float,
    max:Float,
}

impl Add<Float> for &Interval{
    type Output = Interval;
    fn add(self, rhs: Float) -> Self::Output {
        let min = self.min + rhs;
        let max = self.max + rhs;
        Self::Output{min, max}
    }
}

impl Interval{
    const NONE: Self = Self{min:INFINITY, max:NEG_INFINITY};
    const ALL: Self = Self{min:NEG_INFINITY, max:INFINITY};
    fn len(&self) -> Float{
        self.max - self.min
    }
    fn contains(&self, x:Float) -> bool{
        self.min <= x && x <= self.max
    }
    fn surrounds(&self, x:Float) -> bool{
        self.min < x && x < self.max
    }
    fn clamp(&self, x:Float) -> Float{
        x.clamp(self.min, self.max)
    }
    fn expanded(&self, d:Float) -> Interval{
        Interval{min:self.min-d, max:self.max+d}
    }
    fn ordered(x:Float, y:Float) -> Self{
        let (min, max) = if x>=y {(y, x)} else {(x, y)};
        Self{min, max}
    }
    fn enclosing(a: &Self, b:&Self) -> Self{
        let min = a.min.min(b.min);
        let max = a.max.max(b.max);
        Interval{min, max}
    }
}

#[derive(Clone)]
struct AABB{
    x: Interval,
    y: Interval,
    z: Interval
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
    const NONE: Self = Self{
        x:Interval::NONE,
        y:Interval::NONE,
        z:Interval::NONE,
    };
    fn enclosing_point(a: &Vector, b: &Vector) -> Self{
        let x = Interval::ordered(a.x, b.x);
        let y = Interval::ordered(a.y, b.y);
        let z = Interval::ordered(a.z, b.z);
        Self{x, y, z}.padded()
    }
    fn enclosing_volume(a: &Self, b: &Self) -> Self{
        let x = Interval::enclosing(&a.x, &b.x);
        let y = Interval::enclosing(&a.y, &b.y);
        let z = Interval::enclosing(&a.z, &b.z);
        Self{x, y, z}.padded()
    }
    fn hit(&self, ray: &Ray, ray_t: &Interval) -> bool{
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
    fn longest_axis(&self) -> usize {
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

struct BVHNode{
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
    fn from_hittable_list(list: &mut HittableList) -> Self{
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


struct Camera{
    background: Color,
    aspect_ratio: Float,
    image_width: usize,
    image_height: usize,
    pixel_density: usize,
    max_depth: usize,
    inverse_density: Float,
    vfov: Float,
    center: Point,
    pixel_corner: Point,
    pixel_delta_u: Vector,
    pixel_delta_v: Vector,
    defocus_disk_u: Vector,
    defocus_disk_v: Vector,
    defocus_angle: Float,
    u:Vector,
    v:Vector,
    w:Vector,
}

impl Camera{
    fn render(&self, world:&dyn Hittable) {
        println!("P3\n{} {}\n255\n", self.image_width, self.image_height);
        let mut data = vec![Color::ZERO; self.image_height*self.image_width];
        let unsafe_world = Arc::new(UnsafeSync(world));
        data.par_iter_mut().enumerate().for_each(|(x, c)| {
            let world = unsafe_world.0;
            let j = x / self.image_width;
            let i = x % self.image_width;
            *c = (0..self.pixel_density)
                .map(|_| self.color(&self.get_ray(i, j), world, self.max_depth))
                .sum::<Color>()*self.inverse_density;

        });
        for c in data{
            c.write();
        }
    }
    fn color(&self, ray: &Ray, world:&dyn Hittable, depth: usize) -> Color{
        if depth==0{
            return Color::ZERO;
        }
        match world.hit(ray, &mut Interval{min:0.001, max:INFINITY}){
            Some(record) => {
                // Option<T>::as_ref() -> Option<&T>
                match  record.mat.as_ref().scatter(ray, &record){
                    Some((attenuation, scattered)) =>
                    attenuation * self.color(&scattered, world, depth-1),
                    None => record.mat.as_ref().emitted(record.uv, &record.p)
                }
            },
            None => self.background
        }
    }
    fn get_ray(&self, i:usize, j:usize) -> Ray{
        let offset = Vector::random_on_pixel();
        let pixel_sample = self.pixel_corner + ((i as Float + offset.x) * self.pixel_delta_u) + ((j as Float + offset.y) * self.pixel_delta_v);
        let ray_origin = if self.defocus_angle > 0. {self.defocus_disk_sample()} else {self.center};
        let ray_direction = pixel_sample - ray_origin;
        Ray{orig:ray_origin, dir:ray_direction}
    }
    fn defocus_disk_sample(&self) -> Point{
        let p = Point::random_on_disk();
        self.center + p.x * self.defocus_disk_u + p.y * self.defocus_disk_v
    }
    fn new(lookfrom: Point, lookat: Point, vup:Vector, background:Color, vfov:Float, defocus_angle:Float, aspect_ratio:Float) -> Self {
        let image_width = unsafe { CONFIGS[0].value } as usize;
        let image_height = ((image_width as Float)/aspect_ratio) as usize;
        let pixel_density = unsafe { CONFIGS[2].value } as usize;
        let inverse_density = (pixel_density as Float).recip();
        let max_depth = unsafe { CONFIGS[3].value } as usize;
        //let focus_dist = 10.0;
        let focus_dist = (lookfrom-lookat).length();
        let w = (lookfrom-lookat).normalize();
        let u = vup.cross(w).normalize();
        let v = w.cross(u);
        let h = (vfov/2.).to_radians().tan();
        let viewport_h = 2.0*h*focus_dist;
        let viewport_w = viewport_h*(image_width as Float/image_height as Float);
        let center = lookfrom;
        let viewport_u = viewport_w * u;
        let viewport_v = viewport_h * -v;
        let pixel_delta_u = viewport_u/image_width as Float;
        let pixel_delta_v = viewport_v/image_height as Float;
        let viewport_upper_left = center - (focus_dist*w) - 0.5*(viewport_u+viewport_v);
        let pixel_corner = viewport_upper_left + 0.5*(pixel_delta_v+pixel_delta_u);

        let defocus_radius = focus_dist * (defocus_angle/2.).to_radians().tan();
        let defocus_disk_u = defocus_radius * u;
        let defocus_disk_v = defocus_radius * v;
        Camera{
            background,
            aspect_ratio,
            image_width,
            image_height,
            pixel_density,
            inverse_density,
            max_depth,
            vfov,
            center,
            pixel_corner,
            pixel_delta_u,
            pixel_delta_v,
            defocus_disk_u,
            defocus_disk_v,
            defocus_angle,
            u,
            v,
            w
        }
    }
}

struct Lambertian{
    tex: Rc<dyn Texture>
}

impl Lambertian{
    fn from_color(albedo:Color) -> Self{
        let tex = Rc::new(SolidColor{albedo});
        Self{tex}
    }
}

struct Metal{
    albedo: Color,
    fuzz: Float,
}

struct Dielectric{
    refractive_index: Float
}

struct DiffuseLight{
    tex: Rc<dyn Texture>
}

impl DiffuseLight{
    fn from_color(albedo:Color) -> Self{
        let tex = Rc::new(SolidColor{albedo});
        Self{tex}
    }
}

struct Isotropic{
    tex: Rc<dyn Texture>
}

trait Material{
    fn scatter(&self, ray: &Ray, record:&HitRecord) -> Option<(Color, Ray)>;
    fn emitted(&self, u: UV, p: &Point) -> Color{
        Color::ZERO
    }
}

impl Material for Lambertian{
    fn scatter(&self, ray: &Ray, record:&HitRecord) -> Option<(Color, Ray)>{
        let scatter_direction = {
            let dir = record.n + Vector::random_unit_vector();
            if dir.near_zero() { record.n } else { dir }
        };
        let scattered = Ray{orig:record.p, dir:scatter_direction};
        let attenuation = self.tex.texture(record.uv, &record.p);
        Some((attenuation, scattered))
    }
}

impl Material for Metal{
    fn scatter(&self, ray: &Ray, record:&HitRecord) -> Option<(Color, Ray)>{
        let reflected = {
            let dir:Vector = ray.dir.reflect(record.n);
            dir.normalize() + (self.fuzz * Vector::random_unit_vector())
        };
        let scattered = Ray{orig:record.p, dir:reflected};
        let attenuation = self.albedo;
        Some((attenuation, scattered))
    }
}

impl Material for Dielectric{
    fn scatter(&self, ray: &Ray, record:&HitRecord) -> Option<(Color, Ray)>{
        let ri = if record.front {self.refractive_index.recip()} else {self.refractive_index};
        let unit = ray.dir.normalize();
        let cos = record.n.dot(-unit).min(1.0);
        let sin = (1.0 - cos*cos).sqrt();
        let dir = if ri * sin > 1.0 || Dielectric::reflectance(cos, ri) > rand(){
            unit.reflect(record.n)
        } else{
            unit.refract(record.n, ri)
        };
        let scattered = Ray{orig:record.p, dir};
        let attenuation = Color::ONE;
        Some((attenuation, scattered))
    }
}

impl Material for DiffuseLight{
    fn scatter(&self, ray: &Ray, record:&HitRecord) -> Option<(Color, Ray)> {
        None
    }
    fn emitted(&self, uv: UV, p: &Point) -> Color {
        self.tex.texture(uv, p)
    }
}

impl Material for Isotropic{
    fn scatter(&self, ray: &Ray, record:&HitRecord) -> Option<(Color, Ray)> {
        let scattered = Ray{orig:record.p, dir:Vector::random_unit_vector()};
        let attenuation = self.tex.texture(record.uv, &record.p);
        Some((attenuation, scattered))
    }
}

impl Dielectric{
    fn reflectance(cos: Float, ri: Float) -> Float{
        let r0 = {
            let rt = (1.-ri)/(1.+ri);
            rt*rt
        };
        r0 * (1.0-r0)*(1.0 - cos).powi(5)
    }
}

impl Isotropic{
    fn from_color(albedo: Color) -> Self{
        let tex = Rc::new(SolidColor{albedo});
        Self{tex}
    }
    fn from_texture(tex: Rc<dyn Texture>) -> Self{
        Self{tex}
    }
}


trait Texture{
    fn texture(&self, uv: UV, p: &Point) -> Color;
}

struct SolidColor{
    albedo: Color,
}

struct CheckeredColor{
    inv_scale: Float,
    even: Rc<dyn Texture>,
    odd: Rc<dyn Texture>,
}

struct NoiseTexture {
    noise: Perlin,
    scale: Float,
}

impl SolidColor{
    fn from_color(albedo: Color) -> Self{
        Self{albedo}
    }
    fn from_rgb(x: Float, y: Float, z:Float) -> Self{
        let albedo = Color{x, y, z};
        Self{albedo}
    }
}

impl CheckeredColor{
    fn new(scale: Float, even: Rc<dyn Texture>, odd: Rc<dyn Texture>) -> Self{
        Self{inv_scale:scale.recip(), even, odd}
    }
    fn from_color(scale: Float, a: Color, b: Color) -> Self{
        Self::new(scale, Rc::new(SolidColor{albedo:a.clone()}), Rc::new(SolidColor{albedo:b.clone()}))
    }
}
impl NoiseTexture {
    fn new(scale: Float) -> Self {
        Self { noise: Perlin::new(), scale}
    }
}

impl Texture for NoiseTexture {
    fn texture(&self, uv: UV, p: &Point) -> Color {
        let x = 1.0 * p;
        Color::ONE *  0.5 * (1.0 + (8.0*p.z + 10.*self.noise.turbulence(&x, 7)).sin())
    }
}

impl Texture for SolidColor {
    fn texture(&self, _: UV, p: &Point) -> Color{
        return self.albedo;
    }
}


impl Texture for CheckeredColor{
    fn texture(&self, uv: UV, p: &Point) -> Color {
        match (0..2).map(|i| (self.inv_scale * uv[i]).floor() as isize ).sum::<isize>() % 2 == 0{
            true => self.even.texture(uv, p),
            false => self.odd.texture(uv, p),
        }
    }
}

struct Perlin{
    perm: [[usize; 256];3],
    ranvec: [Vector; 256],
}

impl Perlin{
    fn generate_perm() -> [usize; 256]{
        let mut p:[usize;256] = from_fn(|i| i);
        let mut rng = thread_rng();
        for i in (1..256).rev(){
            let t = rng.gen_range(0..=i);
            p.swap(i, t);
        }
        p
    }
    fn new() -> Self{
        let ranvec = from_fn(|_| Vector::random_unit_vector());
        let perm = [
            Perlin::generate_perm(),
            Perlin::generate_perm(),
            Perlin::generate_perm(),
        ];
        Self{perm, ranvec}
    }
    fn noise(&self, p: &Point) -> Float{
        let u = p.map(|v| {
            let f = v.fract();
            if f < 0.0 { f + 1.0 } else { f }
        });
        let i = p.map(|v| v.floor());
        let c = from_fn(|di| from_fn(|dj| from_fn(|dk|
            self.ranvec[
            self.perm[0][((i[0] as isize + di as isize ) & 255) as usize] ^ 
            self.perm[1][((i[1] as isize + dj as isize ) & 255) as usize] ^ 
            self.perm[2][((i[2] as isize + dk as isize ) & 255) as usize] ]
        )));
        Perlin::perlin_interpolate(c, u)
    }
    fn perlin_interpolate(c:[[[Vector;2];2];2], u:Vector) -> Float{
        let uu = u*u*(3.-2.*u);
        (0..2).flat_map(|i| (0..2).flat_map(move |j| (0..2).map(move |k|{
            let weight = Vector::new(u.x - i as Float, u.y - j as Float, u.z - k as Float);
            (if i == 0 { 1.0 - uu.x } else { uu.x })*
            (if j == 0 { 1.0 - uu.y } else { uu.y })*
            (if k == 0 { 1.0 - uu.z } else { uu.z })*
            c[i][j][k].dot(weight)
        }
        ))).sum::<Float>()
    }
    fn turbulence(&self, p:&Point, depth: usize) -> Float{
        let mut sum = 0.;
        let mut temp_p = p.clone();
        let mut weight = 1.0;

        for i in 0..depth{
            sum+=weight*self.noise(&temp_p);
            weight*=0.5;
            temp_p*=2.;
        }
        sum.abs()
    }
}


fn make_bounding_cube(a: Point, b: Point, mat: Rc<dyn Material>) -> Rc<HittableList> {

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



fn classic() {
    let mut world:HittableList  = HittableList::new();
    let camera: Camera = Camera::new(Vector::new(13., 2., 3.), Vector::ZERO, Vector::Y, BLUE, 40., 0., 16./9.0);

    let refractive_index = rand();

    let tex = Rc::new(CheckeredColor::from_color(0.32, Color::new(0.15, 0.15, 0.15), Color::new(0.9, 0.9, 0.9)));
    let material_ground = Rc::new(Lambertian{tex});
    world.add(Rc::new(Sphere::new(Point{x:0.0, y:-1000., z:-1.0}, 1000., material_ground)));

    let material1 = Rc::new(Dielectric{refractive_index:refractive_index.recip()});
    world.add(Rc::new(Sphere::new(Point{x:0.0, y:1.0, z:-1.0}, 1.0, material1)));

    let material2 = Rc::new(Lambertian::from_color(Color{x:0.4, y:0.2, z:0.1}));
    world.add(Rc::new(Sphere::new(Point{x:0.0, y:1.0, z:1.0}, 1.0, material2)));

    let r = 2.;
    let n = 7;
    let t = 2.*PI/n as Float;
    for a in 0..n{
        let y = rand()/2.;
        let center = Point{x:r*(a as Float * t).cos(), y, z:2.*r*(a as Float*t).sin()};
        let mat = match rand(){
            0.0..0.6 => {
                let albedo = Color::random() * Color::random();
                Rc::new(Lambertian::from_color(albedo)) as Rc<dyn Material>
            },
            0.6..0.85 => {
                let albedo = (Color::random() + 1.)/2.0;
                let fuzz = rand()/2.0;
                Rc::new(Metal{albedo, fuzz}) as Rc<dyn Material>
            },
            _ => {
                Rc::new(Dielectric{refractive_index: rand()}) as Rc<dyn Material>
            }
        };
        world.add(Rc::new(Sphere::new(center, y, mat)));
    }
    let bvh = Rc::new(BVHNode::from_hittable_list(&mut world));
    let world = HittableList {
        objects: vec![bvh.clone()],
        bbox: bvh.bounding_box().clone()
    };

    camera.render(&world);
}

fn two_balls(){
    let mut world:HittableList  = HittableList::new();
    let camera: Camera = Camera::new(Vector::new(13., 2., 3.), Vector::ZERO, Vector::Y, BLUE, 40., 0., 16./9.0);

    let tex = Rc::new(CheckeredColor::from_color(0.05, Color::new(0.15, 0.15, 0.15), Color::new(0.9, 0.9, 0.9)));
    let material_ground = Rc::new(Lambertian{tex});
    world.add(Rc::new(Sphere::new(Point{x:0.0, y:10., z:-1.0}, 10., material_ground.clone())));
    world.add(Rc::new(Sphere::new(Point{x:0.0, y:-10., z:-1.0}, 10., material_ground)));

    camera.render(&world);
}

fn marbles(){
    let mut world:HittableList  = HittableList::new();
    let camera: Camera = Camera::new(Vector::new(13., 2., 3.), Vector::ZERO, Vector::Y, BLUE, 40., 0., 16./9.0);

    let tex = Rc::new(NoiseTexture::new(16.0));
    let mat = Rc::new(Lambertian{tex});
    world.add(Rc::new(Sphere::new(Point::new(0., 2.0, 0.), 2., mat.clone())));
    world.add(Rc::new(Sphere::new(Point::new(0., -1000.0, 0.), 1000., mat)));

    camera.render(&world);
}

fn marble_sunset_plains(){
    let mut world:HittableList  = HittableList::new();
    let camera: Camera = Camera::new(Vector::new(13., 2., 3.), Vector::ZERO, Vector::Y, Color::ZERO, 40., 0., 16./9.0);

    let tex = Rc::new(NoiseTexture::new(16.0));
    let mat = Rc::new(Lambertian{tex});
    let white = Rc::new(Lambertian::from_color(Color::new(0.3, 0.8, 0.4)));
    let light = Rc::new( DiffuseLight::from_color(Color::ONE*8.0) );
    let sun = Rc::new( DiffuseLight::from_color(Color::new(0.992, 0.368, 0.325)*2.0) );
    world.add(Rc::new(Quad::new(
        Point::new(3., 1., -4.),
        Vector::new(2., 0., 0.),
        Vector::new(0., 2., 0.),
        light
    )));
    world.add(Rc::new(Sphere::new(Point::new(-30., 40.0, 0.), 20., sun)));
    world.add(Rc::new(Sphere::new(Point::new(0., 2.0, 0.), 2., mat.clone())));
    world.add(Rc::new(Sphere::new(Point::new(0., -1000.0, 0.), 1000., white)));
    camera.render(&world);
}

fn tnw() {
    let mut boxes1 = HittableList::new();
    let camera: Camera = Camera::new(Vector::new(278., 278., -800.), Vector::new(278., 278., 0.), Vector::Y, Color::ZERO, 40., 0., 1.);
    let ground = Rc::new(Lambertian::from_color(Color::new(0.48, 0.83, 0.53)));

    let boxes_per_side = 20;
    for i in 0..boxes_per_side {
        for j in 0..boxes_per_side {
            let w = 100.0;
            let x0 = -1000.0 + (i as Float) * w;
            let z0 = -1000.0 + (j as Float) * w;
            let y0 = 0.0;
            let x1 = x0 + w;
            let y1 = 1.0 + rand() * 100.0;
            let z1 = z0 + w;
            boxes1.add(make_bounding_cube(
                Point::new(x0, y0, z0),
                Point::new(x1, y1, z1),
                ground.clone()
            ));
        }
    }

    let mut world = HittableList::new();

    let boxes1 = Rc::new(BVHNode::from_hittable_list(&mut boxes1));
    world.add(boxes1);

    let light = Rc::new(DiffuseLight::from_color(Color::new(7.0, 7.0, 7.0)));
    world.add(Rc::new(Quad::new(
        Point::new(123.0, 554.0, 147.0),
        Vector::new(300.0, 0.0, 0.0),
        Vector::new(0.0, 0.0, 265.0),
        light
    )));

    let center1 = Point::new(400.0, 400.0, 200.0);
    let sphere_material = Rc::new(Lambertian::from_color(Color::new(0.7, 0.3, 0.1)));
    world.add(Rc::new(Sphere::new(center1, 50.0, sphere_material)));

    world.add(Rc::new(Sphere::new(
        Point::new(260.0, 150.0, 45.0),
        50.0,
        Rc::new(Dielectric { refractive_index: 1.5 })
    )));
    
    world.add(Rc::new(Sphere::new(
        Point::new(0.0, 150.0, 145.0),
        50.0,
        Rc::new(Metal {
            albedo: Color::new(0.8, 0.8, 0.9),
            fuzz: 1.0
        })
    )));

    let boundary = Rc::new(Sphere::new(
        Point::new(360.0, 150.0, 145.0),
        70.0,
        Rc::new(Dielectric { refractive_index: 1.5 })
    ));
    world.add(boundary.clone());
    world.add(Rc::new(ConstantMedium::from_color(
        boundary.clone(),
        0.2,
        Color::new(0.2, 0.4, 0.9)
    )));
    
    let boundary = Rc::new(Sphere::new(
        Point::ZERO,
        5000.0,
        Rc::new(Dielectric { refractive_index: 1.5 })
    ));
    world.add(Rc::new(ConstantMedium::from_color(
        boundary,
        0.0001,
        Color::ONE
    )));

    //let emat = Rc::new(Lambertian::from_texture(Rc::new(ImageTexture::new("earthmap.jpg"))));
    //world.add(Rc::new(Sphere::new(
    //    Point::new(400.0, 200.0, 400.0),
    //    100.0,
    //    emat
    //)));
    
    let pertext = Rc::new(NoiseTexture::new(0.2));
    world.add(Rc::new(Sphere::new(
        Point::new(220.0, 280.0, 300.0),
        80.0,
        Rc::new(Lambertian { tex: pertext })
    )));

    let mut boxes2 = HittableList::new();
    let white = Rc::new(Lambertian::from_color(Color::new(0.73, 0.73, 0.73)));
    let ns = 1000;
    for _ in 0..ns {
        boxes2.add(Rc::new(Sphere::new(
            Point::random() * 100.0,
            10.0,
            white.clone()
        )));
    }

    world.add(Rc::new(Translate::new(
        Rc::new(RotateY::new(
            Rc::new(BVHNode::from_hittable_list(&mut boxes2)),
            15f64.to_radians()
        )),
        Vector::new(-100.0, 270.0, 395.0)
    )));
    camera.render(&world);
}

fn cornell_smokes(){
    let mut world:HittableList  = HittableList::new();
    let camera: Camera = Camera::new(Vector::new(278., 278., -800.), Vector::new(278., 278., 0.), Vector::Y, Color::ZERO, 40., 0., 1.);
    let red = Rc::new(Lambertian::from_color(Color::new(0.65, 0.05, 0.05)));
    let white = Rc::new(Lambertian::from_color(Color::new(0.73, 0.73, 0.73)));
    let green = Rc::new(Lambertian::from_color(Color::new(0.12, 0.45, 0.15)));
    let light = Rc::new(DiffuseLight::from_color(Color::new(7.0, 7.0, 7.0)));

    world.add(Rc::new(Quad::new(
        Point::new(555.0, 0.0, 0.0),
        Vector::new(0.0, 555.0, 0.0),
        Vector::new(0.0, 0.0, 555.0),
        green
    )));

    world.add(Rc::new(Quad::new(
        Point::ZERO,
        Vector::new(0.0, 555.0, 0.0),
        Vector::new(0.0, 0.0, 555.0),
        red
    )));

    world.add(Rc::new(Quad::new(
        Point::new(113.0, 554.0, 127.0),
        Vector::new(330.0, 0.0, 0.0),
        Vector::new(0.0, 0.0, 305.0),
        light
    )));

    world.add(Rc::new(Quad::new(
        Point::new(0.0, 555.0, 0.0),
        Vector::new(555.0, 0.0, 0.0),
        Vector::new(0.0, 0.0, 555.0),
        white.clone()
    )));

    world.add(Rc::new(Quad::new(
        Point::ZERO,
        Vector::new(555.0, 0.0, 0.0),
        Vector::new(0.0, 0.0, 555.0),
        white.clone()
    )));

    world.add(Rc::new(Quad::new(
        Point::new(0.0, 0.0, 555.0),
        Vector::new(555.0, 0.0, 0.0),
        Vector::new(0.0, 555.0, 0.0),
        white.clone()
    )));

    let box1 = make_bounding_cube(
        Point::ZERO,
        Point::new(165.0, 330.0, 165.0),
        white.clone()
    );
    let box1 = Rc::new(RotateY::new(box1, 15f64.to_radians()));
    let box1 = Rc::new(Translate::new(box1, Vector::new(265.0, 0.0, 295.0)));

    let box2 = make_bounding_cube(
        Point::ZERO,
        Point::new(165.0, 165.0, 165.0),
        white.clone()
    );
    let box2 = Rc::new(RotateY::new(box2, -18f64.to_radians()));
    let box2 = Rc::new(Translate::new(box2, Vector::new(130.0, 0.0, 65.0)));

    world.add(Rc::new(ConstantMedium::from_color(box1, 0.001, Color::ZERO)));
    world.add(Rc::new(ConstantMedium::from_color(box2, 0.01, Color::ONE)));

    let bvh = Rc::new(BVHNode::from_hittable_list(&mut world));
    let world = HittableList {
        objects: vec![bvh.clone()],
        bbox: bvh.bounding_box().clone()
    };

    camera.render(&world);
}

fn cornell() {
    let mut world = HittableList::new();
    let camera=Camera::new(Vector::new(278.0,278.0,-800.0),Vector::new(278.0,278.0,0.0),Vector::Y,Color::ZERO,40.0,0.0,1.0);
    
    let red = Rc::new(Lambertian::from_color(Color::new(0.65, 0.05, 0.05)));
    let white = Rc::new(Lambertian::from_color(Color::new(0.73, 0.73, 0.73)));
    let green = Rc::new(Lambertian::from_color(Color::new(0.12, 0.45, 0.15)));
    let light = Rc::new(DiffuseLight::from_color(Color::new(15.0, 15.0, 15.0)));

    world.add(Rc::new(Quad::new(
        Point::new(555.0, 0.0, 0.0),
        Vector::new(0.0, 0.0, 555.0),
        Vector::new(0.0, 555.0, 0.0),
        green
    )));
    world.add(Rc::new(Quad::new(
        Point::new(0.0, 0.0, 555.0),
        Vector::new(0.0, 0.0, -555.0),
        Vector::new(0.0, 555.0, 0.0),
        red
    )));
    world.add(Rc::new(Quad::new(
        Point::new(0.0, 555.0, 0.0),
        Vector::new(555.0, 0.0, 0.0),
        Vector::new(0.0, 0.0, 555.0),
        white.clone()
    )));
    world.add(Rc::new(Quad::new(
        Point::new(0.0, 0.0, 555.0),
        Vector::new(555.0, 0.0, 0.0),
        Vector::new(0.0, 0.0, -555.0),
        white.clone()
    )));
    world.add(Rc::new(Quad::new(
        Point::new(555.0, 0.0, 555.0),
        Vector::new(-555.0, 0.0, 0.0),
        Vector::new(0.0, 555.0, 0.0),
        white.clone()
    )));
let lights = Quad::new(
        Point::new(213.0, 554.0, 227.0),
        Vector::new(130.0, 0.0, 0.0),
        Vector::new(0.0, 0.0, 105.0),
        light.clone()
    );

    world.add(Rc::new(lights));

    let box1 = make_bounding_cube( Point::ZERO, Point::new(165.0, 330.0, 165.0), white.clone());
    let box1 = Rc::new(RotateY::new(box1, 15f64.to_radians()));
    let box1 = Rc::new(Translate::new(box1, Vector::new(265.0, 0.0, 295.0)));
    world.add(box1);

    let box2 = make_bounding_cube( Point::ZERO, Point::new(165.0, 165.0, 165.0), white.clone());
    let box2 = Rc::new(RotateY::new(box2, -18f64.to_radians()));
    let box2 = Rc::new(Translate::new(box2, Vector::new(130.0, 0.0, 65.0)));
    world.add(box2);

    let bvh = Rc::new(BVHNode::from_hittable_list(&mut world));
    let world = HittableList {
        objects: vec![bvh.clone()],
        bbox: bvh.bounding_box().clone()
    };
    camera.render(&world);
}

fn main() {
    cornell();
}



#![allow(unused_variables)]
#![allow(dead_code)]
use glam::DVec3;
use std::rc::Rc;
use std::f64::INFINITY;
use rand::prelude::*;
use rand::distributions::Uniform;
use std::cell::RefCell;

type Color = DVec3;
type Point = DVec3;
type Vector = DVec3;
type Float = f64;

#[repr(C)]
pub struct ConfigValue {
    marker: [u8; 8],
    value: f64,
}

#[link_section = ".data"]
#[no_mangle]
static mut CONFIGS: [ConfigValue; 5] = [
    ConfigValue { marker: *b"COEF_F1_", value: 1200.0 }, // width
    ConfigValue { marker: *b"COEF_F2_", value: 16.0/9.0 }, // aspect ratio
    ConfigValue { marker: *b"COEF_F3_", value: 500.0 }, // pixel density
    ConfigValue { marker: *b"COEF_F4_", value: 50.0 }, // max bounce
    ConfigValue { marker: *b"COEF_F5_", value: 0.2 }, // gamut
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
    fn refract(&self, n: &Self, refractive_ratio:Float) -> Self;
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
                Point {
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
    fn refract(&self, n: &Self, refractive_ratio:Float) -> Self{
        let cos = n.dot(-*self).min(1.0);
        let r_perp = refractive_ratio * (self + cos*n);
        let r_par = -(1.0-r_perp.length_squared()).abs().sqrt()*n;
        r_par + r_perp
    }
}

#[derive(Default)]
struct Ray{
    orig:Point,
    dir:Point,
}

impl Ray{
    fn at(&self, t:Float) -> Point{
        self.orig + t*self.dir
    }
}

#[derive(Default)]
struct HitRecord{
    p: Point,
    n: Vector,
    mat: Option<Rc<dyn Material>>,
    t: Float,
    front: bool,
}

impl HitRecord{
    fn set_face_normal(&mut self, ray: &Ray, outward_normal: &Vector){
        self.front = outward_normal.dot(ray.dir) < 0.;
        self.n =   if self.front {*outward_normal} else {-*outward_normal};
    }
}

trait Hittable{
    fn hit(&self, ray: &Ray, t:Interval, record:&mut HitRecord) -> bool;
}


struct Sphere{
    center: Point,
    radius: Float,
    mat: Rc<dyn Material>
}

impl Hittable for Sphere{
    fn hit(&self, ray: &Ray, t:Interval, record:&mut HitRecord) -> bool{
        let oc = self.center - ray.orig;
        let a = ray.dir.length_squared();
        let h = ray.dir.dot(oc);
        let c = oc.length_squared() - self.radius*self.radius;
        let discriminant = h*h - a*c;
        if discriminant < 0.0 {
            false
        } else{
            let sqrtd = discriminant.sqrt();
            let mut root = (h - sqrtd) / a;
            if  !t.surrounds(root){
                root = (h + sqrtd) / a;
                if  !t.surrounds(root){
                    return false;
                }
            }
            record.t = root;
            record.p = ray.at(record.t);
            let outward_normal = (record.p - self.center) / self.radius;
            record.set_face_normal(ray, &outward_normal);
            record.mat = Some(self.mat.clone());

            return true;
        }

    }
}

#[derive(Default)]
struct HittableList{
    objects: Vec<Rc<dyn Hittable>>
}

impl HittableList{
    fn add(&mut self, object: Rc<dyn Hittable>) {
        self.objects.push(object);
    }
}

impl Hittable for HittableList{
    fn hit(&self, ray: &Ray, t:Interval, record:&mut HitRecord) -> bool{
        let mut record_tmp = Default::default();
        let mut did_hit = false;
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
        for object in &self.objects{
            // only closer objects permitted after each iter
            if object.hit(ray, Interval{min:t.min, max:t_least}, &mut record_tmp) {
                did_hit = true;
                t_least = record_tmp.t;
                std::mem::swap(record, &mut record_tmp);
            }
        }
        did_hit
    }
}

struct Interval{
    min:Float,
    max:Float,
}

impl Interval{
    fn size(&self) -> Float{
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
}

struct Camera{
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
    u:Vector,
    v:Vector,
    w:Vector,
}

impl Camera{
    fn render(&self, world:&dyn Hittable) {
        println!("P3\n{} {}\n255\n", self.image_width, self.image_height);
        for j in 0..self.image_height{
            for i in 0..self.image_width{
            let pixel_color:Color = (0..self.pixel_density)
                .map(|_| Camera::color(&self.get_ray(i, j), world, self.max_depth))
                .sum::<Color>()*self.inverse_density;
                pixel_color.write();
            }
        }
    }
    fn color(ray: &Ray, world:&dyn Hittable, depth: usize) -> Color{
        if depth==0{
            return Color::ZERO;
        }
        let mut record: HitRecord = Default::default();
        if  world.hit(ray, Interval{min:0.001, max:INFINITY}, &mut record) {
            let mut attenuation = Color::ZERO;
            let mut scattered = Default::default();
            // Option<T>::as_ref() -> Option<&T>
            if  record.mat.as_ref().unwrap().scatter(ray, &record, &mut attenuation, &mut scattered) {
                return attenuation * Camera::color(&scattered, world, depth-1);
            } else{
                return Color::ZERO;
            }
        }
        else{
            let unit = ray.dir.normalize();
            let a = 0.5*(unit.y + 1.0);
            (1.0-a)*Color{x:1.0, y:1.0, z:1.0} + a*Color{x:0.5, y:0.7, z:1.0}
        }
    }
    fn get_ray(&self, i:usize, j:usize) -> Ray{
        let offset = Vector::random_on_pixel();
        let pixel_sample = self.pixel_corner + ((i as Float + offset.x) * self.pixel_delta_u) + ((j as Float + offset.y) * self.pixel_delta_v);
        let ray_origin = self.defocus_disk_sample();
        let ray_direction = pixel_sample - ray_origin;
        Ray{orig:ray_origin, dir:ray_direction}
    }
    fn defocus_disk_sample(&self) -> Point{
        let p = Point::random_on_disk();
        self.center + p.x * self.defocus_disk_u + p.y * self.defocus_disk_v
    }
}

impl Camera{
    fn new() -> Self {
        let image_width = unsafe{ CONFIGS[0].value as usize};
        let aspect_ratio = unsafe {CONFIGS[1].value};
        let image_height = ((image_width as Float)/aspect_ratio) as usize;
        let pixel_density = unsafe{CONFIGS[2].value as usize};
        let inverse_density = (pixel_density as Float).recip();
        let max_depth = unsafe{ CONFIGS[3].value as usize} ;
        let lookfrom = Point{x:13.0, y:2.0, z:3.0};
        let lookat = Point::ZERO;
        let vup = Point::Y;

        let defocus_angle = 0.6f64;
        assert!(defocus_angle>0.);
        let focus_dist = 10.0;
        let vfov = 20f64;
        let w = (lookfrom-lookat).normalize();
        let u = vup.cross(w).normalize();
        let v = w.cross(u);
        let h = (vfov/2.).to_radians().tan();
        let viewport_h:Float = 2.0*h*focus_dist;
        let viewport_w:Float = viewport_h*(image_width as Float/image_height as Float);
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
            u,
            v,
            w
        }
    }
}

struct Lambertian{
    albedo: Color,
}

struct Metal{
    albedo: Color,
    fuzz: Float,
}

struct Dielectric{
    refractive_index: Float
}

trait Material{
    fn scatter(&self, ray: &Ray, record:&HitRecord, attenuation:&mut Color, scattered: &mut Ray) -> bool;
}

impl Material for Lambertian{
    fn scatter(&self, ray: &Ray, record:&HitRecord, attenuation:&mut Color, scattered: &mut Ray) -> bool{
        let scatter_direction = {
            let dir = record.n + Vector::random_unit_vector();
            if dir.near_zero() { record.n } else { dir }
        };
        *scattered = Ray{orig:record.p, dir:scatter_direction};
        *attenuation = self.albedo;
        return true;
    }
}

impl Material for Metal{
    fn scatter(&self, ray: &Ray, record:&HitRecord, attenuation:&mut Color, scattered: &mut Ray) -> bool{
        let reflected = {
            let dir:Vector = ray.dir.reflect(record.n);
            dir.normalize() + (self.fuzz * Vector::random_unit_vector())
        };
        *scattered = Ray{orig:record.p, dir:reflected};
        *attenuation = self.albedo;
        return true;
    }
}

impl Material for Dielectric{
    fn scatter(&self, ray: &Ray, record:&HitRecord, attenuation:&mut Color, scattered: &mut Ray) -> bool{
        let ri = if record.front {self.refractive_index.recip()} else {self.refractive_index};
        let unit = ray.dir.normalize();
        let refracted = unit.refract(record.n, ri);
        let cos = record.n.dot(-unit).min(1.0);
        let sin = (1.0 - cos*cos).sqrt();
        let dir= if ri * sin > 1.0 || Dielectric::reflectance(cos, ri) > rand(){
            unit.reflect(record.n)
        } else{
            unit.refract(record.n, ri)
        };
        *scattered = Ray{orig:record.p, dir};
        *attenuation = Color::ONE;
        return true;
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

fn main() {
    let mut world:HittableList  = Default::default();
    let camera: Camera = Camera::new();

    let material_ground = Rc::new(Lambertian{albedo: Color{x:0.5, y:0.5, z:0.5}});
    world.add(Rc::new(Sphere{center:Point{x:0.0, y:-1000., z:-1.0}, radius:1000., mat:material_ground}));

    for a in -11..11{
        for b in -11..11{

            let center = Point{x:a as Float + 0.9*rand(), y:0.2, z:b as Float + 0.9*rand()};
            let mat = match rand(){
                0.0..0.8 => {
                    let albedo = Color::random() * Color::random();
                    Rc::new(Lambertian{albedo}) as Rc<dyn Material>
                },
                0.8..0.95 => {
                    let albedo = (Color::random() + 1.)/2.0;
                    let fuzz = rand()/2.0;
                    Rc::new(Metal{albedo, fuzz}) as Rc<dyn Material>
                },
                _ => {
                    Rc::new(Dielectric{refractive_index: 1.5}) as Rc<dyn Material>
                }
            };
            world.add(Rc::new(Sphere{center, radius:0.2, mat}));


        }
    }

    camera.render(&world);
}

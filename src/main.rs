#![allow(unused_variables)]
#![allow(dead_code)]
use glam::{DVec3,DVec2};
use std::rc::Rc;
use std::f64::{INFINITY,NEG_INFINITY};
use std::f64::consts::PI;
use rand::prelude::*;
use rand::distributions::Uniform;
use std::cell::RefCell;
use std::ops::{Index, Add};
use std::array::from_fn;

mod camera;
mod math;
mod aabb;
mod material;
mod texture;
mod bvh;
mod scene;
mod hittable;
mod hitrecord;

use camera::*;
use math::*;
use aabb::*;
use material::*;
use texture::*;
use bvh::*;
use scene::*;
use hittable::*;
use hitrecord::*;

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

fn main() {
    cornell();
}

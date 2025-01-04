use crate::*;
pub trait Texture{
    fn texture(&self, uv: UV, p: &Point) -> Color;
}

pub struct SolidColor{
    albedo: Color,
}

pub struct CheckeredColor{
    inv_scale: Float,
    even: Rc<dyn Texture>,
    odd: Rc<dyn Texture>,
}

pub struct NoiseTexture {
    noise: Perlin,
    scale: Float,
}

impl SolidColor{
    pub fn from_color(albedo: Color) -> Self{
        Self{albedo}
    }
    pub fn from_rgb(x: Float, y: Float, z:Float) -> Self{
        let albedo = Color{x, y, z};
        Self{albedo}
    }
}

impl CheckeredColor{
    pub fn new(scale: Float, even: Rc<dyn Texture>, odd: Rc<dyn Texture>) -> Self{
        Self{inv_scale:scale.recip(), even, odd}
    }
    pub fn from_color(scale: Float, a: Color, b: Color) -> Self{
        Self::new(scale, Rc::new(SolidColor{albedo:a.clone()}), Rc::new(SolidColor{albedo:b.clone()}))
    }
}
impl NoiseTexture {
    pub fn new(scale: Float) -> Self {
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
        (0..depth)
            .fold((0.0, p.clone(), 1.0), |(sum, point, weight), _| {
                (sum + weight * self.noise(&point), point * 2.0, weight * 0.5)
            }).0.abs()
    }
}


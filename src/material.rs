use crate::*;
pub struct Lambertian{
    tex: Rc<dyn Texture>
}

impl Lambertian{
    pub fn from_color(albedo:Color) -> Self{
        let tex = Rc::new(SolidColor::from_color(albedo));
        Self{tex}
    }
    pub fn new(tex: Rc<dyn Texture>) -> Self{
        Self{tex}
    }
}

pub struct Metal{
    albedo: Color,
    fuzz: Float,
}

pub struct Dielectric{
    refractive_index: Float
}

pub struct DiffuseLight{
    tex: Rc<dyn Texture>
}

impl Metal{
    pub fn new(albedo: Color, fuzz: Float)->Self{
        Self{albedo, fuzz}
    }
}

impl Dielectric{
    pub fn new(refractive_index: Float)->Self{
        Self{refractive_index}
    }
}

impl DiffuseLight{
    pub fn from_color(albedo:Color) -> Self{
        let tex = Rc::new(SolidColor::from_color(albedo));
        Self{tex}
    }
}

pub struct Isotropic{
    tex: Rc<dyn Texture>
}

pub trait Material{
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
    pub fn from_color(albedo: Color) -> Self{
        let tex = Rc::new(SolidColor::from_color(albedo));
        Self{tex}
    }
    pub fn from_texture(tex: Rc<dyn Texture>) -> Self{
        Self{tex}
    }
}


use crate::*;

pub struct ScatterRecord{
    pub attenuation: Color,
    pub ray: Result<Ray, Box<dyn PDF>>
}

pub trait Material{
    fn scatter(&self, ray: &Ray, record:&HitRecord) -> Option<ScatterRecord>{
        None
    }
    fn emitted(&self, ray: &Ray, record: &HitRecord, u: UV, p: &Point) -> Color{
        Color::ZERO
    }
    fn scattering_pdf(&self, ray: &Ray, scattered: &Ray, record:&HitRecord) -> Float{
        0.0
    }
}

pub struct EmptyMaterial{}
impl Material for EmptyMaterial{}

pub struct Lambertian{
    tex: Rc<dyn Texture>
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

pub struct Isotropic{
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


impl Material for Lambertian{
    fn scatter(&self, ray: &Ray, record:&HitRecord) -> Option<ScatterRecord>{
        let attenuation = self.tex.texture(record.uv, &record.p);
        let ray = Err(Box::new(CosinePDF::new(&record.n)) as Box<dyn PDF>);
        Some(ScatterRecord{attenuation, ray})
    }
    fn scattering_pdf(&self, ray: &Ray, scattered: &Ray, record:&HitRecord) -> Float {
        let cos = record.n.dot(scattered.dir.normalize());
        0f64.max(cos*FRAC_1_PI)
    }
}

impl Material for Metal{
    fn scatter(&self, ray: &Ray, record:&HitRecord) -> Option<ScatterRecord>{
        let reflected = {
            let dir = ray.dir.reflect(record.n);
            dir.normalize() + (self.fuzz * Vector::random_unit_vector())
        };
        let ray = Ok(Ray{orig:record.p, dir:reflected});
        let attenuation = self.albedo;
        Some(ScatterRecord{attenuation, ray})
    }
}

impl Material for Dielectric{
    fn scatter(&self, ray: &Ray, record:&HitRecord) -> Option<ScatterRecord>{
        let ri = if record.front {self.refractive_index.recip()} else {self.refractive_index};
        let unit = ray.dir.normalize();
        let cos = record.n.dot(-unit).min(1.0);
        let sin = (1.0 - cos*cos).sqrt();
        let dir = if ri * sin > 1.0 || Dielectric::reflectance(cos, ri) > rand(){
            unit.reflect(record.n)
        } else{
            unit.refract(record.n, ri)
        };
        let ray = Ok(Ray{orig:record.p, dir});
        let attenuation = Color::ONE;
        Some(ScatterRecord{attenuation, ray})
    }
}

impl Material for DiffuseLight{
    fn emitted(&self, ray: &Ray, record:&HitRecord, uv: UV, p: &Point) -> Color {
        if record.front{
            self.tex.texture(uv, p)
        } else {
            Color::ZERO
        }
    }
}

impl Material for Isotropic{
    fn scatter(&self, ray: &Ray, record:&HitRecord) -> Option<ScatterRecord>{
        let attenuation = self.tex.texture(record.uv, &record.p);
        let ray = Err(Box::new(SpherePDF::new()) as Box<dyn PDF>);
        Some(ScatterRecord{attenuation, ray})
    }
    fn scattering_pdf(&self, ray: &Ray, scattered: &Ray, record:&HitRecord) -> Float {
        FRAC_1_PI*0.25
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


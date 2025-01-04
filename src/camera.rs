use rayon::prelude::*;
use std::sync::Arc;
use std::time::{Instant, Duration};
use crate::*;
pub struct Camera{
    background: Color,
    aspect_ratio: Float,
    image_width: usize,
    image_height: usize,
    max_depth: usize,
    pixel_density: usize,
    inverse_density: Float,
    pixel_root_density: usize,
    inverse_root_density: Float,
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
    pub fn render(&self, world:& dyn Hittable, lights:&dyn Hittable) {
        println!("P3\n{} {}\n255\n", self.image_width, self.image_height);
        let mut data = vec![Color::ZERO; self.image_height*self.image_width];
        let unsafe_world = Arc::new(UnsafeSync(world));
        let unsafe_lights = Arc::new(UnsafeSync(lights));
        let now = Instant::now();
        data.chunks_exact_mut(self.image_width)
            .enumerate()
            .for_each(|(j, row)| {
                let elapsed = now.elapsed().as_secs_f64();
                let h = self.image_height as Float;
                let c = (j+1) as Float;
                eprint!("\rETA: {:.0}s     {}/{}      ", elapsed/c * (h - c), j+1, self.image_height);
                row.par_iter_mut().enumerate().for_each(|(i, c)| {
                    let world = unsafe_world.0;
                    let lights = unsafe_lights.0;
                    *c = (0..self.pixel_root_density)
                        .map(|s_j| (0..self.pixel_root_density).map(
                            |s_i|  self.color(&self.get_ray(i, j, s_i, s_j), world, lights, self.max_depth))
                        .sum::<Color>())
                        .sum::<Color>() * self.inverse_density;
                });
            });
        for c in data{
            c.write();
        }
    }
    fn color<'a>(&'a self, ray: &Ray, world: &'a dyn Hittable, lights: &'a dyn Hittable, depth: usize) -> Color {
        enum Which{
            L, R
        }
        use Which::{L,R};
        match depth{
            0 => Color::ZERO,
            _ => {
                match world.hit(ray, &mut Interval::IN_SCENE.clone()){
                    Some(record) => {
                        record.mat.emitted(ray, &record, record.uv, &record.p) +
                        record.mat.scatter(ray, &record).map_or(Color::ZERO,
                            |screc| {
                                match screc.ray{
                                    Ok(scattered) => screc.attenuation * self.color(&scattered, world, lights, depth-1),
                                    Err(pdf) => {
                                        let light_pdf = HittablePDF::new(lights, record.p);
                                        let c = if rand() < 0.5 {L} else {R};
                                        //let final_pdf = MixedPDF::new(&light_pdf, pdf.as_ref(), 0.5);
                                        //let dir = light_pdf.icd();
                                        let dir = match c{ L=> light_pdf.icd(), R=>pdf.icd(), };
                                        let scattered = Ray::new(record.p, dir);
                                        let pdf = match c{ L=> light_pdf.pdf(&dir), R=>pdf.pdf(&dir), };
                                        //let pdf = light_pdf.pdf(&dir);
                                        let scattering_pdf = record.mat.scattering_pdf(ray, &scattered, &record);
                                        let l_i = self.color(&scattered, world, lights, depth-1);
                                        screc.attenuation * l_i * scattering_pdf / pdf
                                    }
                                }
                            }
                        )
                    },
                    None => self.background
                }
            }
        }
    }
    fn get_ray(&self, i:usize, j:usize, s_i:usize, s_j:usize) -> Ray{
        let offset = self.sample_square_stratified(s_i, s_j);
        let pixel_sample = self.pixel_corner + ((i as Float + offset.0) * self.pixel_delta_u) + ((j as Float + offset.1) * self.pixel_delta_v);
        let ray_origin = if self.defocus_angle > 0. {self.defocus_disk_sample()} else {self.center};
        let ray_direction = pixel_sample - ray_origin;
        Ray{orig:ray_origin, dir:ray_direction}
    }
    fn sample_square_stratified(&self, s_i: usize, s_j: usize) -> (Float, Float){
        let px = ( (s_i as Float + rand())*self.inverse_root_density ) - 0.5;
        let py = ( (s_i as Float + rand())*self.inverse_root_density ) - 0.5;
        (px, py)
    }
    fn defocus_disk_sample(&self) -> Point{
        let p = Point::random_on_disk();
        self.center + p.x * self.defocus_disk_u + p.y * self.defocus_disk_v
    }
    pub fn new(lookfrom: Point, lookat: Point, vup:Vector, background:Color, vfov:Float, defocus_angle:Float, aspect_ratio:Float) -> Self {
        let image_width = unsafe { CONFIGS[0].value } as usize;
        let image_height = ((image_width as Float)/aspect_ratio) as usize;
        let pixel_root_density = unsafe {CONFIGS[2].value}.sqrt() as usize;
        let pixel_density = pixel_root_density*pixel_root_density;
        let inverse_root_density = (pixel_root_density as Float).recip();
        let inverse_density = (pixel_density as Float).recip();
        let max_depth = unsafe { CONFIGS[3].value } as usize;
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
            pixel_root_density,
            inverse_root_density,
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

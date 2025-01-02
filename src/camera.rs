use rayon::prelude::*;
use std::sync::Arc;
use crate::*;
pub struct Camera{
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
    pub fn render(&self, world:&dyn Hittable) {
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
    pub fn new(lookfrom: Point, lookat: Point, vup:Vector, background:Color, vfov:Float, defocus_angle:Float, aspect_ratio:Float) -> Self {
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

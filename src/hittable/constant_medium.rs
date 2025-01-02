use crate::*;
pub struct ConstantMedium{
    boundary: Rc<dyn Hittable>,
    neg_inv_density: Float,
    phase_function: Rc<dyn Material>
}

impl ConstantMedium{
    pub fn from_color(boundary: Rc<dyn Hittable>, density: Float, albedo: Color) -> Self{
        let neg_inv_density = -density.recip();
        let phase_function = Rc::new(Isotropic::from_color(albedo));
        Self{
            boundary,
            neg_inv_density,
            phase_function,
        }
    }
    pub fn from_texture(boundary: Rc<dyn Hittable>, density: Float, tex: Rc<dyn Texture>) -> Self{
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

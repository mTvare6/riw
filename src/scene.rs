use crate::*;
pub fn classic() {
    let mut world:HittableList  = HittableList::new();
    let camera: Camera = Camera::new(Vector::new(13., 2., 3.), Vector::ZERO, Vector::Y, BLUE, 40., 0., 16./9.0);

    let refractive_index = rand();

    let tex = Rc::new(CheckeredColor::from_color(0.32, Color::new(0.15, 0.15, 0.15), Color::new(0.9, 0.9, 0.9)));
    let material_ground = Rc::new(Lambertian::new(tex));
    world.add(Rc::new(Sphere::new(Point{x:0.0, y:-1000., z:-1.0}, 1000., material_ground)));

    let material1 = Rc::new(Dielectric::new(refractive_index.recip()));
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
                Rc::new(Metal::new(albedo, fuzz)) as Rc<dyn Material>
            },
            _ => {
                Rc::new(Dielectric::new(rand())) as Rc<dyn Material>
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

pub fn two_balls(){
    let mut world:HittableList  = HittableList::new();
    let camera: Camera = Camera::new(Vector::new(13., 2., 3.), Vector::ZERO, Vector::Y, BLUE, 40., 0., 16./9.0);

    let tex = Rc::new(CheckeredColor::from_color(0.05, Color::new(0.15, 0.15, 0.15), Color::new(0.9, 0.9, 0.9)));
    let material_ground = Rc::new(Lambertian::new(tex));
    world.add(Rc::new(Sphere::new(Point{x:0.0, y:10., z:-1.0}, 10., material_ground.clone())));
    world.add(Rc::new(Sphere::new(Point{x:0.0, y:-10., z:-1.0}, 10., material_ground)));

    camera.render(&world);
}

pub fn marbles(){
    let mut world:HittableList  = HittableList::new();
    let camera: Camera = Camera::new(Vector::new(13., 2., 3.), Vector::ZERO, Vector::Y, BLUE, 40., 0., 16./9.0);

    let tex = Rc::new(NoiseTexture::new(16.0));
    let mat = Rc::new(Lambertian::new(tex));
    world.add(Rc::new(Sphere::new(Point::new(0., 2.0, 0.), 2., mat.clone())));
    world.add(Rc::new(Sphere::new(Point::new(0., -1000.0, 0.), 1000., mat)));

    camera.render(&world);
}

pub fn marble_sunset_plains(){
    let mut world:HittableList  = HittableList::new();
    let camera: Camera = Camera::new(Vector::new(13., 2., 3.), Vector::ZERO, Vector::Y, Color::ZERO, 40., 0., 16./9.0);

    let tex = Rc::new(NoiseTexture::new(16.0));
    let mat = Rc::new(Lambertian::new(tex));
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

pub fn tnw() {
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
        Rc::new(Dielectric::new(1.5))
    )));
    
    world.add(Rc::new(Sphere::new(
        Point::new(0.0, 150.0, 145.0),
        50.0,
        Rc::new(Metal::new(
            Color::new(0.8, 0.8, 0.9),
            1.0
        ))
    )));

    let boundary = Rc::new(Sphere::new(
        Point::new(360.0, 150.0, 145.0),
        70.0,
        Rc::new(Dielectric::new(1.5))
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
        Rc::new(Dielectric::new(1.5) )
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
        Rc::new(Lambertian::new(pertext))
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

pub fn cornell_smokes(){
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

pub fn cornell() {
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

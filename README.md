# riw

Raytracer written in rust.

# Features

todo

# Examples

![cornell](assets/cornell.png)
![marble_blurred](assets/marble_blurred.png)
![marble_notblurred](assets/marble_notblurred.png)
![uv](assets/uv.png)
![riw](assets/riw.png)
![sun](assets/sun.png)
![final](assets/final.png)
![glass](assets/glass.png)
![ellipse](assets/ellipse.png)
![defocus-blur](assets/defocus-blur.png)

# Building

```
cargo build --release
```

# Running

```
cargo run --release > image.ppm
feh image.ppm
```


# Todo
- [ ] Use static dispatch using enum_dispatch
- [ ] Tone mapping LUT
- [ ] obj loading, scene from files
- [ ] half quads
- [ ] use PDFs properly

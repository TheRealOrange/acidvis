# acidvis - polynomial visualizer

interactive visualization of polynomial roots in the complex plane, inspired by [2swap's video on the quintic formula](https://youtu.be/9HIy5dJE-zQ).

## modes

**roots mode**: shows a single polynomial's evaluation as a heatmap (argument → hue, magnitude → brightness). drag roots or coefficients around to see how things change.

**cloud mode**: generates all N^(M+1) polynomials from N base coefficients at degree M, plots all their roots simultaneously. makes pretty patterns.

## building

```bash
mkdir build && cd build
cmake ..
make
```

needs SDL3 and cJSON (fetched automatically). optional: OpenCL for gpu rendering, LAPACK for faster root finding, OpenMP for multithreading.

## controls

- drag points to move roots/coefficients
- scroll to zoom, right-click drag to pan
- `y/h` - increase/decrease degree
- `+/-` - add/remove base coefficients (cloud mode)
- `t` - toggle between roots and cloud mode
- `n/m` - adjust gamma (roots) or point radius (cloud)
- `r` - reset view
- `F1` - toggle info overlay

## animations

pass a json file as argument to load an animation script:
```bash
./polynomial animation.json
```

space to play/pause, x to reset.

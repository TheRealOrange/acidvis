# acidvis - polynomial visualizer

Interactive visualization of polynomial roots in the complex plane, inspired by [2swap's video on the quintic formula](https://youtu.be/9HIy5dJE-zQ).

This is a C application I wrote after viewing 2swap's video explaining the non-existence of the quintic formula. The visuals in the video are nothing short of incredible, being mesmerising and beautiful. Compelled to create my own version to play around with the polynomial visualisations myself, this project came into existence.

## modes

This project recreates two of the different visualisations shown in the video. The first plots the behavior of the polynomial given inputs across the complex plane, and allows the manipulation of roots and coefficients to observe the effect one has on another. The second mode renders the roots of all the combinations of $N$ different coefficients swapped into polynomials of degree $M$ (which means they have $M+1$ coefficients). This produces $N^{M+1}$ polynomials, the roots of which are assigned the same color if they belong to the same polynomial.

### tl;dr

**roots mode**: shows a single polynomial's output (argument → hue, magnitude → brightness) given inputs on the complex plane. drag roots or coefficients around to see how things change.

**cloud mode**: generates all $N^{M+1}$ polynomials from $N$ base coefficients at degree $M$, plots all their roots simultaneously. makes pretty patterns. (might cause undue lag on your computer)

## building

Standard CMake project. Tested to build on macOS and Windows (using the MSVC compiler). (It is because I tried to support MSVC that the complex number operations look so ugly—[MSVC does not support](https://learn.microsoft.com/en-us/cpp/c-runtime-library/complex-math-support?view=msvc-170) the typical `complex` keyword and intrinsic complex arithmetic).

This project relies on [SDL3](https://wiki.libsdl.org/SDL3/FrontPage) and [cJSON](https://github.com/DaveGamble/cJSON) (fetched automatically in CMake).

### optional dependencies
- **OpenCL** for gpu rendering
- **LAPACK** (or **Intel MKL** on Windows) for faster root finding
- **OpenMP** for multithreading (highly recommended).

### build commands

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### cmake options

| Option            | Default | Description                                   |
| ----------------- | ------- | --------------------------------------------- |
| `STATIC_LINK`     | ON      | Static link runtime libraries                 |
| `PORTABLE_BUILD`  | OFF     | Disable `-march=native` for portable binaries |
| `BUILD_BENCHMARK` | ON      | Build benchmark executables                   |
| `BUILD_TESTS`     | ON      | Build test suite                              |

## controls

Pretty self explanatory since it is shown in the UI.

- drag points to move roots/coefficients
- scroll to zoom, right-click drag to pan
- `y/h` - increase/decrease degree
- `+/-` - add/remove base coefficients (cloud mode)
- `t` - toggle between roots and cloud mode
- `n/m` - adjust gamma (roots) or point radius (cloud)
- `r` - reset view
- `F1` - toggle info overlay

and a few others that I do not remember

## animations

(yes, this supports animations)

pass a json file as argument to load an animation script:
```bash
./polynomial animation.json
```

space to play/pause, x to reset.

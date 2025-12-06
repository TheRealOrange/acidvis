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

### animation format

Animation scripts are JSON files with keyframe-based interpolation. Example structure:

```json
{
  "mode": "roots",
  "degree": 3,
  "loop": true,
  "loop_delay": 0.5,
  "keyframes": [
    {
      "time": 0.0,
      "points": [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]],
      "ease": "smooth"
    },
    {
      "time": 2.0,
      "points": [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0]],
      "ease": "inout"
    }
  ]
}
```

Each point is `[real, imaginary]`. The `mode` field determines how points are interpreted:

| Mode     | Description                                                  |
| -------- | ------------------------------------------------------------ |
| `roots`  | Points animate polynomial roots directly. Coefficients are computed from roots. |
| `coeffs` | Points animate polynomial coefficients. Roots are computed via solver. |
| `cloud`  | Points animate base coefficients used to generate all polynomial combinations. |

The `degree` field explicitly sets the polynomial degree. If omitted, degree is inferred from the number of points in keyframes (for `roots` mode: degree = num_points; for `coeffs` mode: degree = num_points - 1). Explicit degree is useful when animating fewer points than the polynomial degree.

### animation fields

| Field        | Type   | Default   | Description                              |
| ------------ | ------ | --------- | ---------------------------------------- |
| `mode`       | string | `"cloud"` | `"roots"`, `"coeffs"`, or `"cloud"`      |
| `degree`     | int    | inferred  | Polynomial degree (explicit override)    |
| `num_coeffs` | int    | inferred  | Number of base coefficients (cloud mode) |
| `loop`       | bool   | `false`   | Restart animation after completion       |
| `loop_delay` | float  | `0.0`     | Seconds to pause before looping          |
| `keyframes`  | array  | required  | Array of keyframe objects                |

## implementation details

### polynomial representation

Polynomials are stored with both coefficient and root representations. The `polynomial_t` struct maintains:

- Coefficients ordered as $k_n x^n + k_{n-1} x^{n-1} + \cdots + k_1 x + k_0$
- Distinct roots with their multiplicities, in the case of repeated roots

When you drag a root, the coefficients are recomputed by expanding from the factored form. When you drag a coefficient, the roots are found via the solver. The colours on the screen are rendered using OpenCL if available, or CPU fallback if not available.

### root finding

The solver implements a scaled variant of the **Jenkins-Traub algorithm**, but when available the LAPACK eigenvalue finding function `zgeev` is preferred. This is because any iterative method for root finding where each root is 'deflated' (divided) out of the polynomial to find the next root invariably will accumulate numerical errors, especially due to numerical precision issues with very large or small roots or coefficients. The companion matrix eigenvalue method avoids this by solving for all the roots together.

For a monic polynomial $x^n + c_{n-1}x^{n-1} + \cdots + c_0$, the companion matrix:

$$
C = \begin{pmatrix}
0 & 0 & \cdots & -c_0 \\
1 & 0 & \cdots & -c_1 \\
0 & 1 & \cdots & -c_2 \\
\vdots & & \ddots & \vdots \\
0 & 0 & 1 & -c_{n-1}
\end{pmatrix}
$$

has eigenvalues equal to the polynomial's roots.

### cloud mode combinatorics

Given $N$ base coefficients and polynomial degree $M$, cloud mode generates all $N^{M+1}$ possible polynomials by placing each base coefficient at each position. Each combination index is treated as a base-$N$ number where each digit selects which base coefficient fills that position.

It would be prohibitively computationally expensive to perform root-finding on tens of thousands of polynomials every frame during drag operations. Several optimizations make this tractable:

#### incremental solving

The observation is that the roots have to morph continuously. When a base coefficient moves slightly, most roots also move slightly. Instead of full re-solving, we can:

1. Estimates new root positions using first-order approximation (implicit differentiation of $P(r) = 0$)
2. Uses only Stage 3 of Jenkins-Traub to refine from these estimates
3. Falls back to full solve if refinement fails to converge

This exploits temporal coherence—during a drag, we have excellent initial guesses from the previous frame.

#### staggered full solves

Even with incremental solving, numerical drift accumulates. The solver maintains a `since_last_update` counter per combination, triggering periodic full re-solves staggered across frames. This distributes the computational load evenly rather than causing frame hitches.

#### GPU rendering (OpenCL)

When OpenCL is available, rendering runs entirely on the GPU via four kernels:

1. **`evaluate_polynomial`** — Evaluates $P(z)$ at each pixel using Horner's method. Each work item handles one pixel.

2. **`compute_max_magnitude`** / **`reduce_max_final`** — Two-stage parallel reduction to find the global maximum magnitude for normalization.

3. **`render_frame`** — Color mapping kernel. Reads the evaluation results, computes HSL from magnitude/argument, applies gamma correction, and outputs RGBA pixels.

For cloud mode, these kernels handle the point rendering:

4. **`render_point_cloud`** — Each work item handles one polynomial's roots. Uses atomic operations to accumulate hue vectors (as cos/sin components) into a per-pixel buffer to average hues when multiple points overlap.

5. **`render_point_cloud_hue`** — Converts accumulated hue vectors to final pixel colors and outputs to a RGBA buffer.

The OpenCL source (`render.cl`) is embedded directly into the binary at compile time.

#### CPU fallback

Without OpenCL, rendering uses OpenMP-parallelized CPU evaluation. The pixel buffer is partitioned across threads with static scheduling. Cloud mode uses per-thread hue accumulation buffers to avoid contention.

### cross-platform notes

The code jumps through some hoops for MSVC compatibility (seriously why...):

- No `_Complex` keyword—uses explicit struct-based complex numbers with function wrappers (`cxadd`, `cxmul`, etc.)
- Long double precision is optional (`COMPAT_COMPLEX_DOUBLE_ONLY`)—MSVC's `long double` is just `double` anyway

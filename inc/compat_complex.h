#ifndef COMPAT_COMPLEX_H
#define COMPAT_COMPLEX_H

#include <math.h>

#if defined(_MSC_VER) && !defined(__clang__)

// msvc implementation
// uses struct-based complex types with explicit function calls

#include <complex.h>

typedef _Dcomplex cxdouble;
typedef _Fcomplex cxfloat;
typedef _Dcomplex cxldouble;  // msvc: long double == double anyway

// construction
#define CX(re, im)   _Cbuild((double)(re), (double)(im))
#define CXF(re, im)  _FCbuild((float)(re), (float)(im))
#define CXL(re, im)  _Cbuild((double)(re), (double)(im))

#define CX_I   _Cbuild(0.0, 1.0)
#define CXF_I  _FCbuild(0.0f, 1.0f)
#define CXL_I  _Cbuild(0.0, 1.0)

// double accessors
static inline double cxreal(cxdouble z) { return creal(z); }
static inline double cximag(cxdouble z) { return cimag(z); }
static inline double cxabs(cxdouble z)  { return cabs(z); }
static inline double cxarg(cxdouble z)  { return carg(z); }

// double arithmetic
static inline cxdouble cxadd(cxdouble a, cxdouble b) {
    return _Cbuild(creal(a) + creal(b), cimag(a) + cimag(b));
}

static inline cxdouble cxsub(cxdouble a, cxdouble b) {
    return _Cbuild(creal(a) - creal(b), cimag(a) - cimag(b));
}

static inline cxdouble cxmul(cxdouble a, cxdouble b) {
    double ar = creal(a), ai = cimag(a);
    double br = creal(b), bi = cimag(b);
    return _Cbuild(ar*br - ai*bi, ar*bi + ai*br);
}

static inline cxdouble cxdiv(cxdouble a, cxdouble b) {
    double ar = creal(a), ai = cimag(a);
    double br = creal(b), bi = cimag(b);
    double denom = br*br + bi*bi;
    return _Cbuild((ar*br + ai*bi) / denom, (ai*br - ar*bi) / denom);
}

static inline cxdouble cxneg(cxdouble z) {
    return _Cbuild(-creal(z), -cimag(z));
}

static inline cxdouble cxconj(cxdouble z) {
    return _Cbuild(creal(z), -cimag(z));
}

static inline cxdouble cxscale(cxdouble z, double s) {
    return _Cbuild(creal(z) * s, cimag(z) * s);
}

static inline cxdouble cxscalediv(cxdouble z, double s) {
    return _Cbuild(creal(z) / s, cimag(z) / s);
}

// double math functions
static inline cxdouble cxsqrt(cxdouble z) { return csqrt(z); }
static inline cxdouble cxexp(cxdouble z)  { return cexp(z); }
static inline cxdouble cxlog(cxdouble z)  { return clog(z); }
static inline cxdouble cxsin(cxdouble z)  { return csin(z); }
static inline cxdouble cxcos(cxdouble z)  { return ccos(z); }
static inline cxdouble cxtan(cxdouble z)  { return ctan(z); }
static inline cxdouble cxpow(cxdouble a, cxdouble b) { return cpow(a, b); }

// long double accessors (maps to double on msvc)
static inline double cxreall(cxldouble z) { return creal(z); }
static inline double cximagl(cxldouble z) { return cimag(z); }
static inline double cxabsl(cxldouble z)  { return cabs(z); }
static inline double cxargl(cxldouble z)  { return carg(z); }

// long double arithmetic
static inline cxldouble cxaddl(cxldouble a, cxldouble b) { return cxadd(a, b); }
static inline cxldouble cxsubl(cxldouble a, cxldouble b) { return cxsub(a, b); }
static inline cxldouble cxmull(cxldouble a, cxldouble b) { return cxmul(a, b); }
static inline cxldouble cxdivl(cxldouble a, cxldouble b) { return cxdiv(a, b); }
static inline cxldouble cxnegl(cxldouble z) { return cxneg(z); }
static inline cxldouble cxconjl(cxldouble z) { return cxconj(z); }
static inline cxldouble cxscalel(cxldouble z, long double s) { return cxscale(z, (double)s); }
static inline cxldouble cxscaledivl(cxldouble z, long double s) { return cxscalediv(z, (double)s); }

static inline cxldouble cxsqrtl(cxldouble z) { return csqrt(z); }
static inline cxldouble cxexpl(cxldouble z)  { return cexp(z); }
static inline cxldouble cxlogl(cxldouble z)  { return clog(z); }
static inline cxldouble cxpowl(cxldouble a, cxldouble b) { return cpow(a, b); }

// conversions (no-op on msvc, same type)
static inline cxdouble cxl_to_cx(cxldouble z) { return z; }
static inline cxldouble cx_to_cxl(cxdouble z) { return z; }

// float accessors
static inline float cxrealf(cxfloat z) { return crealf(z); }
static inline float cximagf(cxfloat z) { return cimagf(z); }
static inline float cxabsf(cxfloat z)  { return cabsf(z); }
static inline float cxargf(cxfloat z)  { return cargf(z); }

// float arithmetic
static inline cxfloat cxaddf(cxfloat a, cxfloat b) {
    return _FCbuild(crealf(a) + crealf(b), cimagf(a) + cimagf(b));
}

static inline cxfloat cxsubf(cxfloat a, cxfloat b) {
    return _FCbuild(crealf(a) - crealf(b), cimagf(a) - cimagf(b));
}

static inline cxfloat cxmulf(cxfloat a, cxfloat b) {
    float ar = crealf(a), ai = cimagf(a);
    float br = crealf(b), bi = cimagf(b);
    return _FCbuild(ar*br - ai*bi, ar*bi + ai*br);
}

static inline cxfloat cxdivf(cxfloat a, cxfloat b) {
    float ar = crealf(a), ai = cimagf(a);
    float br = crealf(b), bi = cimagf(b);
    float denom = br*br + bi*bi;
    return _FCbuild((ar*br + ai*bi) / denom, (ai*br - ar*bi) / denom);
}

#else

// gcc/clang implementation (c99+)
// uses native _Complex types with operators

#include <complex.h>

typedef double _Complex cxdouble;
typedef float _Complex cxfloat;

#ifdef COMPAT_COMPLEX_DOUBLE_ONLY
typedef double _Complex cxldouble;
#else
typedef long double _Complex cxldouble;
#endif

// construction
#if defined(CMPLX)
#define CX(re, im)   CMPLX(re, im)
#define CXF(re, im)  CMPLXF(re, im)
#ifdef COMPAT_COMPLEX_DOUBLE_ONLY
#define CXL(re, im)  CMPLX(re, im)
#else
#define CXL(re, im)  CMPLXL(re, im)
#endif
#else
#define CX(re, im)   ((double)(re) + (double)(im) * I)
#define CXF(re, im)  ((float)(re) + (float)(im) * I)
#ifdef COMPAT_COMPLEX_DOUBLE_ONLY
#define CXL(re, im)  ((double)(re) + (double)(im) * I)
#else
#define CXL(re, im)  ((long double)(re) + (long double)(im) * I)
#endif
#endif

#define CX_I   I
#define CXF_I  I
#define CXL_I  I

// double accessors
static inline double cxreal(cxdouble z) { return creal(z); }
static inline double cximag(cxdouble z) { return cimag(z); }
static inline double cxabs(cxdouble z)  { return cabs(z); }
static inline double cxarg(cxdouble z)  { return carg(z); }

// double arithmetic
static inline cxdouble cxadd(cxdouble a, cxdouble b) { return a + b; }
static inline cxdouble cxsub(cxdouble a, cxdouble b) { return a - b; }
static inline cxdouble cxmul(cxdouble a, cxdouble b) { return a * b; }
static inline cxdouble cxdiv(cxdouble a, cxdouble b) { return a / b; }
static inline cxdouble cxneg(cxdouble z) { return -z; }
static inline cxdouble cxconj(cxdouble z) { return conj(z); }
static inline cxdouble cxscale(cxdouble z, double s) { return z * s; }
static inline cxdouble cxscalediv(cxdouble z, double s) { return z / s; }

static inline cxdouble cxsqrt(cxdouble z) { return csqrt(z); }
static inline cxdouble cxexp(cxdouble z)  { return cexp(z); }
static inline cxdouble cxlog(cxdouble z)  { return clog(z); }
static inline cxdouble cxsin(cxdouble z)  { return csin(z); }
static inline cxdouble cxcos(cxdouble z)  { return ccos(z); }
static inline cxdouble cxtan(cxdouble z)  { return ctan(z); }
static inline cxdouble cxpow(cxdouble a, cxdouble b) { return cpow(a, b); }

// long double accessors
#ifdef COMPAT_COMPLEX_DOUBLE_ONLY
static inline double cxreall(cxldouble z) { return creal(z); }
static inline double cximagl(cxldouble z) { return cimag(z); }
static inline double cxabsl(cxldouble z)  { return cabs(z); }
static inline double cxargl(cxldouble z)  { return carg(z); }
#else
static inline long double cxreall(cxldouble z) { return creall(z); }
static inline long double cximagl(cxldouble z) { return cimagl(z); }
static inline long double cxabsl(cxldouble z)  { return cabsl(z); }
static inline long double cxargl(cxldouble z)  { return cargl(z); }
#endif

// long double arithmetic
static inline cxldouble cxaddl(cxldouble a, cxldouble b) { return a + b; }
static inline cxldouble cxsubl(cxldouble a, cxldouble b) { return a - b; }
static inline cxldouble cxmull(cxldouble a, cxldouble b) { return a * b; }
static inline cxldouble cxdivl(cxldouble a, cxldouble b) { return a / b; }
static inline cxldouble cxnegl(cxldouble z) { return -z; }
static inline cxldouble cxscalel(cxldouble z, long double s) { return z * s; }
static inline cxldouble cxscaledivl(cxldouble z, long double s) { return z / s; }

#ifdef COMPAT_COMPLEX_DOUBLE_ONLY
static inline cxldouble cxconjl(cxldouble z) { return conj(z); }
static inline cxldouble cxsqrtl(cxldouble z) { return csqrt(z); }
static inline cxldouble cxexpl(cxldouble z)  { return cexp(z); }
static inline cxldouble cxlogl(cxldouble z)  { return clog(z); }
static inline cxldouble cxpowl(cxldouble a, cxldouble b) { return cpow(a, b); }
#else
static inline cxldouble cxconjl(cxldouble z) { return conjl(z); }
static inline cxldouble cxsqrtl(cxldouble z) { return csqrtl(z); }
static inline cxldouble cxexpl(cxldouble z)  { return cexpl(z); }
static inline cxldouble cxlogl(cxldouble z)  { return clogl(z); }
static inline cxldouble cxpowl(cxldouble a, cxldouble b) { return cpowl(a, b); }
#endif

// conversions
static inline cxdouble cxl_to_cx(cxldouble z) { return (cxdouble)z; }
static inline cxldouble cx_to_cxl(cxdouble z) { return (cxldouble)z; }

// float accessors
static inline float cxrealf(cxfloat z) { return crealf(z); }
static inline float cximagf(cxfloat z) { return cimagf(z); }
static inline float cxabsf(cxfloat z)  { return cabsf(z); }
static inline float cxargf(cxfloat z)  { return cargf(z); }

// float arithmetic
static inline cxfloat cxaddf(cxfloat a, cxfloat b) { return a + b; }
static inline cxfloat cxsubf(cxfloat a, cxfloat b) { return a - b; }
static inline cxfloat cxmulf(cxfloat a, cxfloat b) { return a * b; }
static inline cxfloat cxdivf(cxfloat a, cxfloat b) { return a / b; }

#endif // _MSC_VER

#endif // COMPAT_COMPLEX_H
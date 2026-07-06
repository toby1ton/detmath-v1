"""detmath — bit-identical transcendental functions for every platform.

The problem: math.sin, math.exp, math.log call your platform's libm, and
glibc, musl, MSVC and Apple's libm all disagree in the last bits. That breaks
lockstep simulation, multiplayer determinism, and reproducible science.

The fix: implement them in pure Python using only operations that IEEE 754
*guarantees* are correctly rounded and therefore identical everywhere:
float + - * /, math.frexp/ldexp (exact exponent manipulation), and Python's
exact big integers. Pure Python also means no compiler: no FMA contraction,
no reassociation, a single defined evaluation order.

Result: every function here returns the exact same 64 bits on every machine
CPython runs on. Accuracy is within ~2 ULP of the true result (measured
against mpmath in test_detmath.py).

Functions: sin, cos, exp, log. Domain/error behavior mirrors the math module.

License: MIT.
"""

import math

# ---------------------------------------------------------------------------
# Constants.
# ln2 split (fdlibm-style: hi part has trailing zero bits so k*LN2_HI is
# EXACT for |k| <= 2^20 — this keeps argument reduction error-free).
# ---------------------------------------------------------------------------

_LN2_HI = float.fromhex("0x1.62e42feep-1")        # 33 significant bits
_LN2_LO = float.fromhex("0x1.a39ef35793c76p-33")  # ln2 - LN2_HI
_INV_LN2 = float.fromhex("0x1.71547652b82fep+0")
_PIO2_HI = float.fromhex("0x1.921fb54442d18p+0")
_PIO2_LO = float.fromhex("0x1.1a62633145c07p-54")
_PIO4 = float.fromhex("0x1.921fb54442d18p-1")

# floor(2/pi * 2^1200): enough bits to reduce any double exactly.
_TWO_OVER_PI_1200 = 10961624472040172306819695974506753824149736177290862121133802293415128153967426855786600179383830818005251620686212276383741462382840646446665240090662947422797946208107084433317427984328678325390393467973966095626422319580541169421538667751778104036224809616456980795286610787367809947266181514671532645040128653394793474164295916360012309370980042855028292731


# ---------------------------------------------------------------------------
# Error-free float transforms (classic Dekker/Knuth building blocks).
# Every operation below is a plain IEEE + - * : deterministic by construction.
# ---------------------------------------------------------------------------

def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _split(a):
    t = a * 134217729.0            # 2^27 + 1
    hi = t - (t - a)
    return hi, a - hi


def _two_prod(a, b):
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


# ---------------------------------------------------------------------------
# Exact argument reduction: x mod pi/2 via big-integer multiply.
# This is the step platform libms get inconsistently wrong for large x.
# ---------------------------------------------------------------------------

def _reduce_pio2(x):
    """|x| > pi/4. Returns (quadrant n mod 4, r_hi, r_lo) with
    x = n*(pi/2) + r and |r| <= pi/4, r exact to ~2^-106."""
    m, e = math.frexp(x)           # x = m * 2^e, 0.5 <= m < 1, exact
    mi = int(math.ldexp(m, 53))    # exact 53-bit integer mantissa
    e -= 53
    shift = 1200 - e               # e <= 971 for finite doubles => shift >= 229
    prod = mi * _TWO_OVER_PI_1200  # = x*(2/pi) * 2^shift, exact
    n = prod >> shift
    frac = prod - (n << shift)     # fractional part of x*(2/pi), exact
    if frac >= (1 << (shift - 1)): # round to nearest quadrant
        n += 1
        frac -= 1 << shift         # now -0.5 <= frac*2^-shift < 0.5
    # Keep top 120 bits of the fraction (truncation error < 2^-119).
    drop = shift - 120
    if frac >= 0:
        f_int = frac >> drop
    else:
        f_int = -((-frac) >> drop)
    # f = f_int * 2^-120 as a double-double, exactly.
    fh = math.ldexp(float(f_int), -120)          # float() rounds: deterministic
    rem = f_int - int(math.ldexp(fh, 120))        # exact residual integer
    fl = math.ldexp(float(rem), -120)
    # r = f * (pi/2) in double-double.
    rh, rl = _two_prod(fh, _PIO2_HI)
    rl += fh * _PIO2_LO + fl * _PIO2_HI
    s, err = _two_sum(rh, rl)
    return n & 3, s, err


# ---------------------------------------------------------------------------
# Polynomial kernels on |r| <= pi/4 (Taylor, truncation < 0.25 ULP).
# ---------------------------------------------------------------------------

def _ksin(x, y):
    """sin(x + y), |x| <= pi/4, |y| tiny."""
    z = x * x
    p = z * (-1.0 / 6.0 + z * (1.0 / 120.0 + z * (-1.0 / 5040.0
        + z * (1.0 / 362880.0 + z * (-1.0 / 39916800.0
        + z * (1.0 / 6227020800.0 + z * (-1.0 / 1307674368000.0)))))))
    return x + (y + x * p)


def _kcos(x, y):
    """cos(x + y), |x| <= pi/4, |y| tiny."""
    z = x * x
    p = z * z * (1.0 / 24.0 + z * (-1.0 / 720.0 + z * (1.0 / 40320.0
        + z * (-1.0 / 3628800.0 + z * (1.0 / 479001600.0
        + z * (-1.0 / 87178291200.0 + z * (1.0 / 20922789888000.0)))))))
    hz = 0.5 * z
    w = 1.0 - hz
    return w + (((1.0 - w) - hz) + (p - x * y))


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def sin(x):
    x = float(x)
    if x != x:                     # nan
        return x
    if x in (float("inf"), float("-inf")):
        raise ValueError("math domain error")
    ax = -x if x < 0.0 else x
    if ax <= _PIO4:
        return _ksin(x, 0.0)
    n, rh, rl = _reduce_pio2(ax)
    if n == 0:
        v = _ksin(rh, rl)
    elif n == 1:
        v = _kcos(rh, rl)
    elif n == 2:
        v = -_ksin(rh, rl)
    else:
        v = -_kcos(rh, rl)
    return -v if x < 0.0 else v


def cos(x):
    x = float(x)
    if x != x:
        return x
    if x in (float("inf"), float("-inf")):
        raise ValueError("math domain error")
    ax = -x if x < 0.0 else x
    if ax <= _PIO4:
        return _kcos(ax, 0.0)
    n, rh, rl = _reduce_pio2(ax)
    if n == 0:
        return _kcos(rh, rl)
    if n == 1:
        return -_ksin(rh, rl)
    if n == 2:
        return -_kcos(rh, rl)
    return _ksin(rh, rl)


def exp(x):
    x = float(x)
    if x != x:
        return x
    if x > 709.782712893384:
        return float("inf")
    if x < -745.1332191019412:
        return 0.0
    k = math.floor(x * _INV_LN2 + 0.5)
    hi = x - k * _LN2_HI           # exact: LN2_HI has 33 sig bits, |k| < 2^11
    lo = k * _LN2_LO
    r = hi - lo
    c = (hi - r) - lo              # residual of the reduction
    # exp(r), |r| <= ln2/2, Taylor through r^13 (truncation < 0.05 ULP)
    p = 1.0
    for d in (13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0):
        p = 1.0 + r * p / d
    e = 1.0 + r * p + c
    return math.ldexp(e, k)


def log(x):
    x = float(x)
    if x != x:
        return x
    if x <= 0.0:
        raise ValueError("math domain error")
    if x == float("inf"):
        return x
    m, e = math.frexp(x)           # m in [0.5, 1)
    if m < 0.7071067811865476:
        m *= 2.0
        e -= 1
    s = (m - 1.0) / (m + 1.0)      # m-1 is EXACT (Sterbenz); |s| <= 0.1716
    z = s * s
    # 2*atanh(s) = 2s + s*poly(z); series through s^19 (truncation < 1e-17)
    p = z * (2.0 / 3.0 + z * (2.0 / 5.0 + z * (2.0 / 7.0 + z * (2.0 / 9.0
        + z * (2.0 / 11.0 + z * (2.0 / 13.0 + z * (2.0 / 15.0
        + z * (2.0 / 17.0 + z * (2.0 / 19.0)))))))))
    t = 2.0 * s
    hi, lo = _two_sum(e * _LN2_HI, t)   # e*LN2_HI exact (33-bit constant)
    return hi + (lo + (s * p + e * _LN2_LO))

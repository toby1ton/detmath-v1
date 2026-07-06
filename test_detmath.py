"""detmath tests: ULP accuracy vs mpmath ground truth + determinism fingerprint.

Run: python3 test_detmath.py
"""
import math
import random
import struct
import hashlib
from mpmath import mp
import detmath

mp.prec = 120
random.seed(7)
failures = []


def ulp_diff(got, x, mp_fn):
    """Bit distance between got and the correctly-rounded true value."""
    want = float(mp_fn(mp.mpf(x)))
    if got == want:
        return 0
    gi = struct.unpack("<q", struct.pack("<d", got))[0]
    wi = struct.unpack("<q", struct.pack("<d", want))[0]
    if gi < 0:
        gi = -9223372036854775808 - gi
    if wi < 0:
        wi = -9223372036854775808 - wi
    return abs(gi - wi)


def sweep(name, fn, mp_fn, samples, tol):
    worst, worst_x = 0, None
    for x in samples:
        try:
            d = ulp_diff(fn(x), x, mp_fn)
        except (ValueError, OverflowError):
            continue
        if d > worst:
            worst, worst_x = d, x
    status = "PASS" if worst <= tol else "FAIL"
    if status == "FAIL":
        failures.append(name)
    print("%-26s worst %3d ULP at x=%-16.9g (tol %d)  %s"
          % (name, worst, worst_x if worst_x is not None else 0.0, tol, status))


def r(lo, hi, n=3000):
    return [random.uniform(lo, hi) for _ in range(n)]


def rlog(lo_exp, hi_exp, n=2000):
    return [math.ldexp(random.uniform(1, 2), random.randint(lo_exp, hi_exp))
            * random.choice((1, -1)) for _ in range(n)]


# --- accuracy ---------------------------------------------------------------
sweep("sin small", detmath.sin, mp.sin, r(-0.78, 0.78), 1)
sweep("sin moderate", detmath.sin, mp.sin, r(-30, 30), 2)
sweep("sin large", detmath.sin, mp.sin, r(-1e6, 1e6), 2)
sweep("sin huge (reduction)", detmath.sin, mp.sin,
      [abs(v) for v in rlog(60, 1020)], 2)
sweep("cos moderate", detmath.cos, mp.cos, r(-30, 30), 2)
sweep("cos near pi/2", detmath.cos, mp.cos,
      [math.pi / 2 + d for d in r(-1e-6, 1e-6)], 2)
sweep("exp mid", detmath.exp, mp.exp, r(-10, 10), 2)
sweep("exp full range", detmath.exp, mp.exp, r(-700, 700), 2)
sweep("log wide", detmath.log, mp.log,
      [abs(v) for v in rlog(-1000, 1000)], 2)
sweep("log near 1", detmath.log, mp.log, r(0.9, 1.1), 2)

# --- hard cases -------------------------------------------------------------
hard = [1e300, 1e22, 6381956970095103.0 * 2.0 ** 797,  # famous worst case
        math.pi, 2 * math.pi, 355.0 / 113.0]
sweep("sin worst-known args", detmath.sin, mp.sin, hard, 2)

# --- semantics match math module --------------------------------------------
for bad_fn, bad_x in ((detmath.sin, float("inf")), (detmath.log, -1.0),
                      (detmath.log, 0.0)):
    try:
        bad_fn(bad_x)
        failures.append("domain error missing")
        print("domain error missing for", bad_fn.__name__, bad_x)
    except ValueError:
        pass
assert detmath.exp(float("-inf")) == 0.0
assert detmath.exp(800.0) == float("inf")
assert detmath.exp(float("nan")) != detmath.exp(float("nan"))
print("%-26s %s" % ("special values", "PASS"))

# --- determinism fingerprint -------------------------------------------------
# Same SHA256 on every platform <=> bit-identical outputs. Run this on your
# Windows/Mac/Linux/ARM machines and compare.
h = hashlib.sha256()
rng = random.Random(123456)
for _ in range(20000):
    x = math.ldexp(rng.uniform(1, 2), rng.randint(-60, 60)) * rng.choice((1, -1))
    h.update(struct.pack("<d", detmath.sin(x)))
    h.update(struct.pack("<d", detmath.cos(x)))
    h.update(struct.pack("<d", detmath.exp(x % 100 - 50)))
    h.update(struct.pack("<d", detmath.log(abs(x))))
print("determinism fingerprint:  ", h.hexdigest())

print()
if failures:
    print("FAILURES:", ", ".join(failures))
    raise SystemExit(1)
print("ALL PASS")

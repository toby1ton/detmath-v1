# detmath — bit-identical transcendental math on every platform

Pure-Python `sin`, `cos`, `exp`, `log` that return the **exact same 64 bits
on every machine** — Windows, macOS, Linux, x86, ARM. Zero dependencies,
single file, MIT.

## The problem

Python's `math.sin`, `math.exp`, etc. call your platform's C math library,
and glibc, musl, MSVC, and Apple's libm all disagree in the final bits.
Basic float ops (`+ - * /`) are IEEE-guaranteed identical everywhere — but
the transcendentals are not. This silently breaks:

- **Lockstep multiplayer / replays** — simulations desync across OSes
- **Reproducible science** — "same code, same inputs, different results"
- **Distributed simulation** — nodes on different hardware drift apart

## The fix

`detmath` implements the transcendentals using *only* operations IEEE 754
guarantees are correctly rounded (float `+ - * /`, exact `frexp`/`ldexp`)
plus Python's exact big integers. Pure Python also means no compiler — no
FMA contraction, no reassociation, one defined evaluation order. Determinism
by construction.

The hard part — reducing huge arguments mod pi/2 (where platform libms
disagree most) — is done with an exact 1200-bit integer multiply, so even
`sin(1e300)` is reduced without any error.

## Accuracy (measured against 120-bit mpmath ground truth)

| Function | Range tested | Worst error |
|---|---|---|
| sin | full double range, incl. 1e300+ | 1 ULP |
| cos | ±30 and near pi/2 | 1 ULP |
| exp | ±700 | 1 ULP |
| log | 2^-1000 to 2^1000 | 2 ULP |

The known worst-case argument for sin reduction (6381956970095103·2^797)
evaluates exactly correctly. Domain errors and special values (nan, inf,
overflow to inf, underflow to 0) mirror the `math` module.

## Verify determinism on YOUR machines

```
python3 test_detmath.py
```

The last line prints a SHA-256 fingerprint over 80,000 function outputs:

```
determinism fingerprint: 321fc632953e1db0cb6da61c6539e0f38d6b25892fa789a8e01c3d63e0970f0b
```

Run it on any two machines. Same hash = bit-identical behavior, proven.
(Reference hash above from CPython 3.12, x86-64 Linux. If you find a
platform that produces a different hash, please open an issue — that is
exactly the bug report this project exists for.)

## Usage

```python
import detmath
detmath.sin(1e300)   # same bits everywhere
detmath.exp(2.5)
```

Drop-in for the math-module functions it covers. It is slower than libm
(pure Python), so use it where determinism matters — game logic ticks,
simulation state — not in throwaway rendering math.

## Caveats, stated honestly

- CPython only. PyPy/GraalPy likely fine (same float semantics) but unverified.
- Assumes IEEE 754 binary64 hardware floats — true on every mainstream
  platform; exotic targets (pre-SSE2 x87 builds) are out of scope.
- v1 covers sin/cos/exp/log. tan, atan2, pow, sqrt-of-negative handling
  for complex use are planned next.

## License

MIT.

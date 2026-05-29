# Monoped Jumping Controller — Implementation Notes

## Reference Paper
**"A Co-Design Framework for Energy-Aware Monoped Jumping with Detailed Actuator Modeling"**
Singh, Mishra, Kapa, Joshi, Kolathaya — arXiv:2510.05923, Oct 2025

---

## What Was Implemented

### File: `jump.py`

A virtual spring-damper jump controller for the MuJoCo monoped that directly follows the control architecture described in **Section II-B** of the paper.

---

## Algorithm Overview

The paper models the monoped leg as a **1R–1P mechanism** (one revolute hip joint + one prismatic leg-extension joint) controlled by a virtual spring-damper. Our monoped has a **2R leg** (hip + knee), so the virtual forces are mapped to joint torques via the **Jacobian transpose**.

### 1. Virtual Leg Spring-Damper (Paper Eq. 1)

```
F_l = K * (L0 - l) - C * l_dot
```

| Symbol | Meaning | Value Used |
|--------|---------|------------|
| `K` | Spring stiffness (N/m) | 1000 |
| `C` | Damping coefficient (N·s/m) | 15 |
| `L0` | Natural leg length (m) — near full extension | 0.38 |
| `l` | Current leg length (m) = distance hip→foot | from FK |
| `l_dot` | Rate of change of leg length (m/s) | from Jacobian |

When the leg is compressed (`l < L0`), `F_l > 0` — the spring pushes the foot away from the pelvis, generating an upward reaction on the body.

### 2. Virtual Torsional Spring (Paper Eq. 2)

```
τ_l = T * (α0 - α)
```

| Symbol | Meaning | Value Used |
|--------|---------|------------|
| `T` | Torsional spring constant (N·m/rad) | 25 |
| `α0` | Resting leg orientation (rad) | 0.0 (vertical) |
| `α` | Current leg angle from downward vertical | from FK |

The torsional spring resists lateral lean, keeping the leg vertical and providing stability.

### 3. Force Decomposition in World Frame

The scalar spring forces are resolved into Cartesian components in the pelvis frame:

```
Fx = F_l*(xf/l) + (τ_l/l)*(-zf/l)    ← along-leg + perpendicular
Fz = F_l*(zf/l) + (τ_l/l)*(xf/l)
```

This is the vector equivalent of **Paper Eqs. 3–4** (without the `−mg` term, since MuJoCo applies gravity separately).

### 4. Jacobian-Transpose Mapping (Paper §II-B)

```
τ_joints = J(θ1, θ2)ᵀ · [Fx, Fz]ᵀ
```

The 2×2 body Jacobian for a 2R leg (upper link L1, lower link L2):

```
J = [ L1·cos(θ1)+L2·cos(θ1+θ2),   L2·cos(θ1+θ2) ]
    [ L1·sin(θ1)+L2·sin(θ1+θ2),   L2·sin(θ1+θ2) ]
```

Motor commands are normalised: `ctrl = clip(τ / gear, −1, 1)`.

---

## State Machine

```
SETTLE (0 → 0.15 s)
  └─ PD control holds the crouched pose while physics settle
  └─ Transitions to STANCE at t = 0.15 s

STANCE (foot z < 0.018 m)
  └─ Virtual spring-damper torques applied (Eqs. 1–2 + Jacobian)
  └─ Spring compressed → explosive push-off
  └─ Transitions to FLIGHT when foot leaves ground

FLIGHT (foot z ≥ 0.018 m)
  └─ Zero torques (matches paper: "torques applied only during ground contact")
  └─ Transitions back to STANCE on landing → repeated jumping
```

---

## Parameters — Paper vs. Implementation

| Parameter | Paper Case-C | This Implementation | Reason for Difference |
|-----------|-------------|---------------------|----------------------|
| `l1` (m) | 0.49 | 0.20 | Fixed model geometry |
| `l2` (m) | 0.36 | 0.20 | Fixed model geometry |
| `K` (N/m) | 7.9 | 1000 | Scaled up for shorter leg |
| `C` (N·s/m) | 2.2 | 15 | Tuned for stability |
| `T` (N·m/rad) | 8.4 | 25 | Scaled for shorter leg |
| `L0` (m) | ~0.85 (est.) | 0.38 | Near max extension 0.40 m |
| `H_init` (m) | 0.50 | 0.24 | Same % compression (60 %) |
| `g_h` | 4:1 | 55 N·m | Baked into XML gear |
| `g_k` | 6:1 | 45 N·m | Baked into XML gear |

The paper's robot has a leg ~2× longer than ours. Spring force scales as `K × Δl`, so to achieve equivalent ground reaction forces with half the compression distance, `K` must be ~6–10× larger. `T` scales similarly.

### Estimated Performance

```
Spring PE at launch ≈ 0.5 × K × (L0 − H_init)²
                    = 0.5 × 1000 × (0.38 − 0.24)²
                    = 9.8 J

Jump height (50 % efficiency) ≈ PE / (M × g)
                               ≈ 9.8 / (1.5 × 9.81) ≈ 0.33 m
```

---

## XML Change (`monoped.xml`)

The `slide_z` joint upper limit was increased from `0.02 m` to `2.00 m`:

```xml
<!-- Before -->
<joint name="slide_z" … range="-0.34 0.02" …/>

<!-- After -->
<joint name="slide_z" … range="-0.34 2.00" …/>
```

This is required to allow the pelvis to rise during flight. The sit-stand simulation is unaffected (it never exceeds 0.36 m pelvis height).

---

## Live Plot (mirrors Paper Fig. 7)

Three-panel real-time window:
1. **Base height z (m) and velocity ż (m/s)** vs time
2. **Hip (θ1) and knee (θ2) angles and angular velocities** vs time
3. **Hip and knee control torques** (N·m) vs time

---

## How to Run

```bash
cd /home/yaghisvar/mujoco_sim
/home/yaghisvar/mujoco_env/bin/python jump.py
```

Two windows open:
- **MuJoCo viewer** — 3-D simulation
- **Matplotlib window** — live telemetry plots

Expected console output:
```
  t=0.150s  SETTLE → STANCE  (spring active)
  t=0.38s   STANCE → FLIGHT  pelvis=0.41m  vz=+2.3m/s
  t=0.62s   FLIGHT → STANCE  (peak Δh=0.27m)
  ...
```

The robot repeats the jump cycle continuously.

---

## What Was NOT Implemented (Paper's Full Framework)

The paper describes a **3-stage co-design optimisation**:

| Stage | Paper | This Work |
|-------|-------|-----------|
| Stage 1: Actuator optimisation (ISSPG/ESSPG gear parameters) | ✓ | Not implemented |
| Stage 2: CMA-ES co-design (gear ratios + link lengths + K,C,T) | ✓ | Parameters fixed manually |
| Stage 3: Parametric CAD model generation | ✓ | Not implemented |

What **is** implemented: the core **control architecture** (Section II-B) which is the inner loop used by all three optimisation stages. This is sufficient to demonstrate the jumping behaviour in simulation.

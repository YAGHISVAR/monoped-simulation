"""
Monoped — Virtual Spring-Damper Jump Controller
================================================
Reference: Singh et al., arXiv:2510.05923, 2025  (Section II-B)

Control law:
  F_l  = K*(L0 - l) - C*l_dot       virtual leg spring-damper   (Eq. 1)
  τ_l  = T*(α0 - α)                  virtual torsional spring    (Eq. 2)
  τ_jt = J(θ1,θ2)ᵀ · F_world        Jacobian-transpose map      (§II-B)

State machine:
  CROUCH  — PD sweeps pelvis from standing down to H_INIT over 0.8 s
  HOLD    — PD holds crouched; waits for confirmed ground contact
  STANCE  — virtual spring active (foot on ground)
  FLIGHT  — zero torques (paper: "only during ground contact")
"""

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Robot geometry ─────────────────────────────────────────────
L1 = 0.20
L2 = 0.20

# ── Virtual spring-damper parameters ──────────────────────────
# Joint-space spring: drives θ1→0, θ2→0 (full extension).
# Avoids the Cartesian-spring Jacobian singularity at l=0.40 m.
# Maps to paper's K (leg spring) and C (damping) in joint space.
KJ = 280.0   # joint spring stiffness  (N·m/rad)
DJ = 3.0     # joint damping           (N·m·s/rad)
# Torsional spring T not needed: robot only moves vertically (α≡0).

# ── Motion profile ─────────────────────────────────────────────
H_STAND     = 0.36   # initial standing height (m)
H_INIT      = 0.22   # crouch height — gives L0−l ≈ 0.15 m compression
CROUCH_DUR  = 0.80   # seconds to lower from STAND to H_INIT
HOLD_DUR    = 0.40   # seconds to hold crouched and wait for contact

# ── Hardware limits ────────────────────────────────────────────
HIP_GEAR  = 55.0
KNEE_GEAR = 45.0

# ── Contact hysteresis ─────────────────────────────────────────
MIN_FLIGHT_DT = 0.06   # must be airborne ≥ 60 ms to enter FLIGHT

# ── Sensor indices ─────────────────────────────────────────────
S_HIP_P  = 3;  S_KNEE_P = 4
S_HIP_V  = 6;  S_KNEE_V = 7;  S_SLZ_V = 8
S_FOOT_Z = 11; S_PEL_Z  = 14

# ── Phase IDs ──────────────────────────────────────────────────
CROUCH, HOLD, STANCE, FLIGHT = 0, 1, 2, 3
PHASE_NAMES = ["CROUCH", "HOLD", "STANCE", "FLIGHT"]


# ─────────────────────────────────────────────────────────────
# Kinematics
# ─────────────────────────────────────────────────────────────

def ik_angles(h: float):
    h = np.clip(h, 0.06 + 1e-4, (L1 + L2) - 1e-4)
    hip = float(np.arccos(h / (L1 + L2)))
    return hip, -2.0 * hip


def jacobian(t1: float, t2: float) -> np.ndarray:
    c1, s1   = np.cos(t1),    np.sin(t1)
    c12, s12 = np.cos(t1+t2), np.sin(t1+t2)
    return np.array([[L1*c1 + L2*c12,  L2*c12],
                     [L1*s1 + L2*s12,  L2*s12]])


def spring_torques(t1: float, t2: float,
                   dt1: float, dt2: float):
    """
    Joint-space spring-damper (paper §II-B adapted for 2R leg).

    Drives both joints toward full extension (θ1→0, θ2→0) which
    lengthens the leg and launches the pelvis upward. This avoids
    the Cartesian-spring Jacobian singularity at l = L1+L2 = 0.40 m.

    In the paper's notation: KJ ≈ K (joint-space stiffness) and
    DJ ≈ C (joint-space damping).
    """
    tau_h = -KJ * t1 - DJ * dt1   # drives hip toward 0 (extend)
    tau_k = -KJ * t2 - DJ * dt2   # drives knee toward 0 (extend)
    return float(tau_h), float(tau_k)


# ─────────────────────────────────────────────────────────────
# Live plot — Fig. 7 style (3 panels)
# ─────────────────────────────────────────────────────────────
plt.ion()
fig = plt.figure(figsize=(10, 7), facecolor="#1e1e2e")
fig.canvas.manager.set_window_title("Monoped Jump — Live (Fig. 7 style)")

BG, FG, MED = "#1e1e2e", "#cdd6f4", "#a6adc8"
gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

for ax in (ax1, ax2, ax3):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#6c7086")
    ax.tick_params(colors=MED)
    ax.set_xlabel("Time (s)", color=MED, fontsize=9)

ax1.set_title("Base Height & Velocity",          color=FG, fontsize=11)
ax2.set_title("Hip & Knee Angles / Velocities",  color=FG, fontsize=11)
ax3.set_title("Control Torques",                 color=FG, fontsize=11)
ax1.set_ylabel("z (m)  /  ż (m/s)",     color=MED, fontsize=9)
ax2.set_ylabel("θ (rad)  /  ω (rad/s)", color=MED, fontsize=9)
ax3.set_ylabel("Torque (N·m)",           color=MED, fontsize=9)
ax3.axhline(0, color="#6c7086", lw=0.7, ls="--")

lz,    = ax1.plot([], [], color="#89b4fa", lw=1.8, label="z pelvis (m)")
lzdot, = ax1.plot([], [], color="#cba6f7", lw=1.4, ls="--", label="ż (m/s)")
lth1,  = ax2.plot([], [], color="#a6e3a1", lw=1.6, label="θ_hip (rad)")
lth2,  = ax2.plot([], [], color="#94e2d5", lw=1.6, label="θ_knee (rad)")
lom1,  = ax2.plot([], [], color="#f38ba8", lw=1.0, ls=":", label="ω_hip")
lom2,  = ax2.plot([], [], color="#fab387", lw=1.0, ls=":", label="ω_knee")
luh,   = ax3.plot([], [], color="#89b4fa", lw=1.6, label="τ_hip (N·m)")
luk,   = ax3.plot([], [], color="#f38ba8", lw=1.6, label="τ_knee (N·m)")

for ax, hs in [(ax1,[lz,lzdot]),(ax2,[lth1,lth2,lom1,lom2]),(ax3,[luh,luk])]:
    ax.legend(facecolor="#313244", edgecolor="#6c7086",
              labelcolor=FG, fontsize=8, loc="upper right", handles=hs)

fig.tight_layout(pad=1.5)
fig.canvas.draw(); plt.pause(0.001)

WINDOW = 6.0
bufs = {k: [] for k in ("t","z","zd","h1","h2","w1","w2","uh","uk")}

def _push(**kw):
    for k, v in kw.items():
        bufs[k].append(v)

def _redraw(t_now):
    arr  = {k: np.asarray(v) for k, v in bufs.items()}
    xmin = max(0.0, t_now - WINDOW)
    xmax = max(t_now, WINDOW)
    m    = arr["t"] >= xmin

    for ln, yk in [(lz,"z"),(lzdot,"zd"),(lth1,"h1"),(lth2,"h2"),
                   (lom1,"w1"),(lom2,"w2"),(luh,"uh"),(luk,"uk")]:
        ln.set_data(arr["t"][m], arr[yk][m])

    for ax, keys in [(ax1,["z","zd"]),(ax2,["h1","h2","w1","w2"]),(ax3,["uh","uk"])]:
        ax.set_xlim(xmin, xmax)
        if m.any():
            vals = np.concatenate([arr[k][m] for k in keys])
            pad  = max(0.05, (vals.max()-vals.min())*0.1)
            ax.set_ylim(vals.min()-pad, vals.max()+pad)

    fig.canvas.draw_idle()
    fig.canvas.flush_events()


# ─────────────────────────────────────────────────────────────
# Load model & initial state
# ─────────────────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path("monoped.xml")
data  = mujoco.MjData(model)

# Geom IDs for contact detection
foot_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "foot")
floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

def is_on_ground() -> bool:
    """True when the foot geom is in active contact with the floor."""
    for i in range(data.ncon):
        c = data.contact[i]
        if (c.geom1 == foot_id and c.geom2 == floor_id) or \
           (c.geom1 == floor_id and c.geom2 == foot_id):
            return True
    return False

# Initialise standing
hip_st, kn_st = ik_angles(H_STAND)
data.qpos[0]  = H_STAND - 0.40
data.qpos[1]  = hip_st
data.qpos[2]  = kn_st
mujoco.mj_forward(model, data)

hip_cr, kn_cr = ik_angles(H_INIT)

print("=" * 60)
print("  Monoped — Virtual Spring-Damper Jump Controller")
print(f"  KJ={KJ} N·m/rad   DJ={DJ} N·m·s/rad  (joint-space spring)")
print(f"  Crouch: {H_STAND}→{H_INIT} m over {CROUCH_DUR} s")
hip0_est, _ = ik_angles(H_INIT)
print(f"  Peak τ_knee ≈ {KJ*2*hip0_est:.1f} N·m  (clipped to ±{KNEE_GEAR})")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# Simulation loop
# ─────────────────────────────────────────────────────────────
phase         = CROUCH
last_off_gnd  = -999.0   # time when foot first left ground (this flight)
prev_on_gnd   = True
plot_last     = -1.0
log_last      = -1.0
peak_z        = H_STAND
PLOT_DT       = 0.04
phase_start   = 0.0

KP_SETTLE = 30.0
KD_SETTLE = 2.5

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth   = 90
    viewer.cam.elevation = -15
    viewer.cam.distance  = 2.4
    viewer.cam.lookat[:] = [0, 0, 0.40]

    while viewer.is_running():
        t      = data.time
        t1     = float(data.sensordata[S_HIP_P])
        t2     = float(data.sensordata[S_KNEE_P])
        dt1    = float(data.sensordata[S_HIP_V])
        dt2    = float(data.sensordata[S_KNEE_V])
        pel_z  = float(data.sensordata[S_PEL_Z])
        pel_vz = float(data.sensordata[S_SLZ_V])
        foot_z = float(data.sensordata[S_FOOT_Z])
        on_gnd = is_on_ground()

        if pel_z > peak_z:
            peak_z = pel_z
        if not on_gnd and prev_on_gnd:
            last_off_gnd = t
        prev_on_gnd = on_gnd

        # ── Phase transitions ─────────────────────────────
        if phase == CROUCH:
            if t - phase_start >= CROUCH_DUR:
                phase = HOLD
                phase_start = t
                print(f"  t={t:.2f}s  CROUCH→HOLD  pelvis={pel_z:.3f}m")

        elif phase == HOLD:
            if t - phase_start >= HOLD_DUR and on_gnd:
                phase = STANCE
                phase_start = t
                print(f"  t={t:.2f}s  HOLD→STANCE  pelvis={pel_z:.3f}m  "
                      f"KJ={KJ} N·m/rad")

        elif phase == STANCE:
            if not on_gnd and (t - last_off_gnd) >= MIN_FLIGHT_DT:
                phase = FLIGHT
                phase_start = t
                print(f"  t={t:.2f}s  STANCE→FLIGHT  "
                      f"pelvis={pel_z:.3f}m  vz={pel_vz:+.3f}m/s")

        elif phase == FLIGHT:
            if on_gnd:
                airborne = peak_z - H_INIT
                print(f"  t={t:.2f}s  FLIGHT→STANCE  "
                      f"peak Δz={airborne:.3f}m  landing={pel_z:.3f}m")
                phase = STANCE
                phase_start = t
                peak_z = pel_z

        # ── Control ──────────────────────────────────────
        if phase == CROUCH:
            # Linearly sweep target height from H_STAND to H_INIT
            frac = min((t - 0.0) / CROUCH_DUR, 1.0)
            h_tgt = H_STAND + frac * (H_INIT - H_STAND)
            hip_t, kn_t = ik_angles(h_tgt)
            tau_h = KP_SETTLE*(hip_t - t1) - KD_SETTLE*dt1
            tau_k = KP_SETTLE*(kn_t  - t2) - KD_SETTLE*dt2

        elif phase == HOLD:
            tau_h = KP_SETTLE*(hip_cr - t1) - KD_SETTLE*dt1
            tau_k = KP_SETTLE*(kn_cr  - t2) - KD_SETTLE*dt2

        elif phase == STANCE:
            tau_h, tau_k = spring_torques(t1, t2, dt1, dt2)

        else:  # FLIGHT — hold crouched pose so robot lands ready to jump again
            tau_h = KP_SETTLE*(hip_cr - t1) - KD_SETTLE*dt1
            tau_k = KP_SETTLE*(kn_cr  - t2) - KD_SETTLE*dt2

        data.ctrl[0] = float(np.clip(tau_h / HIP_GEAR,  -1.0, 1.0))
        data.ctrl[1] = float(np.clip(tau_k / KNEE_GEAR, -1.0, 1.0))

        mujoco.mj_step(model, data)
        viewer.sync()

        # ── Live plot ─────────────────────────────────────
        if t - plot_last >= PLOT_DT:
            plot_last = t
            _push(t=t, z=pel_z, zd=pel_vz,
                  h1=t1, h2=t2, w1=dt1, w2=dt2,
                  uh=tau_h, uk=tau_k)
            _redraw(t)

        # ── Console log ──────────────────────────────────
        if t - log_last >= 0.25:
            log_last = t
            pn = PHASE_NAMES[phase]
            l  = np.hypot(L1*np.sin(t1)+L2*np.sin(t1+t2),
                          L1*np.cos(t1)+L2*np.cos(t1+t2))
            print(f"  [{pn:6s}] t={t:5.2f}s | "
                  f"z={pel_z:.3f}m  vz={pel_vz:+.3f}m/s  "
                  f"l={l:.3f}m  gnd={'Y' if on_gnd else 'N'}  "
                  f"τk={tau_k:+.1f}Nm")

plt.ioff()
plt.show()

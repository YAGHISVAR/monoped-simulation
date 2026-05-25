"""
Monoped — Height Control
=========================
Type a target height (m) at the bottom prompt and press Enter.
Angles scroll in the top area — typing is never interrupted.

Layout:
  ┌─ header ──────────────────────────────────┐
  │  scrolling angle output                   │
  ├───────────────────────────────────────────┤
  │  height (m) > _          (type here)      │
  │  status message                           │
  └───────────────────────────────────────────┘

Mechanism: equal links L = 0.20 m
  pelvis_z = 2 × L × cos(hip)
  Range: 0.06 m (crouched)  →  0.40 m (extended)
"""

import mujoco
import mujoco.viewer
import numpy as np
import curses

# ── Mechanism constants ────────────────────────────────────────
L            = 0.20
HEIGHT_MIN   = 0.06
HEIGHT_MAX   = 0.40
HEIGHT_STAND = 0.36
DURATION     = 1.5


def fk_pelvis_z(hip: float) -> float:
    return 2.0 * L * np.cos(hip)


def ik_hip_angle(target_z: float) -> float:
    target_z = np.clip(target_z, HEIGHT_MIN + 1e-4, HEIGHT_MAX - 1e-4)
    return float(np.arccos(np.clip(target_z / (2.0 * L), 0.0, 1.0)))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-10.0 * (x - 0.5)))


# ── Shared state ───────────────────────────────────────────────
state = {"src": HEIGHT_STAND, "target": HEIGHT_STAND, "t_start": 0.0}
active        = [True]
sim_time_ref  = [0.0]
current_pz    = [HEIGHT_STAND]


def commanded_height(t: float) -> float:
    elapsed = t - state["t_start"]
    alpha   = _sigmoid(np.clip(elapsed / DURATION, 0.0, 1.0))
    return state["src"] + alpha * (state["target"] - state["src"])


# ── PID ───────────────────────────────────────────────────────
class PID:
    def __init__(self, Kp, Ki, Kd, dt, limit=1.0):
        self.Kp = Kp; self.Ki = Ki; self.Kd = Kd
        self.dt = dt; self.limit = limit
        self.integral = 0.0; self.prev_err = 0.0

    def compute(self, target, current):
        err           = target - current
        self.integral = np.clip(self.integral + err * self.dt, -2.0, 2.0)
        deriv         = (err - self.prev_err) / self.dt
        self.prev_err = err
        return np.clip(self.Kp * err + self.Ki * self.integral + self.Kd * deriv,
                       -self.limit, self.limit)


# ── Curses simulation loop ─────────────────────────────────────
def run(stdscr, model, data, viewer, pid):
    curses.curs_set(1)
    stdscr.clear()
    rows, cols = stdscr.getmaxyx()

    # Layout rows
    HDR   = 3                          # header height
    FOOT  = 3                          # separator + input + message
    SROWS = max(rows - HDR - FOOT, 4)  # scrolling status rows

    SEP_Y   = HDR + SROWS
    INPUT_Y = SEP_Y + 1
    MSG_Y   = SEP_Y + 2

    # ── Draw header ────────────────────────────────────────────
    def hline(y):
        try: stdscr.addstr(y, 0, "═" * (cols - 1), curses.A_BOLD)
        except curses.error: pass

    hline(0)
    try:
        stdscr.addstr(1, 2, "Monoped — Height Control", curses.A_BOLD)
        stdscr.addstr(1, cols - 30,
                      f"L={L:.2f}m  [{HEIGHT_MIN:.2f} – {HEIGHT_MAX:.2f}] m")
    except curses.error:
        pass
    hline(2)

    # ── Separator + hint ───────────────────────────────────────
    try:
        stdscr.addstr(SEP_Y, 0, "─" * (cols - 1))
    except curses.error:
        pass

    PROMPT = f"  height (m) > "

    def draw_input(buf=""):
        try:
            stdscr.addstr(INPUT_Y, 0, PROMPT + buf + "  ")
            stdscr.move(INPUT_Y, len(PROMPT) + len(buf))
        except curses.error:
            pass
        stdscr.refresh()

    def draw_msg(msg):
        try:
            stdscr.addstr(MSG_Y, 0, ("  " + msg).ljust(cols - 1))
        except curses.error:
            pass
        stdscr.refresh()

    # ── Scrolling pad for status output ───────────────────────
    PAD_H  = 2000
    pad    = curses.newpad(PAD_H, cols)
    pad.scrollok(True)
    pad_line = [0]

    def status(msg):
        ln = pad_line[0] % PAD_H
        try:
            pad.addstr(ln, 0, msg[:cols - 1])
        except curses.error:
            pass
        pad_line[0] += 1
        top = max(0, pad_line[0] - SROWS)
        try:
            pad.refresh(top, 0, HDR, 0, HDR + SROWS - 1, cols - 1)
        except curses.error:
            pass
        draw_input(input_buf[0])  # keep cursor at prompt

    # Initial help text
    status(f"  STAND = {HEIGHT_STAND:.2f} m  "
           f"(hip ≈ {np.degrees(ik_hip_angle(HEIGHT_STAND)):.0f}°)")
    status(f"  Crouch = 0.14 m  "
           f"(hip ≈ {np.degrees(ik_hip_angle(0.14)):.0f}°)")
    status(f"  Full extend = {HEIGHT_MAX:.2f} m  (hip = 0°)")
    status("")

    draw_input()
    draw_msg("type 'q' to quit")
    stdscr.refresh()

    # ── Sensor indices ─────────────────────────────────────────
    S_HIP, S_CONN, S_SLIDE_Z = 3, 4, 5

    # Non-blocking getch
    stdscr.nodelay(True)
    input_buf = [""]

    last_print = -1.0
    PRINT_DT   = 0.30
    SETTLE     = 0.5

    while viewer.is_running() and active[0]:

        # ── Handle keystrokes ──────────────────────────────────
        ch = stdscr.getch()

        if ch in (10, 13, curses.KEY_ENTER):          # Enter
            raw = input_buf[0].strip()
            input_buf[0] = ""
            draw_input()
            if raw.lower() in ("q", "quit", "exit"):
                active[0] = False
                break
            elif raw:
                try:
                    h = float(raw)
                    h = np.clip(h, HEIGHT_MIN, HEIGHT_MAX)
                    hip_req = ik_hip_angle(h)
                    state["src"]     = current_pz[0]
                    state["target"]  = h
                    state["t_start"] = sim_time_ref[0]
                    draw_msg(f"→ {h:.3f} m  |  hip_target = "
                             f"{np.degrees(hip_req):.1f}°  ({hip_req:.3f} rad)")
                    status(f"  ── target → {h:.3f} m ──")
                except ValueError:
                    draw_msg(f"! enter a number between "
                             f"{HEIGHT_MIN:.2f} and {HEIGHT_MAX:.2f}")

        elif ch in (curses.KEY_BACKSPACE, 127, 8):    # Backspace
            if input_buf[0]:
                input_buf[0] = input_buf[0][:-1]
            draw_input(input_buf[0])

        elif ch != -1 and 32 <= ch < 127:             # Printable char
            input_buf[0] += chr(ch)
            draw_input(input_buf[0])

        # ── Simulation step ────────────────────────────────────
        t               = data.time
        hip_now         = float(data.sensordata[S_HIP])
        current_pz[0]   = fk_pelvis_z(hip_now)
        sim_time_ref[0] = t

        hip_tgt      = ik_hip_angle(
            HEIGHT_STAND if t < SETTLE else commanded_height(t))
        data.ctrl[0] = pid.compute(hip_tgt, hip_now)

        mujoco.mj_step(model, data)
        viewer.sync()

        # ── Print angles ───────────────────────────────────────
        if t - last_print >= PRINT_DT:
            last_print = t
            conn = float(data.sensordata[S_CONN])
            sz   = float(data.sensordata[S_SLIDE_Z])
            pz   = current_pz[0]
            status(
                f"  t={t:6.2f}s | "
                f"pelvis={pz:.3f}m  tgt={state['target']:.3f}m  "
                f"err={state['target'] - pz:+.3f}m | "
                f"hip={np.degrees(hip_now):+5.1f}°  "
                f"conn={np.degrees(conn):+5.1f}°  "
                f"slide_z={sz:+.4f}m"
            )


# ── Main ──────────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path("monoped.xml")
data  = mujoco.MjData(model)
dt    = model.opt.timestep

# qpos: [0]=slide_z  [1]=hip  [2]=connect_hinge
data.qpos[1] = ik_hip_angle(HEIGHT_STAND)
mujoco.mj_forward(model, data)

pid = PID(Kp=9.0, Ki=0.05, Kd=0.80, dt=dt)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth   = 90
    viewer.cam.elevation = -12
    viewer.cam.distance  = 1.6
    viewer.cam.lookat[:] = [0, 0, 0.25]
    curses.wrapper(run, model, data, viewer, pid)

"""
Monoped — Manual Hip Control
==============================
One slider drives the single hip motor directly.
The crank rotates, the connecting rod transmits motion,
and the pelvis slides up or down vertically.

ctrl[0] = hip motor  (−1 = CCW, +1 = CW)
"""

import mujoco
import mujoco.viewer
import numpy as np
import threading
import tkinter as tk

# ── Shared state ───────────────────────────────────────────────
motor_cmd = [0.0]
running   = [True]

# ── Mechanism FK for display ───────────────────────────────────
L_CRANK = 0.20
L_ROD   = 0.20

def fk_pelvis_z(hip_angle: float) -> float:
    """World Z of pelvis (inverted crank-slider, foot fixed at z=0)."""
    sin_h = np.sin(hip_angle)
    d = max(L_ROD**2 - L_CRANK**2 * sin_h**2, 0.0)
    return L_CRANK * np.cos(hip_angle) + np.sqrt(d)

# ── Tkinter GUI ────────────────────────────────────────────────
def run_gui():
    root = tk.Tk()
    root.title("Hip Motor Control")
    root.geometry("380x260")
    root.resizable(False, False)
    root.configure(bg="#1e1e2e")

    def label(text, color="#cdd6f4", size=10):
        tk.Label(root, text=text, bg="#1e1e2e", fg=color,
                 font=("Courier", size, "bold")).pack(pady=(6, 0))

    label("CRANK-SLIDER  —  HIP MOTOR", "#89b4fa", 12)
    label("Drag slider: red pelvis rises/falls, foot stays grounded", "#6c7086", 9)

    # Hip slider
    label("Hip Motor  [ -1 CCW | 0 stop | +1 CW ]", "#a6e3a1")
    hip_lbl = tk.Label(root, text="0.00", bg="#1e1e2e", fg="#a6e3a1",
                       font=("Courier", 11))
    hip_lbl.pack()

    def on_hip(v):
        motor_cmd[0] = float(v)
        hip_lbl.config(text=f"{float(v):+.2f}")

    tk.Scale(root, from_=-1.0, to=1.0, resolution=0.01,
             orient=tk.HORIZONTAL, length=320, command=on_hip,
             bg="#313244", fg="#cdd6f4", troughcolor="#45475a",
             highlightthickness=0, sliderlength=22,
             activebackground="#a6e3a1").pack(pady=(0, 12))

    # Buttons
    btn_frame = tk.Frame(root, bg="#1e1e2e")
    btn_frame.pack()

    def reset():   motor_cmd[0] = 0.0
    def cw():      motor_cmd[0] = 0.6
    def ccw():     motor_cmd[0] = -0.6

    for txt, cmd, col in [
        ("Reset", reset, "#6c7086"),
        ("CW",    cw,    "#a6e3a1"),
        ("CCW",   ccw,   "#f38ba8"),
    ]:
        tk.Button(btn_frame, text=txt, command=cmd,
                  bg=col, fg="#1e1e2e",
                  font=("Courier", 10, "bold"),
                  relief=tk.FLAT, padx=12, pady=5,
                  width=8).pack(side=tk.LEFT, padx=6)

    root.protocol("WM_DELETE_WINDOW",
                  lambda: running.__setitem__(0, False) or root.destroy())
    root.mainloop()


# ── MuJoCo simulation ──────────────────────────────────────────
model = mujoco.MjModel.from_xml_path("monoped.xml")
data  = mujoco.MjData(model)
dt    = model.opt.timestep

print("=" * 50)
print("  Monoped — Manual Crank-Slider Control")
print("=" * 50)
print(f"  nq={model.nq}  nu={model.nu}")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"    qpos[{i}] = {name}")
print("=" * 50)

mujoco.mj_forward(model, data)

print("\nSlider GUI opening — drag to rotate crank.\n")

gui_thread = threading.Thread(target=run_gui, daemon=True)
gui_thread.start()

S_HIP     = 3
S_CONN    = 4
S_SLIDE_Z = 5

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth   = 90
    viewer.cam.elevation = -12
    viewer.cam.distance  = 2.0
    viewer.cam.lookat[:] = [0, 0, 0.4]

    step = 0
    while viewer.is_running() and running[0]:
        data.ctrl[0] = motor_cmd[0]

        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1

        if step % 1000 == 0:
            hip  = data.sensordata[S_HIP]
            conn = data.sensordata[S_CONN]
            sz   = data.sensordata[S_SLIDE_Z]
            pz   = fk_pelvis_z(hip)
            print(f"t={data.time:5.2f}s | "
                  f"hip={hip:+.3f}rad | conn={conn:+.3f}rad | "
                  f"slide_z={sz:+.3f}m | pelvis_z(FK)={pz:.3f}m | "
                  f"ctrl={motor_cmd[0]:+.2f}")

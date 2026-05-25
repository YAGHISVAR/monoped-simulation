"""
Monoped — Manual 2-BLDC-Motor Control
=======================================
Two sliders drive the hip and knee motors independently.
Motor 1 (green) directly drives upper_link (hip).
Motor 2 (blue)  drives lower_link via chain (knee).

ctrl[0] = hip_motor   (−1 CCW | 0 stop | +1 CW)
ctrl[1] = knee_motor  (−1 CCW | 0 stop | +1 CW)
"""

import mujoco
import mujoco.viewer
import numpy as np
import threading
import tkinter as tk

# ── Shared state ───────────────────────────────────────────────
motor_cmd = [0.0, 0.0]   # [hip, knee]
running   = [True]

# ── Mechanism FK for display ───────────────────────────────────
L = 0.20

def fk_pelvis_z(hip_angle: float) -> float:
    return 2.0 * L * np.cos(hip_angle)

# ── Tkinter GUI ────────────────────────────────────────────────
def run_gui():
    root = tk.Tk()
    root.title("2-BLDC Motor Control")
    root.geometry("420x380")
    root.resizable(False, False)
    root.configure(bg="#1e1e2e")

    def label(text, color="#cdd6f4", size=10):
        tk.Label(root, text=text, bg="#1e1e2e", fg=color,
                 font=("Courier", size, "bold")).pack(pady=(6, 0))

    label("MONOPED — 2-BLDC-MOTOR LEG", "#89b4fa", 12)
    label("Motor 1 (green) = hip direct | Motor 2 (blue) = knee chain", "#6c7086", 9)

    # ── Hip slider ──────────────────────────────────────────────
    label("Hip Motor  [ ctrl[0] ]", "#a6e3a1")
    hip_lbl = tk.Label(root, text="0.00", bg="#1e1e2e", fg="#a6e3a1",
                       font=("Courier", 11))
    hip_lbl.pack()

    def on_hip(v):
        motor_cmd[0] = float(v)
        hip_lbl.config(text=f"{float(v):+.2f}")

    tk.Scale(root, from_=-1.0, to=1.0, resolution=0.01,
             orient=tk.HORIZONTAL, length=360, command=on_hip,
             bg="#313244", fg="#cdd6f4", troughcolor="#45475a",
             highlightthickness=0, sliderlength=22,
             activebackground="#a6e3a1").pack(pady=(0, 8))

    # ── Knee slider ─────────────────────────────────────────────
    label("Knee Motor  [ ctrl[1] ]", "#89b4fa")
    knee_lbl = tk.Label(root, text="0.00", bg="#1e1e2e", fg="#89b4fa",
                        font=("Courier", 11))
    knee_lbl.pack()

    def on_knee(v):
        motor_cmd[1] = float(v)
        knee_lbl.config(text=f"{float(v):+.2f}")

    tk.Scale(root, from_=-1.0, to=1.0, resolution=0.01,
             orient=tk.HORIZONTAL, length=360, command=on_knee,
             bg="#313244", fg="#cdd6f4", troughcolor="#45475a",
             highlightthickness=0, sliderlength=22,
             activebackground="#89b4fa").pack(pady=(0, 12))

    # ── Buttons ─────────────────────────────────────────────────
    btn_frame = tk.Frame(root, bg="#1e1e2e")
    btn_frame.pack()

    def reset():
        motor_cmd[0] = 0.0
        motor_cmd[1] = 0.0

    def stand():
        # IK for ~0.36 m: hip≈0.46 rad, knee≈-0.92 rad
        # Use gentle positive commands to drive toward standing
        motor_cmd[0] = -0.4
        motor_cmd[1] =  0.4

    def crouch():
        motor_cmd[0] =  0.4
        motor_cmd[1] = -0.4

    for txt, cmd, col in [
        ("Reset",  reset,  "#6c7086"),
        ("Stand",  stand,  "#a6e3a1"),
        ("Crouch", crouch, "#f38ba8"),
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
print("  Monoped — 2-BLDC-Motor Manual Control")
print("=" * 50)
print(f"  nq={model.nq}  nu={model.nu}")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"    qpos[{i}] = {name}")
print("=" * 50)

# Start at standing pose
from math import acos
hip_init     = acos(min(0.36 / (2 * L), 1.0))
data.qpos[1] = hip_init
data.qpos[2] = -2.0 * hip_init
mujoco.mj_forward(model, data)

print("\nSlider GUI opening — use sliders to drive each motor.\n")

gui_thread = threading.Thread(target=run_gui, daemon=True)
gui_thread.start()

S_HIP     = 3
S_KNEE    = 4
S_SLIDE_Z = 5

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth   = 90
    viewer.cam.elevation = -12
    viewer.cam.distance  = 1.6
    viewer.cam.lookat[:] = [0, 0, 0.25]

    step = 0
    while viewer.is_running() and running[0]:
        data.ctrl[0] = motor_cmd[0]
        data.ctrl[1] = motor_cmd[1]

        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1

        if step % 1000 == 0:
            hip  = data.sensordata[S_HIP]
            knee = data.sensordata[S_KNEE]
            sz   = data.sensordata[S_SLIDE_Z]
            pz   = fk_pelvis_z(hip)
            print(f"t={data.time:5.2f}s | "
                  f"hip={np.degrees(hip):+6.1f}° | "
                  f"knee={np.degrees(knee):+6.1f}° | "
                  f"slide_z={sz:+.3f}m | pelvis_z={pz:.3f}m | "
                  f"ctrl=[{motor_cmd[0]:+.2f}, {motor_cmd[1]:+.2f}]")

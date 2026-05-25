"""
Bipedal Robot — ZMP Balance Controller (Fixed)
===============================================
Sensor map (confirmed from debug_sensors.py):
  [0:3]   pelvis_accel   (ax, ay, az)
  [3:6]   left_foot_pos  (x, y, z)
  [6:9]   right_foot_pos (x, y, z)
  [9]     lhr_pos   left  hip roll
  [10]    lhp_pos   left  hip pitch
  [11]    lk_pos    left  knee
  [12]    la_pos    left  ankle
  [13]    rhr_pos   right hip roll
  [14]    rhp_pos   right hip pitch
  [15]    rk_pos    right knee
  [16]    ra_pos    right ankle

Free joint qpos layout (nq=15):
  [0:3]  pelvis position  (x, y, z)
  [3:7]  pelvis quaternion (qw, qx, qy, qz)
  [7:15] joint angles (lhr, lhp, lk, la, rhr, rhp, rk, ra)
"""

import mujoco
import mujoco.viewer
import numpy as np

# ══════════════════════════════════════════════
#  PID Controller
# ══════════════════════════════════════════════
class PIDController:
    def __init__(self, Kp, Ki, Kd, dt, output_limit=1.0):
        self.Kp         = Kp
        self.Ki         = Ki
        self.Kd         = Kd
        self.dt         = dt
        self.limit      = output_limit
        self.integral   = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral   = 0.0
        self.prev_error = 0.0

    def compute(self, target, current):
        error           = target - current
        self.integral   = np.clip(self.integral + error * self.dt, -1.5, 1.5)
        derivative      = (error - self.prev_error) / self.dt
        self.prev_error = error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return np.clip(output, -self.limit, self.limit)


# ══════════════════════════════════════════════
#  ZMP Balance Controller
# ══════════════════════════════════════════════
class ZMPController:
    """
    Inverted pendulum ZMP estimator.

    ZMP equations:
      zmp_x = com_x - (com_z / g) * ax_filtered
      zmp_y = com_y - (com_z / g) * ay_filtered

    Corrections:
      forward error  → adjust hip pitch + ankle
      lateral error  → adjust hip roll
    """
    def __init__(self, g=9.81):
        self.g          = g
        self.Kx         = 0.8    # forward/back gain
        self.Ky         = 0.6    # lateral gain
        self.Ka         = 0.3    # ankle correction gain
        self.alpha      = 0.1    # low-pass filter strength
        self.ax_f       = 0.0    # filtered ax
        self.ay_f       = 0.0    # filtered ay
        # correction limits (rad)
        self.hip_lim    = 0.20
        self.ankle_lim  = 0.12
        self.roll_lim   = 0.12

    def update(self, com_pos, accel, support_xy):
        """
        com_pos    : pelvis [x, y, z]
        accel      : [ax, ay, az] from IMU
        support_xy : [x, y] midpoint of feet
        Returns correction dict
        """
        # Low-pass filter (removes vibration noise)
        self.ax_f = self.alpha * accel[0] + (1 - self.alpha) * self.ax_f
        self.ay_f = self.alpha * accel[1] + (1 - self.alpha) * self.ay_f

        com_z = max(com_pos[2], 0.2)

        # ZMP estimate
        zmp_x = com_pos[0] - (com_z / self.g) * self.ax_f
        zmp_y = com_pos[1] - (com_z / self.g) * self.ay_f

        # Error from support center
        ex = zmp_x - support_xy[0]
        ey = zmp_y - support_xy[1]

        hip_pitch_corr = np.clip(-self.Kx * ex, -self.hip_lim,   self.hip_lim)
        ankle_corr     = np.clip( self.Ka * ex, -self.ankle_lim,  self.ankle_lim)
        hip_roll_corr  = np.clip(-self.Ky * ey, -self.roll_lim,   self.roll_lim)

        return {
            "hip_pitch" : hip_pitch_corr,
            "ankle"     : ankle_corr,
            "hip_roll"  : hip_roll_corr,
            "zmp_x"     : zmp_x,
            "zmp_y"     : zmp_y,
            "ex"        : ex,
            "ey"        : ey,
        }


# ══════════════════════════════════════════════
#  Load model
# ══════════════════════════════════════════════
model = mujoco.MjModel.from_xml_path("biped.xml")
data  = mujoco.MjData(model)

print("=" * 55)
print("  Bipedal Robot — ZMP Balance Controller")
print("=" * 55)
print(f"  nq={model.nq}  nu={model.nu}  nsensor={model.nsensor}")
print("=" * 55)

dt = model.opt.timestep

# ══════════════════════════════════════════════
#  Confirmed sensor indices from debug output
# ══════════════════════════════════════════════
S_ACCEL      = slice(0, 3)
S_LFOOT      = slice(3, 6)
S_RFOOT      = slice(6, 9)
S_L_HR       = 9
S_L_HP       = 10
S_L_KNEE     = 11
S_L_ANKLE    = 12
S_R_HR       = 13
S_R_HP       = 14
S_R_KNEE     = 15
S_R_ANKLE    = 16

# ctrl indices (confirmed from debug output)
C_L_HR    = 0
C_L_HP    = 1
C_L_KNEE  = 2
C_L_ANKLE = 3
C_R_HR    = 4
C_R_HP    = 5
C_R_KNEE  = 6
C_R_ANKLE = 7

# ══════════════════════════════════════════════
#  PID controllers — one per joint
# ══════════════════════════════════════════════
#              Kp    Ki     Kd
pids = {
    C_L_HR    : PIDController(5.0, 0.02, 0.40, dt),
    C_L_HP    : PIDController(6.0, 0.02, 0.50, dt),
    C_L_KNEE  : PIDController(5.0, 0.02, 0.40, dt),
    C_L_ANKLE : PIDController(4.0, 0.01, 0.25, dt),
    C_R_HR    : PIDController(5.0, 0.02, 0.40, dt),
    C_R_HP    : PIDController(6.0, 0.02, 0.50, dt),
    C_R_KNEE  : PIDController(5.0, 0.02, 0.40, dt),
    C_R_ANKLE : PIDController(4.0, 0.01, 0.25, dt),
}

# sensor index for each ctrl index
ctrl_to_sensor = {
    C_L_HR    : S_L_HR,
    C_L_HP    : S_L_HP,
    C_L_KNEE  : S_L_KNEE,
    C_L_ANKLE : S_L_ANKLE,
    C_R_HR    : S_R_HR,
    C_R_HP    : S_R_HP,
    C_R_KNEE  : S_R_KNEE,
    C_R_ANKLE : S_R_ANKLE,
}

zmp = ZMPController(g=9.81)

# ══════════════════════════════════════════════
#  Gait trajectory
# ══════════════════════════════════════════════
def gait_targets(t):
    freq  = 0.7
    omega = 2 * np.pi * freq

    phi_L = omega * t
    phi_R = omega * t + np.pi

    hip_amp   = 0.25
    knee_amp  = 0.38
    ankle_amp = 0.13
    roll_amp  = 0.05

    return {
        C_L_HR    :  roll_amp  * np.sin(phi_L * 0.5),
        C_L_HP    :  hip_amp   * np.sin(phi_L),
        C_L_KNEE  :  knee_amp  * np.clip(np.sin(phi_L + 0.4), 0, 1),
        C_L_ANKLE : -ankle_amp * np.sin(phi_L),
        C_R_HR    : -roll_amp  * np.sin(phi_R * 0.5),
        C_R_HP    :  hip_amp   * np.sin(phi_R),
        C_R_KNEE  :  knee_amp  * np.clip(np.sin(phi_R + 0.4), 0, 1),
        C_R_ANKLE : -ankle_amp * np.sin(phi_R),
    }

# ══════════════════════════════════════════════
#  Main loop
# ══════════════════════════════════════════════
print("\nLaunching viewer — watch terminal for ZMP log...\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth   = 45
    viewer.cam.elevation = -15
    viewer.cam.distance  = 3.5
    viewer.cam.lookat[:] = [0, 0, 0.8]

    step = 0

    while viewer.is_running():
        t = data.time

        # 1. Gait reference
        targets = gait_targets(t)

        # 2. Read sensors (confirmed indices)
        accel   = data.sensordata[S_ACCEL]
        lfoot   = data.sensordata[S_LFOOT]
        rfoot   = data.sensordata[S_RFOOT]

        # 3. CoM position from free joint (qpos[0:3] = x,y,z)
        com_pos = data.qpos[0:3].copy()

        # 4. Support center = midpoint of feet (x,y only)
        support_xy = np.array([
            (lfoot[0] + rfoot[0]) / 2.0,
            (lfoot[1] + rfoot[1]) / 2.0,
        ])

        # 5. ZMP correction
        corr = zmp.update(com_pos, accel, support_xy)

        # 6. Add ZMP corrections to gait targets
        targets[C_L_HP]    += corr["hip_pitch"]
        targets[C_R_HP]    += corr["hip_pitch"]
        targets[C_L_ANKLE] += corr["ankle"]
        targets[C_R_ANKLE] += corr["ankle"]
        targets[C_L_HR]    += corr["hip_roll"]
        targets[C_R_HR]    -= corr["hip_roll"]

        # 7. PID drives joints to corrected targets
        for ctrl_idx, pid in pids.items():
            sensor_idx = ctrl_to_sensor[ctrl_idx]
            current    = data.sensordata[sensor_idx]
            target     = targets[ctrl_idx]
            data.ctrl[ctrl_idx] = pid.compute(target, current)

        # 8. Physics step
        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1

        # 9. Log every 500 steps (~1 sec)
        if step % 500 == 0:
            px, py, pz = data.qpos[0], data.qpos[1], data.qpos[2]
            print(f"t={t:6.2f}s | "
                  f"pelvis z={pz:+.3f}m | "
                  f"ZMP=({corr['zmp_x']:+.3f}, {corr['zmp_y']:+.3f}) | "
                  f"err=({corr['ex']:+.3f}, {corr['ey']:+.3f})")

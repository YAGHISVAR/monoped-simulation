# MuJoCo Robot Simulations

MuJoCo-based physics simulations for a 2-BLDC-motor monoped and a full biped robot, featuring PID control, ZMP balance, and interactive motor control.

---

## Repository Structure

```
mujoco_sim/
├── monoped.xml          # 2-BLDC-motor monoped MuJoCo model
├── biped.xml            # Biped robot MuJoCo model (8 actuators)
├── biped.urdf           # Biped URDF definition
├── manual_control.py    # Monoped — Tkinter GUI slider control
├── simulate_monoped.py  # Monoped — sinusoidal gait with PID
├── sit_stand.py         # Monoped — interactive height control + live plot
├── simulate_biped.py    # Biped — ZMP balance + gait controller
└── debug_sensors.py     # Utility: print sensor/joint/actuator map
```

---

## Prerequisites

- Python 3.10+
- A display (MuJoCo viewer requires a desktop/X11 environment)

---

## Environment Setup

A Python virtual environment lives at `~/mujoco_env`.

### Activate the environment

```bash
source ~/mujoco_env/bin/activate
```

### Deactivate when done

```bash
deactivate
```

### Install dependencies (first-time only)

```bash
source ~/mujoco_env/bin/activate
pip install mujoco numpy matplotlib
```

> `tkinter` ships with the standard Python installation and does not need to be pip-installed.

---

## Running the Simulations

All scripts must be run from the `mujoco_sim/` directory so they can find the XML model files.

```bash
cd ~/mujoco_sim
source ~/mujoco_env/bin/activate
```

---

### 1. Monoped — Manual Motor Control (Tkinter GUI)

Launches a MuJoCo viewer alongside a Tkinter window with two sliders for direct motor control.

```bash
python manual_control.py
```

**Controls:**

| Slider | Joint | Range |
|--------|-------|-------|
| Hip Motor `ctrl[0]` | Hip (direct drive) | −1 CCW → +1 CW |
| Knee Motor `ctrl[1]` | Knee (chain drive) | −1 CCW → +1 CW |

**Buttons:** `Reset` — `Stand` — `Crouch`

---

### 2. Monoped — Sinusoidal Gait (PID)

Runs the monoped with a sinusoidal pelvis height trajectory. PID controllers drive hip and knee joints.

```bash
python simulate_monoped.py
```

The terminal prints joint angles and pelvis height every 500 steps (~0.5 s).

---

### 3. Monoped — Interactive Height Control + Live Plot

Type a target height in the terminal and watch the robot move to it. A matplotlib window displays the live PID height error after each setpoint change.

```bash
python sit_stand.py
```

**Terminal prompt:**

```
height (m) > 0.25
```

- Height range: **0.06 m** (crouched) → **0.40 m** (fully extended)
- Default stand height: **0.36 m**
- Type `q` or `quit` to exit

---

### 4. Biped — ZMP Balance Controller

Runs the full biped with a sinusoidal gait trajectory and a ZMP (Zero Moment Point) inverted-pendulum balance corrector.

```bash
python simulate_biped.py
```

The terminal logs pelvis height and ZMP error every 500 steps (~1 s).

---

### 5. Debug — Sensor / Joint / Actuator Map

Prints the complete sensor index map, joint list, and actuator list for `monoped.xml`. Useful for development.

```bash
python debug_sensors.py
```

---

## Model Details

### Monoped (`monoped.xml`)

- **Degrees of freedom:** slide_z (vertical rail) + hip hinge + knee hinge
- **Actuators:** `hip_motor` (gear 55 Nm), `knee_motor` (gear 45 Nm)
- **IK relation:** `knee = −2 × hip`, pelvis height = `2 × L × cos(hip)`, L = 0.20 m

| Sensor index | Name | Description |
|---|---|---|
| `[0:3]` | `imu_accel` | Pelvis accelerometer (ax, ay, az) |
| `[3]` | `hip_pos` | Hip joint angle (rad) |
| `[4]` | `knee_pos` | Knee joint angle (rad) |
| `[5]` | `slide_z_pos` | Pelvis vertical offset (m) |
| `[6]` | `hip_vel` | Hip joint velocity |
| `[7]` | `knee_vel` | Knee joint velocity |
| `[8]` | `slide_z_vel` | Pelvis vertical velocity |
| `[9:12]` | `rod_tip_pos` | Foot tip world position |
| `[12:15]` | `pelvis_pos` | Pelvis world position |

### Biped (`biped.xml`)

- **Joints per leg:** hip roll, hip pitch, knee, ankle (× 2 legs = 8 total)
- **Actuator torques:** Hip roll ±40 Nm, Hip pitch ±80 Nm, Knee ±60 Nm, Ankle ±40 Nm
- **Base:** free-floating pelvis (6-DOF), nq = 15

---

## Quick Reference

```bash
# Activate environment
source ~/mujoco_env/bin/activate

# Navigate to project
cd ~/mujoco_sim

# Run manual slider control (monoped)
python manual_control.py

# Run sinusoidal gait (monoped)
python simulate_monoped.py

# Run interactive height control with live plot (monoped)
python sit_stand.py

# Run ZMP biped balance simulation
python simulate_biped.py

# Print sensor/joint/actuator map
python debug_sensors.py
```

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("monoped.xml")
data  = mujoco.MjData(model)
mujoco.mj_step(model, data)

print("=" * 55)
print("  SENSOR MAP — monoped_bldc_2dof")
print("=" * 55)
print(f"  sensordata length: {len(data.sensordata)}")
idx = 0
for i in range(model.nsensor):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
    dim  = model.sensor_dim[i]
    vals = data.sensordata[idx:idx+dim]
    print(f"  [{idx:2d}:{idx+dim:2d}]  {name:<20} dim={dim}  val={np.round(vals, 4)}")
    idx += dim

print()
print("=" * 55)
print("  JOINTS")
print("=" * 55)
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"  joint[{i}]: {name}")

print()
print("=" * 55)
print("  ACTUATORS")
print("=" * 55)
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"  ctrl[{i}]: {name}")

print()
print("=" * 55)
print("  EQUALITY CONSTRAINTS")
print("=" * 55)
if model.neq == 0:
    print("  (none — contact physics only)")
else:
    for i in range(model.neq):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i)
        print(f"  eq[{i}]: {name}  active={bool(model.eq_active0[i])}")

print(f"\n  nq={model.nq}  nv={model.nv}  nu={model.nu}  neq={model.neq}")

# src/diag_torque_drop.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.config_loader import ConfigLoader
from src.signal_generator import Scenario, SignalGenerator

loader = ConfigLoader("config")
gen = SignalGenerator(loader)
fault_registry = loader.load_fault_types()

scenario = Scenario.from_yaml(
    "config/scenarios/highway_cruise_torque_drop.yaml",
    fault_registry=fault_registry,
)

# Build a single torque drop at t=30s (150 ms, 70% loss)
n = int(scenario.duration_s * scenario.base_rate_hz)
dt = 1.0 / scenario.base_rate_hz
t_s = np.arange(n) * dt

multiplier = np.ones(n)
drop_start = int(30.0 / dt)
drop_len = int(0.15 / dt)
multiplier[drop_start:drop_start + drop_len] = 0.3

df_clean = gen.generate(scenario)
df_with_mult = gen.generate(scenario, faults={"engine_torque_multiplier": multiplier})

# Focus on the drop window (t=29.9 to t=30.3)
mask = (df_clean["time_s"] >= 29.9) & (df_clean["time_s"] <= 30.3)

print("Time  |  Clean RPM  |  Multiplied RPM  |  Diff")
print("-" * 55)
clean_slice = df_clean.loc[mask, "engine_rpm"].values
mult_slice = df_with_mult.loc[mask, "engine_rpm"].values
for i in range(0, len(clean_slice), 5):
    t = df_clean.loc[mask, "time_s"].iloc[i]
    print(f"{t:6.2f}  |  {clean_slice[i]:8.1f}  |  {mult_slice[i]:10.1f}  |  {clean_slice[i] - mult_slice[i]:6.1f}")

print()
max_diff = abs(clean_slice - mult_slice).max()
print(f"Max diff in drop window: {max_diff:.1f} RPM")
if max_diff > 100:
    print("✅ The fix is active — RPM dips during the drop.")
else:
    print("⚠️  The fix does not affect the drop — investigate.")
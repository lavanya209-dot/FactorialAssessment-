"""
PROTOCOL CHECK - reads the actual test settings from the raw Neware files, for the Methods section:
  - CV cut-off current (current at the end of the charge step), in mA and as a C-rate fraction
  - sampling interval (time between recorded points) during charge, rest and discharge
  - what the file contains BEFORE the charge starts (rest = soak time recorded in the file)
  - rest time between charge and discharge
  - the step sequence of each file (to check the order of operations)

Run (same folder as Data_2026_1.py):
  python Data_2026_Protocol.py --data-path "C:/Users/SASTRA/Videos/libd"
  python Data_2026_Protocol.py --selftest
Output: out_full/protocol/protocol_per_file.csv and a summary printed to the screen.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from Data_2026_1 import load_file
except ImportError:
    from script1_fixed import load_file

NOMINAL_AH = 2.0


def segments(df):
    """Consecutive runs of the same step type: [(type, start_row, end_row), ...]."""
    t = df["step_type"].values
    out, s = [], 0
    for i in range(1, len(t) + 1):
        if i == len(t) or t[i] != t[s]:
            out.append((t[s], s, i - 1))
            s = i
    return out


def protocol(df):
    seg = segments(df)
    seq = " > ".join(x[0] for x in seg)
    r = {"file_id": df["file_id"].iloc[0], "temperature": df["temperature"].iloc[0],
         "pressure": df["pressure"].iloc[0], "c_rate": df["c_rate"].iloc[0],
         "battery_id": df["battery_id"].iloc[0], "step_sequence": seq}
    chg = df[df["step_type"] == "charge"]
    dis = df[df["step_type"] == "discharge"]
    if len(chg):
        r["cv_cutoff_current_mA"] = abs(chg["current"].iloc[-1]) * 1000
        r["cv_cutoff_C_fraction"] = abs(chg["current"].iloc[-1]) / NOMINAL_AH
        r["charge_cc_current_mA"] = abs(chg["current"].iloc[: min(60, len(chg))].median()) * 1000
        r["charge_end_voltage_V"] = chg["voltage"].iloc[-1]
    for name, part in [("charge", chg), ("discharge", dis)]:
        if len(part) > 2:
            r[f"dt_{name}_s_median"] = float(np.median(np.diff(part["elapsed_s"].values)))
    # anything before the first charge (soak / rest recorded in the file)
    first_chg = chg.index[0] if len(chg) else None
    pre = df.loc[: first_chg - 1] if first_chg is not None and first_chg > df.index[0] else df.iloc[0:0]
    r["pre_charge_steps"] = " > ".join(sorted(set(pre["step_type"]))) if len(pre) else "none"
    r["pre_charge_duration_min"] = ((pre["elapsed_s"].iloc[-1] - pre["elapsed_s"].iloc[0]) / 60
                                    if len(pre) > 1 else 0.0)
    r["pre_charge_start_voltage_V"] = df["voltage"].iloc[0]
    # rest between charge and discharge
    rests = [s for s in seg if s[0] == "rest"]
    mid = [s for s in rests if len(chg) and len(dis)
           and df.index[s[1]] > chg.index[-1] and df.index[s[2]] < dis.index[0]]
    if mid:
        a, b = mid[0][1], mid[0][2]
        r["rest_before_discharge_min"] = (df["elapsed_s"].iloc[b] - df["elapsed_s"].iloc[a]) / 60
    # anything after the discharge
    post = df.loc[dis.index[-1] + 1:] if len(dis) else df.iloc[0:0]
    r["post_discharge_steps"] = " > ".join(sorted(set(post["step_type"]))) if len(post) else "none"
    r["post_discharge_duration_min"] = ((post["elapsed_s"].iloc[-1] - post["elapsed_s"].iloc[0]) / 60
                                        if len(post) > 1 else 0.0)
    r["discharge_end_voltage_V"] = dis["voltage"].iloc[-1] if len(dis) else np.nan
    return r


def run(data_path, out_dir):
    base = Path(data_path)
    files = sorted(list(base.rglob("*.xls")) + list(base.rglob("*.xlsx")))
    if not files:
        sys.exit("No files found - check --data-path")
    rows = []
    for i, f in enumerate(files, 1):
        try:
            rows.append(protocol(load_file(f)))
            print(f"[{i}/{len(files)}] {rows[-1]['file_id']}")
        except Exception as e:
            print(f"[{i}/{len(files)}] FAILED {f.name}: {e}")
    p = pd.DataFrame(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p.to_csv(out / "protocol_per_file.csv", index=False)
    summarise(p)
    print(f"\nSaved {out / 'protocol_per_file.csv'}")


def summarise(p):
    pd.set_option("display.width", 200)
    print("\n=== 1. Step sequence in the files (count of files) ===")
    print(p["step_sequence"].value_counts().to_string())
    print("\n=== 2. CV cut-off current (current at end of charge) ===")
    print(p.groupby("c_rate")[["cv_cutoff_current_mA", "cv_cutoff_C_fraction"]]
          .describe().round(3).T.to_string())
    print("\n=== 3. Sampling interval (seconds between points, median per file) ===")
    print(p[[c for c in p.columns if c.startswith("dt_")]].describe().round(2).to_string())
    print("\n=== 4. Before the charge (soak / rest inside the file) ===")
    print(p["pre_charge_steps"].value_counts().to_string())
    print(p["pre_charge_duration_min"].describe().round(1).to_string())
    print("\n=== 5. Rest between charge and discharge (minutes) ===")
    if "rest_before_discharge_min" in p:
        print(p["rest_before_discharge_min"].describe().round(2).to_string())
    print("\n=== 6. After the discharge ===")
    print(p["post_discharge_steps"].value_counts().to_string())
    print(p["post_discharge_duration_min"].describe().round(1).to_string())
    print("\n=== 7. Voltages: start of file, end of charge, end of discharge ===")
    print(p[["pre_charge_start_voltage_V", "charge_end_voltage_V", "discharge_end_voltage_V"]]
          .describe().round(3).to_string())


def selftest():
    n = 600
    pre = pd.DataFrame({"step": ["搁置"] * 30, "voltage": 3.2, "current": 0.0, "capacity": 0.0})
    cc = pd.DataFrame({"step": ["恒流恒压充电"] * n, "voltage": np.r_[np.linspace(3.2, 3.65, 400), [3.65] * 200],
                       "current": np.r_[[1.0] * 400, np.linspace(1.0, 0.1, 200)],
                       "capacity": np.linspace(0, 1.5, n)})
    rest = pd.DataFrame({"step": ["搁置"] * 10, "voltage": 3.4, "current": 0.0, "capacity": 0.0})
    dis = pd.DataFrame({"step": ["恒流放电"] * n, "voltage": np.linspace(3.3, 2.0, n),
                        "current": -1.0, "capacity": np.linspace(0, 1.5, n)})
    df = pd.concat([pre, cc, rest, dis], ignore_index=True)
    from Data_2026_1 import add_cycles
    df = add_cycles(df)
    df["elapsed_s"] = np.arange(len(df)) * 60.0  # 60 s per point
    for k, v in {"file_id": "t", "temperature": 25.0, "pressure": 300, "c_rate": 0.5, "battery_id": 1}.items():
        df[k] = v
    r = protocol(df)
    assert r["step_sequence"] == "rest > charge > rest > discharge", r["step_sequence"]
    assert abs(r["cv_cutoff_current_mA"] - 100) < 1e-6, r
    assert r["dt_charge_s_median"] == 60.0
    assert r["pre_charge_steps"] == "rest" and abs(r["pre_charge_duration_min"] - 29) < 1e-6, r
    assert abs(r["rest_before_discharge_min"] - 9) < 1e-6, r
    summarise(pd.DataFrame([r]))
    print("\nselftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="C:/Users/SASTRA/Videos/libd")
    ap.add_argument("--out-dir", default="out_full/protocol")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else run(a.data_path, a.out_dir)
"""
STEP 2 (CORRECTED) - Per-cell outcomes (one row per Excel file = one cell under one condition)

Reads the 288 raw Excel files through the fixed loader (Data_2026_1.py / script1_fixed.py)
and computes outcomes from the ONE charge/discharge cycle in each file.
Only a small table (288 rows) is written.

CHANGE vs previous version: dc_resistance_ohm is now a "10 s DC resistance":
    (median rest voltage just before discharge  -  voltage 10 s into discharge) / current at 10 s
The old version divided by the current of the FIRST discharge row, which is still ramping at 0.5C
and produced artefacts up to 1.1 ohm. ir_drop_V is the same 10 s voltage drop.

Run:
  python Data_2026_Step2.py --data-path "C:/Users/SASTRA/Videos/libd" --out-dir out_full
  python Data_2026_Step2.py --selftest
Output: <out-dir>/per_cell_outcomes.csv   (check the column 'resistance_method' says 10s_window)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESISTANCE_WINDOW_S = 10.0


def _trapz(y, x):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    if len(y) < 2:
        return 0.0
    return float(np.sum((y[1:] + y[:-1]) / 2.0 * np.diff(x)))


def cell_outcomes(df: pd.DataFrame) -> dict:
    """df = output of load_file() for ONE file (columns: step_type, voltage, current,
    capacity (Ah, accumulated within step), elapsed_s, + condition columns)."""
    chg = df[df["step_type"] == "charge"]
    dis = df[df["step_type"] == "discharge"]
    out = {
        "file_id": df["file_id"].iloc[0],
        "temperature": df["temperature"].iloc[0],
        "pressure": df["pressure"].iloc[0],
        "c_rate": df["c_rate"].iloc[0],
        "battery_id": df["battery_id"].iloc[0],
        "n_cycles_detected": df["cycle"].nunique(),
        "n_rows": len(df),
    }
    if len(chg) < 5 or len(dis) < 5:
        out["usable"] = False
        return out
    out["usable"] = True

    # capacities (Ah) - accumulated within the step, so the max is the step total
    out["charge_capacity_Ah"] = chg["capacity"].max()
    out["discharge_capacity_Ah"] = dis["capacity"].max()
    out["coulombic_efficiency"] = out["discharge_capacity_Ah"] / out["charge_capacity_Ah"]

    # energies (Wh) by trapezoidal integration of |V*I| over time
    e_chg = _trapz(chg["voltage"] * chg["current"].abs(), chg["elapsed_s"]) / 3600.0
    e_dis = _trapz(dis["voltage"] * dis["current"].abs(), dis["elapsed_s"]) / 3600.0
    out["charge_energy_Wh"] = e_chg
    out["discharge_energy_Wh"] = e_dis
    out["energy_efficiency"] = e_dis / e_chg if e_chg > 0 else np.nan
    out["mean_discharge_voltage_V"] = e_dis / out["discharge_capacity_Ah"]

    # durations (s)
    out["charge_time_s"] = chg["elapsed_s"].iloc[-1] - chg["elapsed_s"].iloc[0]
    out["discharge_time_s"] = dis["elapsed_s"].iloc[-1] - dis["elapsed_s"].iloc[0]

    # constant-current vs constant-voltage share of charge (CV = current has fallen below 98% of CC level)
    i_cc = chg["current"].iloc[: min(60, len(chg))].median()
    in_cv = chg["current"] < 0.98 * i_cc
    out["cv_time_fraction"] = float(in_cv.mean())

    # ---- 10 s DC resistance (replaces the single-sample proxy)
    out["ir_drop_V"] = np.nan
    out["dc_resistance_ohm"] = np.nan
    out["resistance_method"] = f"{RESISTANCE_WINDOW_S:g}s_window"
    pos = df.index.get_loc(dis.index[0])
    if pos >= 1:
        before = df.iloc[max(0, pos - 5):pos]
        rest = before[before["step_type"] == "rest"]
        v_rest = (rest if len(rest) else before)["voltage"].median()
        t0 = dis["elapsed_s"].iloc[0]
        late = dis[dis["elapsed_s"] >= t0 + RESISTANCE_WINDOW_S]
        if len(late):
            v_w = late["voltage"].iloc[0]
            i_w = abs(late["current"].iloc[0])
            out["ir_drop_V"] = v_rest - v_w
            out["dc_resistance_ohm"] = (v_rest - v_w) / i_w if i_w > 1e-3 else np.nan

    out["max_charge_voltage_V"] = chg["voltage"].max()
    out["min_discharge_voltage_V"] = dis["voltage"].min()
    return out


def run(data_path, out_dir, limit):
    try:
        from Data_2026_1 import load_file
    except ImportError:
        from script1_fixed import load_file
    base = Path(data_path)
    files = sorted(list(base.rglob("*.xls")) + list(base.rglob("*.xlsx")))
    if not files:
        sys.exit("No files found - check --data-path")
    if limit:
        files = files[:limit]
    rows = []
    for i, f in enumerate(files, 1):
        try:
            d = load_file(f)
            r = cell_outcomes(d)
            rows.append(r)
            print(f"[{i}/{len(files)}] {r['file_id']:<22} cap={r.get('discharge_capacity_Ah', float('nan')):.4f} Ah"
                  f"  R10s={r.get('dc_resistance_ohm', float('nan')):.4f} ohm  usable={r['usable']}")
        except Exception as e:
            print(f"[{i}/{len(files)}] FAILED {f.name}: {e}")
    res = pd.DataFrame(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "per_cell_outcomes.csv", index=False)
    print(f"\nSaved {len(res)} rows -> {out/'per_cell_outcomes.csv'}")
    print(f"Usable cells: {int(res['usable'].sum())} / {len(res)}")
    print(res.groupby(["temperature", "pressure", "c_rate"]).size().describe().to_string())
    if "dc_resistance_ohm" in res:
        print("\nR(10 s) summary:\n", res["dc_resistance_ohm"].describe().round(4).to_string())
        print("missing R:", int(res["dc_resistance_ohm"].isna().sum()))


def selftest():
    n = 1000
    t = np.arange(n, dtype=float)
    chg = pd.DataFrame({"step_type": "charge", "voltage": np.linspace(3.0, 3.6, n),
                        "current": 1.0, "capacity": np.cumsum(np.full(n, 1.0 / 3600)), "elapsed_s": t})
    rest = pd.DataFrame({"step_type": "rest", "voltage": 3.55, "current": 0.0,
                         "capacity": 0.0, "elapsed_s": n + np.arange(300, dtype=float)})
    dis = pd.DataFrame({"step_type": "discharge", "voltage": np.linspace(3.30, 2.5, n),
                        "current": -1.0, "capacity": np.cumsum(np.full(n, 1.0 / 3600)),
                        "elapsed_s": n + 300 + np.arange(n, dtype=float)})
    # make the FIRST discharge row a slow ramp (tiny current) to mimic the 0.5C artefact
    dis.loc[0, "current"] = -0.1
    df = pd.concat([chg, rest, dis], ignore_index=True)
    for k, v in {"file_id": "t", "temperature": 25.0, "pressure": 300, "c_rate": 1.0,
                 "battery_id": 1, "cycle": 0}.items():
        df[k] = v
    r = cell_outcomes(df)
    v10 = 3.30 + (2.5 - 3.30) * 10 / (n - 1)          # voltage 10 s into discharge
    exp_drop = 3.55 - v10
    assert abs(r["discharge_capacity_Ah"] - 1000 / 3600) < 1e-9
    assert abs(r["coulombic_efficiency"] - 1.0) < 1e-9
    assert abs(r["ir_drop_V"] - exp_drop) < 1e-9, r["ir_drop_V"]
    assert abs(r["dc_resistance_ohm"] - exp_drop / 1.0) < 1e-9      # NOT blown up by the 0.1 A first row
    assert r["resistance_method"] == "10s_window"
    assert 2.5 < r["mean_discharge_voltage_V"] < 3.3
    print("selftest OK: R10s =", round(r["dc_resistance_ohm"], 4), "ohm (first-row ramp artefact ignored)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="C:/Users/SASTRA/Videos/libd")
    ap.add_argument("--out-dir", default="out_full")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else run(a.data_path, a.out_dir, a.limit)
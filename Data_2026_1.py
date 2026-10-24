"""
SCRIPT 1 (FIXED) - Battery data loader
Fixes over the old Script1_Dataloader.py:
  1. Unique file_id per file (old 'file_name' repeated across all 48 folders).
  2. Cycle detection from the Chinese Neware step names (old code used sign-flips / English words).
  3. Conditions parsed strictly from the folder path (fails loudly, no silent 25C/300N defaults).
  4. Monotonic elapsed time per file (old time_seconds resets at every step).
  5. Writes cycle_summary.csv (one row per file + cycle) -> this answers reviewers' cycle questions.
  6. No interactive prompts. Run:  python script1_fixed.py --data-path "C:/Users/SASTRA/Videos/libd"
     Quick test on a few files:   python script1_fixed.py --data-path ... --limit 6
     Logic self-test (no data):   python script1_fixed.py --selftest

Neware step names (Chinese):  充电 = charge,  放电 = discharge,  搁置 = rest
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TEMP_RE = re.compile(r"^(\d+(?:\.\d+)?)d$", re.I)       # 10d
PRESS_RE = re.compile(r"^(\d+)n$", re.I)                 # 300N
CRATE_RE = re.compile(r"^(\d+(?:\.\d+)?)c$", re.I)       # 0.5C
BATT_RE = re.compile(r"No\.?\s*(\d+)", re.I)             # Battery No.1

COLUMN_MAP = {
    "Step Name": "step",
    "Record Number": "record_num",
    "Record Time(h:min:s.ms)": "time_str",
    "Voltage(V)": "voltage",
    "Current(mA)": "current_mA",
    "Capacity(mAh)": "capacity_mAh",
    "Power(mWh)": "power_mWh",
    "Absolute Time": "abs_time",
}


# ---------------------------------------------------------------- helpers
def parse_conditions(path: Path) -> dict:
    cond = {}
    for part in path.parts:
        p = str(part).strip()
        if (m := TEMP_RE.match(p)):
            cond["temperature"] = float(m.group(1))
        elif (m := PRESS_RE.match(p)):
            cond["pressure"] = int(m.group(1))
        elif (m := CRATE_RE.match(p)):
            cond["c_rate"] = float(m.group(1))
    m = BATT_RE.search(path.stem)
    if m:
        cond["battery_id"] = int(m.group(1))
    missing = {"temperature", "pressure", "c_rate", "battery_id"} - cond.keys()
    if missing:
        raise ValueError(f"Could not parse {sorted(missing)} from path: {path}")
    return cond


def parse_time_str(s) -> float:
    """'h:mm:ss.fff' -> seconds (step-relative time in Neware exports)."""
    try:
        h, m, sec = str(s).split(":")
        return int(h) * 3600 + int(m) * 60 + float(sec)
    except Exception:
        return np.nan


def step_type(step: str) -> str:
    s = str(step)
    if "放电" in s:
        return "discharge"
    if "充电" in s:
        return "charge"
    if "搁置" in s or "静置" in s:
        return "rest"
    return "other"


def add_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """A new cycle starts at every charge segment that follows a non-charge row.
    Rows before the first charge belong to cycle 0 (usually none)."""
    df = df.copy()
    df["step_type"] = df["step"].map(step_type)
    is_charge = df["step_type"].eq("charge")
    start = is_charge & ~is_charge.shift(fill_value=False)
    df["cycle"] = (start.cumsum() - 1).clip(lower=0).astype(int)
    return df


def summarize_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (file_id, cycle): charge/discharge capacity in Ah, duration, rows."""
    rows = []
    for (fid, cyc), g in df.groupby(["file_id", "cycle"], sort=False):
        chg = g[g["step_type"] == "charge"]
        dis = g[g["step_type"] == "discharge"]
        rows.append({
            "file_id": fid,
            "cycle": cyc,
            "temperature": g["temperature"].iloc[0],
            "pressure": g["pressure"].iloc[0],
            "c_rate": g["c_rate"].iloc[0],
            "battery_id": g["battery_id"].iloc[0],
            "n_rows": len(g),
            "charge_capacity_Ah": chg["capacity"].max() if len(chg) else np.nan,
            "discharge_capacity_Ah": dis["capacity"].max() if len(dis) else np.nan,
            "has_full_discharge": bool(len(dis)),
            "duration_s": g["elapsed_s"].iloc[-1] - g["elapsed_s"].iloc[0],
        })
    return pd.DataFrame(rows)


def load_file(path: Path) -> pd.DataFrame:
    cond = parse_conditions(path)
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=COLUMN_MAP)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]   # drop the label columns

    need = ["step", "voltage", "current_mA", "capacity_mAh", "time_str"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{path.name}: missing columns {miss}; found {list(df.columns)}")

    df["current"] = df["current_mA"] / 1000.0           # A
    df["capacity"] = df["capacity_mAh"] / 1000.0        # Ah, accumulated WITHIN a step
    df["step_time_s"] = df["time_str"].map(parse_time_str)

    # monotonic elapsed time from the absolute timestamp (step time resets every step)
    t = pd.to_datetime(df.get("abs_time"), errors="coerce")
    if t.notna().all():
        df["elapsed_s"] = (t - t.iloc[0]).dt.total_seconds()
    else:
        df["elapsed_s"] = np.arange(len(df), dtype=float)   # fallback: 1 s per row
        print(f"   WARNING {path.name}: abs_time unusable, elapsed_s = row index")

    for k, v in cond.items():
        df[k] = v
    df["file_id"] = f"{cond['temperature']:g}d_{cond['pressure']}N_{cond['c_rate']:g}C_B{cond['battery_id']}"
    return add_cycles(df)


# ---------------------------------------------------------------- main
def run(data_path: str, out_dir: str, limit: int | None):
    base = Path(data_path)
    files = sorted(list(base.rglob("*.xls")) + list(base.rglob("*.xlsx")))
    print(f"Found {len(files)} Excel files under {base}")
    if not files:
        sys.exit("No files found - check --data-path")
    if limit:
        files = files[:limit]

    frames, bad = [], []
    for i, f in enumerate(files, 1):
        try:
            d = load_file(f)
            frames.append(d)
            print(f"[{i}/{len(files)}] {d['file_id'].iloc[0]:<22} rows={len(d):>8,} cycles={d['cycle'].nunique()}")
        except Exception as e:
            bad.append((str(f), str(e)))
            print(f"[{i}/{len(files)}] FAILED {f.name}: {e}")

    df = pd.concat(frames, ignore_index=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\nStep names found (check they are what you expect):")
    print(df.groupby(["step", "step_type"]).size().to_string())
    if (df["step_type"] == "other").any():
        print("WARNING: some steps classified as 'other' - extend step_type() if needed")

    cyc = summarize_cycles(df)
    cyc.to_csv(out / "cycle_summary.csv", index=False)
    per_file = cyc.groupby("file_id").agg(n_cycles=("cycle", "nunique"))
    print("\nCycles per file:")
    print(per_file["n_cycles"].describe().to_string())
    print(f"\nTotal rows: {len(df):,}   files: {df['file_id'].nunique()}")

    # (giant combined CSV disabled: it filled the disk; Step 2 reads the Excel files directly)
    # df.to_csv(out / "combined_battery_data.csv", index=False)
    print(f"\nSaved: {out/'cycle_summary.csv'}")
    if bad:
        pd.DataFrame(bad, columns=["file", "error"]).to_csv(out / "failed_files.csv", index=False)
        print(f"{len(bad)} files failed -> failed_files.csv")


def selftest():
    # 3 cycles: charge, rest, discharge, rest
    names = (["恒流恒压充电"] * 5 + ["搁置"] * 2 + ["恒流放电"] * 5 + ["搁置"] * 2) * 3
    df = pd.DataFrame({"step": names})
    out = add_cycles(df)
    assert out["cycle"].nunique() == 3, out["cycle"].unique()
    assert out["cycle"].tolist()[:14] == [0] * 14 and out["cycle"].iloc[14] == 1
    assert step_type("恒流放电") == "discharge" and step_type("恒流恒压充电") == "charge"
    assert parse_time_str("1:02:03.500") == 3723.5
    c = parse_conditions(Path("C:/x/libd/10d/300N/0.5C/Battery No.4.xls"))
    assert c == {"temperature": 10.0, "pressure": 300, "c_rate": 0.5, "battery_id": 4}, c
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="C:/Users/SASTRA/Videos/libd")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else run(a.data_path, a.out_dir, a.limit)
"""
RUN ORDER CHECK - recovers when each of the 288 tests was run, from the 'Absolute Time'
column in the raw Neware files, and checks whether the "pressure effects" follow the
test order (session / drift) instead of the pressure level.

Run:
  python Data_2026_RunOrder.py --data-path "C:/Users/SASTRA/Videos/libd" --outcomes out_full/per_cell_outcomes.csv --out-dir out_full/runorder
  python Data_2026_RunOrder.py --selftest

Outputs (out-dir):
  run_order.csv          one row per test: conditions, start time, run rank, outcomes
  sessions.csv           one row per temperature x pressure block: first/last start, cells, order
  drift_tests.csv        does test time explain the outcome better than pressure does?
  fig_run_order.png      capacity and resistance residuals against test date, coloured by pressure
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import SingularMatrixWarning
warnings.simplefilter("ignore", SingularMatrixWarning)   # session is nested in T x P by design

try:
    from Data_2026_1 import parse_conditions
except ImportError:
    from script1_fixed import parse_conditions

OUTCOMES = ["discharge_capacity_Ah", "dc_resistance_ohm", "mean_discharge_voltage_V",
            "energy_efficiency", "coulombic_efficiency", "discharge_energy_Wh"]


def file_id(c):
    return f"{c['temperature']:g}d_{c['pressure']}N_{c['c_rate']:g}C_B{c['battery_id']}"


def start_time(path: Path):
    """First valid timestamp in the file. Reads only the first rows (fast)."""
    df = pd.read_excel(path, nrows=20)
    df.columns = [str(c).strip() for c in df.columns]
    if "Absolute Time" not in df.columns:
        raise ValueError(f"no 'Absolute Time' column; found {list(df.columns)}")
    t = pd.to_datetime(df["Absolute Time"], errors="coerce").dropna()
    if t.empty:
        raise ValueError("Absolute Time could not be parsed")
    return t.iloc[0]


def collect(data_path):
    base = Path(data_path)
    files = sorted(list(base.rglob("*.xls")) + list(base.rglob("*.xlsx")))
    if not files:
        sys.exit("No files found - check --data-path")
    rows = []
    for i, f in enumerate(files, 1):
        try:
            c = parse_conditions(f)
            c["start"] = start_time(f)
            c["file_id"] = file_id(c)
            rows.append(c)
            print(f"[{i}/{len(files)}] {c['file_id']:<22} {c['start']}")
        except Exception as e:
            print(f"[{i}/{len(files)}] FAILED {f.name}: {e}")
    return pd.DataFrame(rows)


def analyse(ro, outcomes_csv, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ro = ro.sort_values("start").reset_index(drop=True)
    ro["run_rank"] = np.arange(1, len(ro) + 1)
    t0 = ro["start"].min()
    ro["days"] = (ro["start"] - t0).dt.total_seconds() / 86400

    if outcomes_csv and Path(outcomes_csv).exists():
        oc = pd.read_csv(outcomes_csv)
        keep = ["file_id"] + [o for o in OUTCOMES if o in oc.columns]
        ro = ro.merge(oc[keep], on="file_id", how="left")
    ro.to_csv(out / "run_order.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\nTests: {len(ro)}   first: {ro['start'].min()}   last: {ro['start'].max()}"
          f"   span: {ro['days'].max():.1f} days")

    # 1. temperature blocks
    tb = ro.groupby("temperature")["start"].agg(["min", "max"]).sort_values("min")
    print("\n=== 1. Temperature blocks, in the order they were run ===")
    print(tb.to_string())

    # 2. temperature x pressure sessions
    s = ro.groupby(["temperature", "pressure"]).agg(
        first=("start", "min"), last=("start", "max"),
        n_tests=("file_id", "size"), n_cells=("battery_id", "nunique")).reset_index()
    s = s.sort_values("first").reset_index(drop=True)
    s["session_order"] = np.arange(1, len(s) + 1)
    s["span_h"] = (s["last"] - s["first"]).dt.total_seconds() / 3600
    s.to_csv(out / "sessions.csv", index=False)
    print("\n=== 2. Temperature x pressure blocks, in run order ===")
    print(s.to_string(index=False))
    ro = ro.merge(s[["temperature", "pressure", "session_order"]], on=["temperature", "pressure"])

    # 3. were the six cells run at the same time?
    spread = ro.groupby(["temperature", "pressure", "c_rate"])["start"].agg(
        lambda x: (x.max() - x.min()).total_seconds() / 60)
    print("\n=== 3. Spread of start times across the 6 cells for the same condition (minutes) ===")
    print(spread.describe().round(1).to_string())
    simultaneous = spread.median() < 30
    print("-> cells look", "SIMULTANEOUS (tested together)" if simultaneous
          else "SEQUENTIAL (tested one after another)")

    # 4. pressure versus run order: which explains the outcome better?
    rows = []
    for y in [o for o in OUTCOMES if o in ro.columns]:
        d = ro.dropna(subset=[y])
        base = "C(battery_id) + C(temperature)*C(c_rate)"
        m0 = smf.ols(f"{y} ~ {base}", d).fit()
        mp = smf.ols(f"{y} ~ {base} + C(pressure):C(temperature)", d).fit()
        mt = smf.ols(f"{y} ~ {base} + days:C(temperature)", d).fit()
        ms = smf.ols(f"{y} ~ {base} + C(session_order)", d).fit()
        rows.append({
            "outcome": y,
            "R2_base": m0.rsquared,
            "R2_plus_pressure_byT": mp.rsquared, "AIC_pressure": mp.aic,
            "R2_plus_time_trend_byT": mt.rsquared, "AIC_time": mt.aic,
            "R2_plus_session": ms.rsquared, "AIC_session": ms.aic,
            "p_pressure_vs_base": mp.compare_f_test(m0)[1],
            "p_time_vs_base": mt.compare_f_test(m0)[1],
            "p_pressure_added_to_time": smf.ols(
                f"{y} ~ {base} + days:C(temperature) + C(pressure):C(temperature)", d
            ).fit().compare_f_test(mt)[1],
        })
    dt = pd.DataFrame(rows)
    dt.to_csv(out / "drift_tests.csv", index=False)
    print("\n=== 4. Does pressure or run time explain the outcome? (lower AIC = better) ===")
    print(dt.round(4).to_string(index=False))
    print("\nNote: with one session per temperature x pressure block, 'session' and 'pressure'"
          "\nare the same thing, so their R2 and AIC will match exactly. A smooth time trend"
          "\nthat fits as well as pressure, and leaves pressure n.s. once added, points to drift.")

    # 5. figure
    ys = [y for y in ["discharge_capacity_Ah", "dc_resistance_ohm"] if y in ro.columns]
    if ys:
        fig, ax = plt.subplots(1, len(ys), figsize=(5.5 * len(ys), 3.8), squeeze=False)
        cols = {300: "#c6dbef", 400: "#6baed6", 500: "#2171b5", 600: "#08306b"}
        for a, y in zip(ax[0], ys):
            d = ro.dropna(subset=[y])
            r = smf.ols(f"{y} ~ C(battery_id) + C(temperature)*C(c_rate)", d).fit().resid
            for P, g in d.assign(r=r).groupby("pressure"):
                a.scatter(g["start"], g["r"], s=12, color=cols.get(P, "grey"), label=f"{P} N")
            a.axhline(0, color="grey", lw=0.8)
            a.set_ylabel(f"{y} residual")
            a.set_xlabel("Test start date")
            a.tick_params(axis="x", rotation=30)
        ax[0][0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "fig_run_order.png", dpi=300)
        plt.close(fig)
    print(f"\nSaved run_order.csv, sessions.csv, drift_tests.csv, fig_run_order.png in {out}")
    return ro, s, dt


def run(data_path, outcomes_csv, out_dir):
    ro = collect(data_path)
    if ro.empty:
        sys.exit("No start times read")
    analyse(ro, outcomes_csv, out_dir)


def selftest():
    """Synthetic campaign: 25 C first, then 40 C, then 10 C; pressures run in order;
    6 cells together; capacity drifts down with time and has NO pressure effect.
    The check must find the order and prefer the time trend over pressure."""
    import tempfile
    rng = np.random.default_rng(1)
    rows, t = [], pd.Timestamp("2025-01-06 09:00")
    for T in [25, 40, 10]:
        for P in [300, 400, 500, 600]:
            for R in [0.5, 1, 1.5, 2]:
                for b in range(1, 7):
                    rows.append(dict(temperature=float(T), pressure=P, c_rate=R, battery_id=b,
                                     start=t + pd.Timedelta(minutes=b)))
                t += pd.Timedelta(hours=5)
            t += pd.Timedelta(days=1)
    ro = pd.DataFrame(rows)
    ro["file_id"] = [file_id(r) for r in ro.to_dict("records")]
    days = (ro["start"] - ro["start"].min()).dt.total_seconds() / 86400
    base = {10.0: 1.35, 25.0: 1.55, 40.0: 1.53}
    ro["discharge_capacity_Ah"] = (ro["temperature"].map(base) - 0.02 * ro["c_rate"]
                                   + 0.01 * ro["battery_id"] - 0.0015 * days
                                   + rng.normal(0, 0.003, len(ro)))
    with tempfile.TemporaryDirectory() as td:
        oc = Path(td) / "oc.csv"
        ro[["file_id", "discharge_capacity_Ah"]].to_csv(oc, index=False)
        res, s, dt = analyse(ro.drop(columns=["discharge_capacity_Ah"]), oc, Path(td) / "o")
    assert list(s.drop_duplicates("temperature")["temperature"]) == [25.0, 40.0, 10.0]
    r = dt.iloc[0]
    assert r["p_time_vs_base"] < 1e-6, r
    assert r["p_pressure_added_to_time"] > 0.01, r
    # also check the file reader on a tiny Excel file
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "40d" / "500N" / "1.5C"
        p.mkdir(parents=True)
        f = p / "Battery No.3.xlsx"
        pd.DataFrame({"Step Name": ["搁置"] * 3,
                      "Absolute Time": ["2025-02-01 10:00:00", "2025-02-01 10:00:01",
                                        "2025-02-01 10:00:02"]}).to_excel(f, index=False)
        c = collect(td)
        assert c.loc[0, "file_id"] == "40d_500N_1.5C_B3", c
        assert c.loc[0, "start"] == pd.Timestamp("2025-02-01 10:00:00")
    print("\nselftest OK (run order recovered; drift preferred over pressure; Excel reader works)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="C:/Users/SASTRA/Videos/libd")
    ap.add_argument("--outcomes", default="out_full/per_cell_outcomes.csv")
    ap.add_argument("--out-dir", default="out_full/runorder")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else run(a.data_path, a.outcomes, a.out_dir)
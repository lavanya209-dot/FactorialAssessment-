"""
STEP 3 - Three-way factorial ANOVA, with CELL AS A BLOCK
(the same 6 physical cells were run in all 48 conditions: 6 cells x 3 T x 4 P x 4 C = 288 tests,
 so observations are repeated measures on the same cells, not 288 independent cells)

Run:
  python step3_anova.py --input out_full/per_cell_outcomes.csv --out-dir out_full/anova
  python step3_anova.py --selftest
Needs: pip install pandas numpy scipy statsmodels matplotlib

Outputs (in out-dir):
  anova_<outcome>.csv      one table per outcome: SS, df, F, p, eta^2, partial eta^2
  anova_summary.csv        % variance explained per term and outcome
  assumptions.csv          Shapiro-Wilk (residual normality) and Levene (equal variances)
  tukey_pressure_<outcome>.csv   pairwise pressure comparisons
  fig_TxC_interaction.png, fig_pressure_effect.png, fig_variance_explained.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats

OUTCOMES = ["discharge_capacity_Ah", "discharge_energy_Wh", "mean_discharge_voltage_V",
            "coulombic_efficiency", "energy_efficiency", "dc_resistance_ohm"]
TERMS = {"C(cell)": "Cell (block)", "C(T)": "Temperature", "C(P)": "Pressure", "C(R)": "C-rate",
         "C(T):C(P)": "Temp x Pressure", "C(T):C(R)": "Temp x C-rate",
         "C(P):C(R)": "Pressure x C-rate", "C(T):C(P):C(R)": "Temp x P x C"}


def prep(df):
    df = df[df["usable"].astype(bool)].copy()
    df["T"], df["P"], df["R"] = df["temperature"], df["pressure"], df["c_rate"]
    df["cell"] = df["battery_id"]
    return df


def anova_one(df, y):
    d = df.dropna(subset=[y])
    model = smf.ols(f"{y} ~ C(cell) + C(T) * C(P) * C(R)", data=d).fit()
    # balanced design -> type I = II = III
    a = sm.stats.anova_lm(model, typ=2)
    ss_tot = a["sum_sq"].sum()
    a["eta_sq_%"] = 100 * a["sum_sq"] / ss_tot
    ss_err = a.loc["Residual", "sum_sq"]
    a["partial_eta_sq"] = a["sum_sq"] / (a["sum_sq"] + ss_err)
    a.index = [TERMS.get(i, i) for i in a.index]
    return a, model, d


def run(inp, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = prep(pd.read_csv(inp))
    print(f"Cells used: {len(df)}   design counts per condition: "
          f"{sorted(df.groupby(['T','P','R']).size().unique())}")
    summary, assum = {}, []
    models = {}
    for y in [o for o in OUTCOMES if o in df.columns]:
        a, m, d = anova_one(df, y)
        a.round(6).to_csv(out / f"anova_{y}.csv")
        summary[y] = a["eta_sq_%"]
        models[y] = (m, d)
        resid = m.resid
        sh = stats.shapiro(resid)
        groups = [g[y].values for _, g in d.groupby(["T", "P", "R"])]
        lev = stats.levene(*groups)
        assum.append({"outcome": y, "shapiro_W": sh.statistic, "shapiro_p": sh.pvalue,
                      "levene_stat": lev.statistic, "levene_p": lev.pvalue})
        print(f"\n=== {y} ===")
        print(a[["df", "sum_sq", "F", "PR(>F)", "eta_sq_%", "partial_eta_sq"]].round(4).to_string())
    pd.DataFrame(summary).round(3).to_csv(out / "anova_summary.csv")
    pd.DataFrame(assum).round(4).to_csv(out / "assumptions.csv", index=False)
    print("\nAssumption checks:\n", pd.DataFrame(assum).round(4).to_string(index=False))

    # Tukey on pressure for the main outcome
    y0 = "discharge_capacity_Ah"
    if y0 in df.columns:
        tk = pairwise_tukeyhsd(df[y0], df["P"].astype(str))
        pd.DataFrame(tk._results_table.data[1:], columns=tk._results_table.data[0]).to_csv(
            out / f"tukey_pressure_{y0}.csv", index=False)
        print("\nTukey, pressure levels (discharge capacity):\n", tk)

    # pressure effect measured WITHIN each cell (600 N minus 300 N, for every T x C-rate)
    if y0 in df.columns:
        p = df.pivot_table(index=["cell", "T", "R"], columns="P", values=y0)
        diff = (p[600] - p[300]).reset_index(name="d")
        percell = diff.groupby("cell")["d"].mean()
        t = stats.ttest_1samp(percell, 0)
        rep = {"mean_600N_minus_300N": diff["d"].mean(),
               "pct_of_capacity": 100 * diff["d"].mean() / p[300].mean(),
               "n_negative_pairs": int((diff["d"] < 0).sum()), "n_pairs": len(diff),
               "per_cell_ttest_p": t.pvalue}
        pd.Series(rep).to_csv(out / "pressure_within_cell.csv")
        print("\nPressure, within-cell (600 N - 300 N):", {k: round(float(v), 4) for k, v in rep.items()})

    # ---- figures
    if y0 in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for T, g in df.groupby("T"):
            m = g.groupby("R")[y0].agg(["mean", "std"]).reset_index()
            ax.errorbar(m["R"], m["mean"], yerr=m["std"], marker="o", capsize=3, label=f"{T:g} °C")
        ax.set_xlabel("C-rate"); ax.set_ylabel("Discharge capacity (Ah)")
        ax.set_title("Temperature × C-rate interaction (mean ± SD, n=24)")
        ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
        fig.savefig(out / "fig_TxC_interaction.png", dpi=300); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4.2))
        for T, g in df.groupby("T"):
            m = g.groupby("P")[y0].agg(["mean", "std"]).reset_index()
            ax.errorbar(m["P"], m["mean"], yerr=m["std"], marker="s", capsize=3, label=f"{T:g} °C")
        ax.set_xlabel("External pressure (N)"); ax.set_ylabel("Discharge capacity (Ah)")
        ax.set_title("Pressure effect (mean ± SD across C-rates and replicates)")
        ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
        fig.savefig(out / "fig_pressure_effect.png", dpi=300); plt.close(fig)

        s = pd.DataFrame(summary)[y0].drop("Residual", errors="ignore").sort_values()
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.barh(s.index, s.values)
        ax.set_xlabel("Variance explained (%)"); ax.set_title("Discharge capacity: variance explained")
        fig.tight_layout(); fig.savefig(out / "fig_variance_explained.png", dpi=300); plt.close(fig)
    print(f"\nSaved tables and figures in {out}")


def selftest():
    import tempfile
    rng = np.random.default_rng(0)
    rows = []
    for T, tb in zip([10, 25, 40], [1.0, 1.4, 1.3]):
        for P in [300, 400, 500, 600]:
            for R in [0.5, 1, 1.5, 2]:
                for b in range(1, 7):
                    mu = tb - 0.1 * R * (1.5 if T == 10 else 0.4) + 0.02 * b   # T x C interaction + cell offset, NO pressure effect
                    rows.append(dict(file_id=f"{T}_{P}_{R}_{b}", temperature=T, pressure=P, c_rate=R,
                                     battery_id=b, usable=True,
                                     discharge_capacity_Ah=mu + rng.normal(0, 0.01)))
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        run(p, Path(td) / "out")
    a, _, _ = anova_one(prep(pd.DataFrame(rows)), "discharge_capacity_Ah")
    assert a.loc["Pressure", "PR(>F)"] > 0.01 and a.loc["Temperature", "PR(>F)"] < 1e-10
    assert a.loc["Temp x C-rate", "PR(>F)"] < 1e-10
    print("\nselftest OK (pressure n.s., T and TxC significant, as built)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="out_full/per_cell_outcomes.csv")
    ap.add_argument("--out-dir", default="out_full/anova")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else run(a.input, a.out_dir)
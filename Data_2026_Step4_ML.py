"""
STEP 4 (Part A) - Surrogate models for the per-cell outcomes, validated honestly

Predicts each outcome (capacity, energy, mean voltage, efficiencies, 10 s resistance) from the test
conditions: temperature, C-rate and (as an ablation) pressure. 288 rows = 48 conditions x 6 cells.

What it does
  1. Noise floor from the cell-blocked ANOVA: no model that has not seen a cell can have RMSE below
       sqrt(sigma_cell^2 + sigma_resid^2); if the cell is known the floor is sigma_resid.
  2. Baselines + models: condition mean, polynomial ridge, Gaussian process, gradient boosting.
  3. Validation: leave-one-cell-out (main), leave-one-temperature-out, leave-one-C-rate-out
     (extrapolation), plus a random 5-fold run to show what knowing the cell would buy.
  4. Pressure ablation (with vs without pressure as an input) and permutation importance.
  5. Sanity check: shuffled labels must give R2 <= 0.

Run:
  python -m pip install scikit-learn
  python Data_2026_Step4_ML.py --input out_full/per_cell_outcomes.csv --out-dir out_full/ml
  python Data_2026_Step4_ML.py --selftest

Outputs (out-dir): ml_noise_floor.csv, ml_validation_results.csv, ml_pressure_ablation.csv,
                   ml_permutation_importance.csv, fig_ml_capacity_loco.png
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as CK
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

OUTCOMES = ["discharge_capacity_Ah", "discharge_energy_Wh", "mean_discharge_voltage_V",
            "coulombic_efficiency", "energy_efficiency", "dc_resistance_ohm"]
SCHEMES = ["leave_one_cell_out", "leave_one_temperature_out", "leave_one_crate_out"]
MODEL_SPECS = [("condmean", None), ("linear", "TC"), ("linear", "TCP"), ("gp", "TC"),
               ("gp", "TCP"), ("gb", "TC"), ("gb", "TCP")]
RANDOM_SPECS = [("condmean", None), ("linear", "TC"), ("linear", "TCB"), ("gb", "TCB")]
MAIN = "leave_one_cell_out"
CAP = "discharge_capacity_Ah"


# ---------------------------------------------------------------- features / models
def features(df, fs):
    X = df[["temperature", "c_rate"]].astype(float).copy()
    if fs == "TCP":
        X["pressure"] = df["pressure"].astype(float)
    if fs == "TCB":                                   # cell identity (one-hot): "cell is known"
        for b in sorted(df["battery_id"].unique()):
            X[f"cell{b}"] = (df["battery_id"] == b).astype(float)
    return X


def make_model(name, n_feat):
    if name == "linear":
        return make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                             Ridge(alpha=1e-3))
    if name == "gp":
        k = CK(1.0, (1e-3, 1e3)) * RBF([1.0] * n_feat, (0.1, 100.0)) + WhiteKernel(0.1, (1e-6, 1.0))
        return make_pipeline(StandardScaler(),
                             GaussianProcessRegressor(k, normalize_y=True,
                                                      n_restarts_optimizer=1, random_state=0))
    if name == "gb":
        return GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=2,
                                         subsample=0.8, random_state=0)
    raise ValueError(name)


def cond_mean_predict(train, test, y):
    """Mean per (temperature, C-rate); if that condition is unseen fall back to mean per C-rate,
    then per temperature, then the grand mean."""
    m_tr = dict(train.groupby(["temperature", "c_rate"])[y].mean().items())
    m_r = train.groupby("c_rate")[y].mean().to_dict()
    m_t = train.groupby("temperature")[y].mean().to_dict()
    gm = train[y].mean()
    return np.array([m_tr.get((T, R), m_r.get(R, m_t.get(T, gm)))
                     for T, R in zip(test["temperature"], test["c_rate"])], float)


def get_splits(d, scheme):
    if scheme == "leave_one_cell_out":
        return list(LeaveOneGroupOut().split(d, groups=d["battery_id"]))
    if scheme == "leave_one_temperature_out":
        return list(LeaveOneGroupOut().split(d, groups=d["temperature"]))
    if scheme == "leave_one_crate_out":
        return list(LeaveOneGroupOut().split(d, groups=d["c_rate"]))
    if scheme == "random_5fold":
        return list(KFold(5, shuffle=True, random_state=0).split(d))
    raise ValueError(scheme)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def evaluate(d, y, splits, name, fs):
    pred = np.full(len(d), np.nan)
    folds = []
    X = None if name == "condmean" else features(d, fs)
    for tr, te in splits:
        if name == "condmean":
            p = cond_mean_predict(d.iloc[tr], d.iloc[te], y)
        else:
            m = make_model(name, X.shape[1])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(X.iloc[tr], d[y].iloc[tr])
            p = m.predict(X.iloc[te])
        pred[te] = p
        folds.append(rmse(d[y].iloc[te], p))
    return pred, folds


# ---------------------------------------------------------------- noise floor
def noise_floor(d, y):
    m = smf.ols(f"{y} ~ C(battery_id) + C(temperature)*C(pressure)*C(c_rate)", data=d).fit()
    a = sm.stats.anova_lm(m, typ=2)
    ms_cell = a.loc["C(battery_id)", "sum_sq"] / a.loc["C(battery_id)", "df"]
    ms_res = a.loc["Residual", "sum_sq"] / a.loc["Residual", "df"]
    n_per_cell = len(d) / d["battery_id"].nunique()
    var_cell = max((ms_cell - ms_res) / n_per_cell, 0.0)
    mean_y = d[y].mean()
    return {"outcome": y, "mean_y": mean_y,
            "sigma_resid": np.sqrt(ms_res), "sigma_cell": np.sqrt(var_cell),
            "floor_cell_known": np.sqrt(ms_res),
            "floor_cell_unseen": np.sqrt(var_cell + ms_res),
            "floor_unseen_pct_of_mean": 100 * np.sqrt(var_cell + ms_res) / mean_y}


# ---------------------------------------------------------------- main
def run(inp, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(inp)
    df = df[df["usable"].astype(bool)].reset_index(drop=True)
    outcomes = [o for o in OUTCOMES if o in df.columns]
    n_cells = df["battery_id"].nunique()
    print(f"Rows: {len(df)}   cells (battery_id): {n_cells}   outcomes: {outcomes}")

    floors, rows, imp_rows, keep = [], [], [], {}
    rng = np.random.default_rng(0)
    for y in outcomes:
        d = df.dropna(subset=[y]).reset_index(drop=True)
        floors.append(noise_floor(d, y))
        ymean = d[y].mean()
        print(f"\n--- {y} ---")
        for scheme in SCHEMES + ["random_5fold"]:
            splits = get_splits(d, scheme)
            specs = RANDOM_SPECS if scheme == "random_5fold" else MODEL_SPECS
            base_rmse, base_folds = None, None
            for name, fs in specs:
                pred, folds = evaluate(d, y, splits, name, fs)
                r = rmse(d[y], pred)
                if name == "condmean":
                    base_rmse, base_folds = r, folds
                rows.append({"outcome": y, "scheme": scheme, "model": name,
                             "features": fs or "-", "rmse": r,
                             "mae": float(np.mean(np.abs(d[y] - pred))),
                             "r2": r2_score(d[y], pred), "rel_rmse_pct": 100 * r / ymean,
                             "fold_rmse_mean": np.mean(folds), "fold_rmse_sd": np.std(folds, ddof=1),
                             "n_folds": len(folds),
                             "skill_vs_condmean": 1 - r / base_rmse,
                             "folds_better_than_condmean": int(sum(f < b for f, b in zip(folds, base_folds)))})
                if scheme == MAIN and y == CAP and (name, fs) == ("gp", "TC"):
                    keep["pred"] = (d.copy(), pred)
        # sanity: shuffled labels
        ds = d.copy()
        ds[y] = rng.permutation(ds[y].values)
        pred, folds = evaluate(ds, y, get_splits(ds, MAIN), "linear", "TC")
        rows.append({"outcome": y, "scheme": MAIN, "model": "linear_SHUFFLED_LABELS", "features": "TC",
                     "rmse": rmse(ds[y], pred), "mae": float(np.mean(np.abs(ds[y] - pred))),
                     "r2": r2_score(ds[y], pred), "rel_rmse_pct": 100 * rmse(ds[y], pred) / ymean,
                     "fold_rmse_mean": np.mean(folds), "fold_rmse_sd": np.std(folds, ddof=1),
                     "n_folds": len(folds), "skill_vs_condmean": np.nan,
                     "folds_better_than_condmean": np.nan})
        # permutation importance (leave-one-cell-out)
        splits = get_splits(d, MAIN)
        X = features(d, "TCP")
        for name in ("gb", "gp"):
            acc = []
            for tr, te in splits:
                m = make_model(name, X.shape[1])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.fit(X.iloc[tr], d[y].iloc[tr])
                    pi = permutation_importance(m, X.iloc[te], d[y].iloc[te], n_repeats=10,
                                                scoring="neg_root_mean_squared_error", random_state=0)
                acc.append(pi.importances_mean)
            for f, v in zip(X.columns, np.mean(acc, axis=0)):
                imp_rows.append({"outcome": y, "model": name, "feature": f, "rmse_increase": v,
                                 "rmse_increase_pct_of_mean": 100 * v / ymean})

    fl = pd.DataFrame(floors)
    res = pd.DataFrame(rows)
    imp = pd.DataFrame(imp_rows)
    fl.round(6).to_csv(out / "ml_noise_floor.csv", index=False)
    res.round(6).to_csv(out / "ml_validation_results.csv", index=False)
    imp.round(6).to_csv(out / "ml_permutation_importance.csv", index=False)

    # pressure ablation (main scheme)
    m = res[(res.scheme == MAIN) & res.model.isin(["linear", "gp", "gb"])]
    pv = m.pivot_table(index=["outcome", "model"], columns="features", values="rmse")
    pv["pressure_changes_rmse_pct"] = 100 * (pv["TCP"] - pv["TC"]) / pv["TC"]
    pv.round(6).to_csv(out / "ml_pressure_ablation.csv")

    # ---- console summary
    pd.set_option("display.width", 200)
    print("\n=== Noise floor (RMSE no model can beat, from the cell-blocked ANOVA) ===")
    print(fl.round(4).to_string(index=False))
    main = res[res.scheme == MAIN].copy()
    main["spec"] = main.model + "_" + main.features
    print("\n=== Leave-one-cell-out: RMSE as % of the outcome mean ===")
    print(main.pivot_table(index="outcome", columns="spec", values="rel_rmse_pct").round(2).to_string())
    print("\n=== Leave-one-cell-out: skill vs condition-mean baseline (positive = better) ===")
    print(main.pivot_table(index="outcome", columns="spec", values="skill_vs_condmean").round(3).to_string())
    print("\n=== Does pressure as an input change the error? (% change in RMSE, TCP vs TC) ===")
    print(pv["pressure_changes_rmse_pct"].round(2).unstack().to_string())
    for sch in ["leave_one_temperature_out", "leave_one_crate_out"]:
        s = res[res.scheme == sch].copy()
        s["spec"] = s.model + "_" + s.features
        print(f"\n=== {sch}: RMSE as % of the outcome mean ===")
        print(s.pivot_table(index="outcome", columns="spec", values="rel_rmse_pct").round(2).to_string())
    rnd = res[res.scheme == "random_5fold"].copy()
    rnd["spec"] = rnd.model + "_" + rnd.features
    print("\n=== Random 5-fold (TCB = cell identity given) vs leave-one-cell-out linear_TC: RMSE % of mean ===")
    cmp_ = rnd.pivot_table(index="outcome", columns="spec", values="rel_rmse_pct")
    cmp_["LOCO_linear_TC"] = main[main.spec == "linear_TC"].set_index("outcome")["rel_rmse_pct"]
    print(cmp_.round(2).to_string())
    print("\n=== Shuffled labels (R2 should be <= 0) ===")
    print(res[res.model == "linear_SHUFFLED_LABELS"][["outcome", "r2"]].round(3).to_string(index=False))
    print("\n=== Permutation importance, leave-one-cell-out (RMSE increase, % of mean; gb / gp) ===")
    print(imp.pivot_table(index=["outcome", "model"], columns="feature",
                          values="rmse_increase_pct_of_mean").round(2).to_string())

    # ---- figure: observed vs predicted + RMSE vs floor (capacity)
    if "pred" in keep:
        d, pred = keep["pred"]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
        for T, g in d.assign(pred=pred).groupby("temperature"):
            ax[0].scatter(g[CAP], g["pred"], s=14, alpha=.7, label=f"{T:g} °C")
        lo, hi = d[CAP].min(), d[CAP].max()
        ax[0].plot([lo, hi], [lo, hi], "k--", lw=1)
        ax[0].set_xlabel("Observed capacity (Ah)")
        ax[0].set_ylabel("Predicted, held-out cell (Ah)")
        ax[0].set_title("Gaussian process, leave-one-cell-out")
        ax[0].legend(); ax[0].grid(alpha=.3)
        c = main[main.outcome == CAP]
        ax[1].bar(c["spec"], c["rmse"])
        f = fl[fl.outcome == CAP].iloc[0]
        ax[1].axhline(f["floor_cell_unseen"], color="r", ls="--", label="noise floor, unseen cell")
        ax[1].axhline(f["floor_cell_known"], color="g", ls=":", label="noise floor, known cell")
        ax[1].set_ylabel("RMSE (Ah)")
        ax[1].set_title("Capacity: leave-one-cell-out RMSE")
        ax[1].tick_params(axis="x", rotation=60)
        ax[1].legend(); ax[1].grid(alpha=.3, axis="y")
        fig.tight_layout()
        fig.savefig(out / "fig_ml_capacity_loco.png", dpi=300)
        plt.close(fig)
    print(f"\nSaved tables and figure in {out}")
    return fl, res


# ---------------------------------------------------------------- selftest
def selftest():
    import tempfile
    rng = np.random.default_rng(0)
    cell_eff = rng.normal(0, 0.04, 6)
    rows = []
    for T, tb in zip([10, 25, 40], [1.0, 1.4, 1.3]):
        for P in [300, 400, 500, 600]:
            for R in [0.5, 1, 1.5, 2]:
                for b in range(1, 7):
                    mu = tb - 0.1 * R * (1.5 if T == 10 else 0.4) - 0.00002 * (P - 300) + cell_eff[b - 1]
                    rows.append(dict(file_id=f"{T}_{P}_{R}_{b}", temperature=T, pressure=P, c_rate=R,
                                     battery_id=b, usable=True,
                                     discharge_capacity_Ah=mu + rng.normal(0, 0.01)))
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        fl, res = run(p, Path(td) / "ml")
    f = fl.iloc[0]
    g = lambda scheme, model, fs: res[(res.scheme == scheme) & (res.model == model)
                                      & (res.features == fs)].iloc[0]
    loco = g(MAIN, "linear", "TC")
    known = g("random_5fold", "linear", "TCB")
    shuf = res[res.model == "linear_SHUFFLED_LABELS"].iloc[0]
    assert f["floor_cell_unseen"] > f["floor_cell_known"]
    assert loco["rmse"] < 1.3 * f["floor_cell_unseen"], (loco["rmse"], f["floor_cell_unseen"])
    assert known["rmse"] < 0.8 * loco["rmse"], (known["rmse"], loco["rmse"])
    assert shuf["r2"] < 0.1, shuf["r2"]
    print("\nselftest OK (LOCO error near the ANOVA floor; knowing the cell lowers it; shuffled labels fail)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="out_full/per_cell_outcomes.csv")
    ap.add_argument("--out-dir", default="out_full/ml")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else run(a.input, a.out_dir)
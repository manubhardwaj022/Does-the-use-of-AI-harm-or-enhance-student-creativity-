"""CSET485 Assignment #1 - German Credit Data analysis.
Student: Manu Bhardwaj | Enrollment: S24BCAU0034

Run from the repository root with: python cset485_analysis.py
The original UCI data are stored in data/german.data. If absent, the script
downloads the copy committed to this repository.
"""
from __future__ import annotations
import json
import math
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "german.data"
OUTPUT_DIR = ROOT / "outputs"
RAW_URL = "https://raw.githubusercontent.com/manubhardwaj022/Does-the-use-of-AI-harm-or-enhance-student-creativity-/main/data/german.data"
RANDOM_STATE = 34
TEST_SIZE = 0.30

COLUMNS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment_since", "installment_rate", "personal_status_sex",
    "other_debtors", "residence_since", "property", "age", "other_installment_plans",
    "housing", "existing_credits", "job", "people_liable", "telephone",
    "foreign_worker", "class",
]
NUMERIC = ["duration", "credit_amount", "installment_rate", "residence_since",
           "age", "existing_credits", "people_liable"]
CATEGORICAL = [c for c in COLUMNS if c not in NUMERIC and c != "class"]


def load_data() -> pd.DataFrame:
    """Load the original 21-column whitespace-delimited UCI data file."""
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RAW_URL, DATA_PATH)
    df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, names=COLUMNS,
                     dtype={c: "string" for c in CATEGORICAL + ["class"]})
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df.shape != (1000, 21):
        raise ValueError(f"Expected 1000 rows and 21 columns, found {df.shape}")
    if (df == "?").any().any() or df.isna().any().any():
        raise ValueError("Unexpected missing values found; inspect before modelling.")
    # UCI target: 1=Good, 2=Bad. The course brief asks for 1=approve, 0=reject.
    df["approval"] = (df["class"].astype(int) == 1).astype(int)
    # UCI personal_status_sex codes: A92/A95 are female; remaining levels are male.
    df["gender_group"] = np.where(
        df["personal_status_sex"].isin(["A92", "A95"]), "Female", "Male"
    )
    return df


def train_test_indices(n: int):
    """70/30 shuffled split, matching sklearn train_test_split(random_state=34)."""
    from sklearn.model_selection import train_test_split
    return train_test_split(
        np.arange(n), test_size=TEST_SIZE, random_state=RANDOM_STATE,
        shuffle=True, stratify=None
    )


def design_matrices(train: pd.DataFrame, test: pd.DataFrame):
    """One-hot encode categoricals and standardise every design column except the intercept using training-only statistics."""
    levels = {
        c: sorted(train[c].dropna().astype(str).unique().tolist())
        for c in CATEGORICAL
    }
    feature_names = ["intercept"] + NUMERIC
    for c in CATEGORICAL:
        feature_names.extend([f"{c}_{level}" for level in levels[c][1:]])

    def raw_matrix(frame):
        arrays = [np.ones(len(frame))]
        arrays.extend(frame[c].astype(float).to_numpy() for c in NUMERIC)
        for c in CATEGORICAL:
            values = frame[c].astype(str).to_numpy()
            for level in levels[c][1:]:
                arrays.append((values == level).astype(float))
        return np.column_stack(arrays)

    raw_train, raw_test = raw_matrix(train), raw_matrix(test)
    means = raw_train[:, 1:].mean(axis=0)
    stds = raw_train[:, 1:].std(axis=0, ddof=0)
    stds[stds == 0] = 1.0

    def scale(raw):
        z = raw.copy().astype(float)
        z[:, 1:] = (z[:, 1:] - means) / stds
        return z

    return scale(raw_train), scale(raw_test), feature_names


def fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ordinary least squares, with a tiny numerical stabiliser."""
    gram = X.T @ X
    gram[1:, 1:] += np.eye(X.shape[1] - 1) * 1e-8
    return np.linalg.solve(gram, X.T @ y)


def fit_logistic_newton(X: np.ndarray, y: np.ndarray, max_iter: int = 60):
    """Logistic maximum-likelihood fit by Newton/IRLS updates."""
    beta = np.zeros(X.shape[1], dtype=float)
    hessian = None
    for _ in range(max_iter):
        z = np.clip(X @ beta, -25, 25)
        prob = 1.0 / (1.0 + np.exp(-z))
        weight = prob * (1.0 - prob)
        gradient = X.T @ (y - prob)
        hessian = (X.T * weight) @ X
        gradient[1:] -= 1e-7 * beta[1:]
        hessian[1:, 1:] += np.eye(X.shape[1] - 1) * 1e-7
        hessian[0, 0] += 1e-9
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.max(np.abs(step)) < 1e-7:
            break
    return beta, hessian


def wald_summary(beta: float, se: float):
    z = beta / se if se else float("nan")
    p = math.erfc(abs(z) / math.sqrt(2)) if math.isfinite(z) else float("nan")
    return {
        "coef": float(beta), "se": float(se),
        "ci_low": float(beta - 1.96 * se),
        "ci_high": float(beta + 1.96 * se),
        "z": float(z), "p_approx": float(p),
    }


def fit_numeric_submodel(frame: pd.DataFrame, y: np.ndarray):
    raw = np.column_stack(
        [np.ones(len(frame))] + [frame[c].astype(float).to_numpy() for c in NUMERIC]
    )
    means = raw[:, 1:].mean(axis=0)
    stds = raw[:, 1:].std(axis=0, ddof=0)
    stds[stds == 0] = 1.0
    X = raw.copy()
    X[:, 1:] = (X[:, 1:] - means) / stds
    beta, hessian = fit_logistic_newton(X, y)
    se = np.sqrt(np.maximum(0, np.diag(np.linalg.inv(hessian))))
    return beta, se


def ratio(a, b):
    """Directional rate ratio; undefined 0/0 is NaN, positive/0 is infinity."""
    if b == 0:
        return float("inf") if a > 0 else float("nan")
    return float(a / b)


def group_metrics(y_true, prob, group, threshold: float):
    stats = {}
    for label in ["Male", "Female"]:
        mask = group == label
        actual = y_true[mask]
        pred = (prob[mask] >= threshold).astype(int)
        tp = int(((pred == 1) & (actual == 1)).sum())
        fp = int(((pred == 1) & (actual == 0)).sum())
        fn = int(((pred == 0) & (actual == 1)).sum())
        tn = int(((pred == 0) & (actual == 0)).sum())
        stats[label] = {
            "n": int(mask.sum()),
            "approval_rate": float(pred.mean()),
            "tpr": ratio(tp, tp + fn),
            "fpr": ratio(fp, fp + tn),
            "ppv": ratio(tp, tp + fp),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
    A, B = stats["Male"], stats["Female"]
    dp = ratio(A["approval_rate"], B["approval_rate"])
    tpr_ratio = ratio(A["tpr"], B["tpr"])
    fpr_ratio = ratio(A["fpr"], B["fpr"])
    pp = ratio(A["ppv"], B["ppv"])
    # Assignment EO = min(TPR_A/TPR_B, FPR_A/FPR_B). Skip only undefined 0/0.
    comparable = [v for v in (tpr_ratio, fpr_ratio) if not math.isnan(v)]
    eo = min(comparable) if comparable else float("nan")
    return {
        "threshold": float(threshold), "DP": dp, "EO": eo, "PP": pp,
        "TPR_ratio": tpr_ratio if math.isfinite(tpr_ratio) else None,
        "FPR_ratio": fpr_ratio if math.isfinite(fpr_ratio) else None,
        "male": A, "female": B,
    }


def run_analysis():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    train_idx, test_idx = train_test_indices(len(df))
    train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    y_train = train["approval"].to_numpy(dtype=int)
    y_test = test["approval"].to_numpy(dtype=int)
    X_train, X_test, feature_names = design_matrices(train, test)

    # Part 1: OLS linear probability model and logistic regression.
    beta_ols = fit_ols(X_train, y_train)
    beta_log, _ = fit_logistic_newton(X_train, y_train)
    pred_linear = X_test @ beta_ols
    pred_log = 1 / (1 + np.exp(-np.clip(X_test @ beta_log, -25, 25)))
    median_linear = float(np.median(pred_linear))
    class_linear = (pred_linear >= median_linear).astype(int)
    class_log = (pred_log >= 0.5).astype(int)
    cm = np.zeros((2, 2), dtype=int)
    for linear_label, log_label in zip(class_linear, class_log):
        cm[linear_label, log_label] += 1
    mismatch = int((class_linear != class_log).sum())
    accuracy_05 = float((class_log == y_test).mean())

    # Part 2.1: same seven numeric predictors in full, male and female models.
    all_beta, all_se = fit_numeric_submodel(train, y_train)
    male_mask = train["gender_group"].to_numpy() == "Male"
    female_mask = train["gender_group"].to_numpy() == "Female"
    male_beta, male_se = fit_numeric_submodel(train.loc[male_mask], y_train[male_mask])
    female_beta, female_se = fit_numeric_submodel(train.loc[female_mask], y_train[female_mask])
    credit_idx = NUMERIC.index("credit_amount") + 1
    subgroup = {
        "all": wald_summary(all_beta[credit_idx], all_se[credit_idx]),
        "male": wald_summary(male_beta[credit_idx], male_se[credit_idx]),
        "female": wald_summary(female_beta[credit_idx], female_se[credit_idx]),
        "n": {"all": len(train), "male": int(male_mask.sum()), "female": int(female_mask.sum())},
    }

    # Part 2.2: omit checking_status and compare four remaining feature coefficients.
    drop_positions = [i for i, name in enumerate(feature_names)
                      if name.startswith("checking_status_")]
    keep_positions = [i for i in range(len(feature_names)) if i not in drop_positions]
    reduced_names = [feature_names[i] for i in keep_positions]
    beta_reduced, _ = fit_logistic_newton(X_train[:, keep_positions], y_train)
    shifts = []
    for name in ["duration", "credit_amount", "installment_rate", "age"]:
        full_coef = float(beta_log[feature_names.index(name)])
        reduced_coef = float(beta_reduced[reduced_names.index(name)])
        shifts.append({
            "feature": name, "full": full_coef, "reduced": reduced_coef,
            "percent_change": float((reduced_coef - full_coef) / abs(full_coef) * 100),
            "absolute_shift": abs(reduced_coef - full_coef),
        })

    # Part 2.3: confident wrong predictions; retain all if fewer than 50.
    hard_mask = ((pred_log >= 0.75) & (y_test == 0)) | ((pred_log <= 0.25) & (y_test == 1))
    hard_ids = np.flatnonzero(hard_mask)
    if len(hard_ids) > 50:
        hard_ids = np.random.RandomState(RANDOM_STATE).choice(hard_ids, size=50, replace=False)
    adversarial = test.iloc[hard_ids].copy()
    adversarial.insert(0, "test_row_position", hard_ids)
    adversarial["predicted_approval_probability"] = pred_log[hard_ids]
    adversarial["actual_approval"] = y_test[hard_ids]
    adversarial["prediction_type"] = np.where(
        (pred_log[hard_ids] >= 0.75) & (y_test[hard_ids] == 0),
        "False positive: predicted approve, actual bad risk",
        "False negative: predicted reject, actual good risk",
    )
    adversarial.to_csv(OUTPUT_DIR / "adversarial_high_confidence_errors.csv", index=False)

    def trait_pct(frame, condition):
        return float(condition(frame).mean() * 100) if len(frame) else float("nan")
    adv_trait_rows = [
        ("Single male status (A93)", lambda d: d["personal_status_sex"] == "A93"),
        ("Checking account below 0 DM (A11)", lambda d: d["checking_status"] == "A11"),
        ("Credit amount above 5000", lambda d: d["credit_amount"] > 5000),
        ("Duration above 24 months", lambda d: d["duration"] > 24),
        ("Age 40 or above", lambda d: d["age"] >= 40),
    ]
    trait_rows = [
        {"characteristic": name, "adversarial_pct": trait_pct(adversarial, cond),
         "test_pct": trait_pct(test, cond)}
        for name, cond in adv_trait_rows
    ]

    # Part 2.4: thresholds exactly as specified by the course assignment.
    groups = test["gender_group"].to_numpy()
    thresholds = [round(0.05 + 0.10 * i, 2) for i in range(10)]
    fairness = [group_metrics(y_test, pred_log, groups, t) for t in thresholds]
    fairness_csv = pd.DataFrame([
        {k: row[k] for k in ["threshold", "DP", "EO", "PP", "TPR_ratio", "FPR_ratio"]}
        for row in fairness
    ])
    fairness_csv.to_csv(OUTPUT_DIR / "fairness_threshold_metrics.csv", index=False)

    # Trade-off chart: both axes fixed to 0-1.5 as requested.
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    label_offsets = [(-18, -15), (2, -14), (15, 7), (24, -3), (25, 12),
                     (-25, -13), (-24, -3), (12, -15), (-14, 15), (0, 0)]
    for row, offset in zip(fairness, label_offsets):
        if math.isfinite(row["DP"]) and math.isfinite(row["EO"]):
            if 0 <= row["DP"] <= 1.5 and 0 <= row["EO"] <= 1.5:
                ax.scatter(row["DP"], row["EO"], s=48)
                ax.annotate(
                    f"{row['threshold']:.2f}", (row["DP"], row["EO"]),
                    xytext=offset, textcoords="offset points", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
                )
    ax.axvline(1.0, linestyle="--", linewidth=0.8)
    ax.axhline(1.0, linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 1.5)
    ax.set_xlabel("Demographic Parity ratio (Male / Female)")
    ax.set_ylabel("Equalized Odds summary ratio (per assignment formula)")
    ax.set_title("German Credit fairness trade-off by decision threshold")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fairness_tradeoff_chart.png", dpi=180)
    plt.close(fig)

    # Details used in the report's threshold examples, including the UCI 5:1 cost.
    threshold_details = {}
    for t in [0.15, 0.35, 0.55, 0.85]:
        row = next(x for x in fairness if abs(x["threshold"] - t) < 1e-9)
        correct = sum(row[g]["tp"] + row[g]["tn"] for g in ["male", "female"])
        fp = row["male"]["fp"] + row["female"]["fp"]
        fn = row["male"]["fn"] + row["female"]["fn"]
        threshold_details[str(t)] = {
            "accuracy": correct / len(test), "fp": fp, "fn": fn,
            "weighted_cost": 5 * fp + fn, "good_rejected": fn,
            "approval_male": row["male"]["approval_rate"],
            "approval_female": row["female"]["approval_rate"],
            "male_tpr": row["male"]["tpr"], "female_tpr": row["female"]["tpr"],
            "male_fpr": row["male"]["fpr"], "female_fpr": row["female"]["fpr"],
        }

    # IQR outlier audit: flag and retain records rather than deleting genuine profiles.
    iqr_audit = {}
    for col in NUMERIC:
        q1, q3 = float(df[col].quantile(0.25)), float(df[col].quantile(0.75))
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((df[col] < lower) | (df[col] > upper)).sum())
        iqr_audit[col] = {
            "q1": q1, "q3": q3, "iqr": float(iqr),
            "lower_fence": float(lower), "upper_fence": float(upper),
            "flagged_count": count, "flagged_pct": count / len(df) * 100,
            "action": "retained; reviewed rather than deleted",
        }

    subgroup_df = pd.DataFrame([
        {"model": "Full data (numeric predictors only)", **subgroup["all"]},
        {"model": "Male subgroup", **subgroup["male"]},
        {"model": "Female subgroup", **subgroup["female"]},
    ])
    subgroup_df.to_csv(OUTPUT_DIR / "subgroup_coefficients.csv", index=False)
    pd.DataFrame(shifts).to_csv(OUTPUT_DIR / "omitted_variable_coefficients.csv", index=False)

    adv_summary = {
        "count": len(adversarial), "pct_of_test": len(adversarial) / len(test) * 100,
        "false_positives": int(adversarial["prediction_type"].str.startswith("False positive").sum()),
        "false_negatives": int(adversarial["prediction_type"].str.startswith("False negative").sum()),
        "mean_age": float(adversarial["age"].mean()) if len(adversarial) else None,
        "male": int((adversarial["gender_group"] == "Male").sum()),
        "female": int((adversarial["gender_group"] == "Female").sum()),
        "mean_credit_amount": float(adversarial["credit_amount"].mean()) if len(adversarial) else None,
        "traits": trait_rows,
    }
    test_demo = {
        "mean_age": float(test["age"].mean()),
        "male": int((test["gender_group"] == "Male").sum()),
        "female": int((test["gender_group"] == "Female").sum()),
        "mean_credit_amount": float(test["credit_amount"].mean()),
    }
    results = {
        "student": {"name": "Manu Bhardwaj", "enrollment": "S24BCAU0034",
                    "course": "CSET485 - AI and Society"},
        "dataset": {
            "records": len(df), "good": int(df["approval"].sum()),
            "bad": int((1 - df["approval"]).sum()),
            "train": len(train), "test": len(test),
            "train_good": int(y_train.sum()), "train_bad": int((1 - y_train).sum()),
            "test_good": int(y_test.sum()), "test_bad": int((1 - y_test).sum()),
            "features": 20, "encoded_columns": len(feature_names),
        },
        "part1": {
            "linear_prediction_median": median_linear,
            "linear_prediction_range": [float(pred_linear.min()), float(pred_linear.max())],
            "linear_positive": int(class_linear.sum()), "logistic_positive": int(class_log.sum()),
            "confusion_matrix_rows_linear_cols_logistic": cm.tolist(),
            "mismatches": mismatch, "mismatch_pct": mismatch / len(test) * 100,
            "logistic_accuracy_at_0_5": accuracy_05,
        },
        "part2_1": {"predictor": "credit_amount (standardised); numeric-only logistic model", **subgroup},
        "part2_2": {"dropped_variable": "checking_status", "coefficient_shifts": shifts},
        "part2_3": {"adversarial": adv_summary, "test_demographics": test_demo},
        "part2_4": {"sensitive_variable": "personal_status_sex mapped to Male/Female",
                    "table": fairness, "threshold_details": threshold_details},
        "preprocessing_outlier_audit": iqr_audit,
    }
    with open(OUTPUT_DIR / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, allow_nan=False)
    print(json.dumps(results, indent=2, allow_nan=False))
    return results


if __name__ == "__main__":
    run_analysis()

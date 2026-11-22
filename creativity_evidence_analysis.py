"""Reproducible evidence summary for the assignment on AI and student creativity.

This script summarises a manually coded matrix of six published student studies.
It does NOT reanalyse participant-level data, estimate a pooled effect size, or
establish a causal effect. Numerical study results are cited in the assignment.

Run from the project root:
    python creativity_evidence_analysis.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MATRIX = ROOT / "evidence_matrix.csv"
OUTPUTS = ROOT / "outputs"
REQUIRED = {"study", "design", "sample", "area", "classification", "key_finding", "limit"}


def main() -> pd.DataFrame:
    if not MATRIX.exists():
        raise FileNotFoundError(f"Could not find {MATRIX}. Run from the project folder.")
    df = pd.read_csv(MATRIX)
    missing = REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f"Evidence matrix is missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Evidence matrix has no study rows.")

    allowed = {
        "Mixed; positive task scores with qualitative concerns",
        "Positive on solution quality and originality",
        "Mixed; no overall evidence of decline",
        "Mixed; effects depended on discipline and dimension",
        "Mixed; creativity preserved but ideas converged semantically",
        "Positive on average; benefit varied by baseline creative potential",
    }
    unknown = set(df["classification"].dropna()) - allowed
    if unknown:
        raise ValueError(f"Unexpected classification(s): {sorted(unknown)}")

    OUTPUTS.mkdir(exist_ok=True)
    counts = (
        df["classification"].map(
            lambda s: "Positive on measured outcomes" if s.startswith("Positive")
            else "Mixed / context-dependent"
        )
        .value_counts()
        .reindex(["Positive on measured outcomes", "Mixed / context-dependent"], fill_value=0)
        .rename_axis("evidence_group")
        .reset_index(name="study_count")
    )
    counts.to_csv(OUTPUTS / "study_classification_counts.csv", index=False)
    df.to_csv(OUTPUTS / "evidence_matrix_checked.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(counts["evidence_group"], counts["study_count"])
    ax.set_xlabel("Number of selected empirical studies")
    ax.set_title("Findings in the focused literature review")
    ax.set_xlim(0, max(4, int(counts["study_count"].max()) + 1))
    for i, value in enumerate(counts["study_count"]):
        ax.text(value + 0.05, i, str(int(value)), va="center")
    ax.text(
        0.01, -0.30,
        "Descriptive coding of six selected studies; not a meta-analysis or pooled effect estimate.",
        transform=ax.transAxes, fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(OUTPUTS / "evidence_review_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Study records:", len(df))
    print(counts.to_string(index=False))
    print("Outputs saved in:", OUTPUTS)
    return df


if __name__ == "__main__":
    main()

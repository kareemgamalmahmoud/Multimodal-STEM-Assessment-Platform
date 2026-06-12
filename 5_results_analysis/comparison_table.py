"""
Generate a comparison table of all four experiments.

Outputs:
  - ASCII formatted table (printed to console)
  - LaTeX table source (for the methodology report)
  - JSON file with all metrics
"""

import json
from pathlib import Path


DISPLAY_NAMES = {
    "exp1_baseline_direct": "Exp 1 — Baseline (Direct)",
    "exp2_chain_of_thought": "Exp 2 — Chain-of-Thought",
    "exp3_rubric_decomposed": "Exp 3 — Rubric-Decomposed",
    "exp4_astra_ours": "Exp 4 — ASTRA (Ours)",
}


def fmt(value, precision: int = 4) -> str:
    """Format a float or return 'N/A'."""
    if value is None or (isinstance(value, float) and (value != value)):  # NaN check
        return "N/A"
    return f"{float(value):.{precision}f}"


def build_ascii_table(all_metrics: list[dict]) -> str:
    """Format metrics as a readable ASCII table."""
    col_w = [36, 8, 10, 8]
    header = (
        f"{'Method':<{col_w[0]}} {'QWK':>{col_w[1]}} {'Pearson r':>{col_w[2]}} {'RMSE':>{col_w[3]}}"
    )
    sep = "-" * sum(col_w)

    rows = [header, sep]
    for m in all_metrics:
        method_key = m.get("method", "?")
        display = DISPLAY_NAMES.get(method_key, method_key)
        rows.append(
            f"{display:<{col_w[0]}} "
            f"{fmt(m.get('qwk')):>{col_w[1]}} "
            f"{fmt(m.get('pearson_r')):>{col_w[2]}} "
            f"{fmt(m.get('rmse')):>{col_w[3]}}"
        )

    return "\n".join(rows)


def build_latex_table(all_metrics: list[dict]) -> str:
    """Generate a LaTeX tabular environment for the methodology report."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Comparison of scoring methods on FERMAT dataset. "
        r"QWK = Quadratic Weighted Kappa; higher is better. RMSE: lower is better.}",
        r"\label{tab:results}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Method} & \textbf{QWK} & \textbf{Pearson r} & \textbf{RMSE} \\",
        r"\midrule",
    ]
    for m in all_metrics:
        method_key = m.get("method", "?")
        display = DISPLAY_NAMES.get(method_key, method_key)
        # Bold the ASTRA row
        if "astra" in method_key:
            display = r"\textbf{" + display + r"}"
        lines.append(
            f"{display} & {fmt(m.get('qwk'))} & {fmt(m.get('pearson_r'))} & {fmt(m.get('rmse'))} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def build_language_breakdown_table(all_metrics: list[dict]) -> str:
    """ASCII table showing per-language metrics for each experiment."""
    lines = ["Per-Language Pearson r Breakdown", "=" * 60]
    for m in all_metrics:
        method_key = m.get("method", "?")
        display = DISPLAY_NAMES.get(method_key, method_key)
        lines.append(f"\n  {display}")
        per_lang = m.get("per_language", {})
        if not per_lang:
            lines.append("    (no per-language data)")
        for lang, lm in sorted(per_lang.items()):
            lines.append(
                f"    {lang:<10} | n={lm['n']:<5} | "
                f"QWK={fmt(lm.get('qwk'), 3):<7} | "
                f"r={fmt(lm.get('pearson_r'), 3):<7} | "
                f"RMSE={fmt(lm.get('rmse'), 3)}"
            )
    return "\n".join(lines)


def generate_tables(all_metrics: list[dict], output_dir: Path):
    """Write all table formats to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    ascii_table = build_ascii_table(all_metrics)
    latex_table = build_latex_table(all_metrics)
    lang_table = build_language_breakdown_table(all_metrics)

    combined = "\n\n".join([
        "=" * 60,
        "RESULTS COMPARISON TABLE",
        "=" * 60,
        ascii_table,
        "",
        lang_table,
        "",
        "LaTeX Source:",
        "-" * 40,
        latex_table,
    ])

    table_path = output_dir / "comparison_table.txt"
    table_path.write_text(combined, encoding="utf-8")
    print(f"\n  Comparison table → {table_path}")

    # Also save raw metrics
    metrics_path = output_dir / "metrics_all_experiments.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  All metrics JSON → {metrics_path}")

    # Print to console
    print("\n" + ascii_table + "\n")
    print(lang_table)

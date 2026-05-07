#!/usr/bin/env python3
"""
Analysis tools for model scores and duel results.

Usage:
    python analysis.py correlation       Compute correlations and generate plots
    python analysis.py proposer-failures Generate bar chart of proposer failure types
"""

import argparse
import json
import os
import re
from scipy import stats
import altair as alt
import pandas as pd

# Import from elo.py
from elo import compute_all_elos, load_all_results, load_duels, extract_match_outcomes


def load_model_scores(scores_file="model-scores.json"):
    """Load model scores from JSON file."""
    with open(scores_file, 'r') as f:
        return json.load(f)

def strip_model_id(s):
    s = s.split("-2025")[0]
    s = s.split("-2024")[0]
    return s

def create_scatter_plot(df, x_col, y_col, x_label, y_label, output_path):
    """Create a scatter plot with best fit line using Altair."""
    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1

    x_scale = alt.Scale(domain=[x_min - x_margin, x_max + x_margin])
    y_scale = alt.Scale(domain=[y_min - y_margin, y_max + y_margin])
    df['short_model'] = df['model'].apply(strip_model_id)

    points = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X(x_col, title=x_label, scale=x_scale),
        y=alt.Y(y_col, title=y_label, scale=y_scale),
        tooltip=['model', x_col, y_col]
    )

    line = points.transform_regression(
        x_col, y_col
    ).mark_line(color='rgba(255, 100, 100, 100)', strokeDash=[5, 5])

    text = alt.Chart(df).mark_text(
        align='left',
        baseline='middle',
        dx=7,
        fontSize=16
    ).encode(
        x=alt.X(x_col, scale=x_scale),
        y=alt.Y(y_col, scale=y_scale),
        text='short_model'
    )

    chart = (points + line + text).properties(
        width=500,
        height=400,
        title=f'{y_label} vs {x_label}'
    ).configure_title(
        fontSize=24
    ).configure_axis(
        titleFontSize=20,
        labelFontSize=16
    )

    chart.save(output_path, scale_factor=3)
    print(f"  Saved plot: {output_path}")


def compute_win_rates(pattern="runs/classic/*.jsonl"):
    """Compute win rates for each model in different roles."""
    from collections import defaultdict

    results = load_all_results(pattern)
    duels = load_duels(pattern)

    # Track per-model stats
    stats = defaultdict(lambda: {
        "proposer_wins": 0, "proposer_games": 0,
        "solver_wins": 0, "solver_games": 0,
    })

    for r in results:
        proposer = r.get("inventer")
        solver = r.get("solver")
        outcome = r.get("outcome", "")

        if not proposer or not solver:
            continue

        stats[proposer]["proposer_games"] += 1
        stats[solver]["solver_games"] += 1

        if outcome in ("failure", "failure-ex"):
            stats[proposer]["proposer_wins"] += 1
        elif outcome in ("puzzle-wrong", "puzzle-wrong-ex"):
            stats[solver]["solver_wins"] += 1
        elif outcome == "success":
            stats[solver]["solver_wins"] += 1

    # Compute match win rates
    match_outcomes = extract_match_outcomes(duels)
    match_stats = defaultdict(lambda: {"wins": 0, "games": 0})

    for winner, loser, is_draw in match_outcomes:
        match_stats[winner]["games"] += 1
        match_stats[loser]["games"] += 1
        if not is_draw:
            match_stats[winner]["wins"] += 1

    # Build win rate dict
    win_rates = {}
    for model in stats:
        s = stats[model]
        ms = match_stats[model]
        win_rates[model] = {
            "proposer_win_rate": s["proposer_wins"] / s["proposer_games"] if s["proposer_games"] > 0 else 0,
            "solver_win_rate": s["solver_wins"] / s["solver_games"] if s["solver_games"] > 0 else 0,
            "match_win_rate": ms["wins"] / ms["games"] if ms["games"] > 0 else 0,
        }

    return win_rates


def run_correlation_analysis(elo_types, rate_types, benchmarks, benchmark_lookup, win_rates,
                              exclude_pattern=None, suffix=""):
    """Run correlation analysis, optionally excluding models matching a pattern."""

    def should_include(model_id):
        if exclude_pattern is None:
            return True
        return not re.search(exclude_pattern, model_id)

    def shorten_model_name(m):
        return m.replace("-2025", "\n-2025").replace("-20251101", "\n-20251101").replace("-20250929", "\n-20250929")

    excluded_label = f" (excluding /{exclude_pattern}/)" if exclude_pattern else ""

    print("\n" + "=" * 70)
    print(f"CORRELATIONS: TTG Elo vs Benchmarks{excluded_label}")
    print("=" * 70)

    results = []

    for elo_key, elo_name, elo_dict in elo_types:
        print(f"\n{elo_name}")
        print("-" * 50)

        for bench_key, bench_name in benchmarks:
            pairs = []
            for model_id, elo in elo_dict.items():
                if should_include(model_id) and model_id in benchmark_lookup[bench_key]:
                    bench = benchmark_lookup[bench_key][model_id]
                    pairs.append((model_id, elo, bench))

            if len(pairs) < 3:
                print(f"  vs {bench_name}: Not enough data (n={len(pairs)})")
                continue

            elos = [p[1] for p in pairs]
            benches = [p[2] for p in pairs]

            pearson_r, pearson_p = stats.pearsonr(elos, benches)
            spearman_r, spearman_p = stats.spearmanr(elos, benches)

            print(f"  vs {bench_name} (n={len(pairs)})")
            print(f"    Pearson r:  {pearson_r:+.3f}  (p={pearson_p:.4f})")
            print(f"    Spearman ρ: {spearman_r:+.3f}  (p={spearman_p:.4f})")

            results.append({
                "elo_type": elo_name,
                "benchmark": bench_name,
                "n": len(pairs),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
            })

            df = pd.DataFrame([
                {"model": shorten_model_name(m), "elo": e, "benchmark": b}
                for m, e, b in pairs
            ])
            output_path = f"plots/{elo_key}_elo_vs_{bench_key}{suffix}.png"
            create_scatter_plot(df, "elo", "benchmark", elo_name, bench_name, output_path)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"SUMMARY TABLE{excluded_label}")
    print("=" * 70)
    print(f"{'Elo Type':<20}{'Benchmark':<10}{'n':<5}{'Pearson r':<12}{'Spearman ρ':<12}")
    print("-" * 70)

    for r in results:
        sig_p = "*" if r["pearson_p"] < 0.05 else ""
        sig_s = "*" if r["spearman_p"] < 0.05 else ""
        print(f"{r['elo_type']:<20}{r['benchmark']:<10}{r['n']:<5}"
              f"{r['pearson_r']:+.3f}{sig_p:<4}    {r['spearman_r']:+.3f}{sig_s:<4}")

    print("-" * 70)
    print("* = p < 0.05")
    print("=" * 70)

    # Rank-based analysis using win rates
    print("\n\n" + "=" * 70)
    print(f"RANK-BASED ANALYSIS (using win rates){excluded_label}")
    print("=" * 70)

    rank_results = []

    for rate_key, rate_name in rate_types:
        print(f"\n{rate_name}")
        print("-" * 50)

        for bench_key, bench_name in benchmarks:
            pairs = []
            for model_id, rates in win_rates.items():
                if should_include(model_id) and model_id in benchmark_lookup[bench_key]:
                    bench = benchmark_lookup[bench_key][model_id]
                    pairs.append((model_id, rates[rate_key], bench))

            if len(pairs) < 3:
                print(f"  vs {bench_name}: Not enough data (n={len(pairs)})")
                continue

            df = pd.DataFrame(pairs, columns=["model", "rate", "benchmark"])
            df["rate_rank"] = df["rate"].rank(ascending=False)
            df["bench_rank"] = df["benchmark"].rank(ascending=False)

            spearman_r, spearman_p = stats.spearmanr(df["rate"], df["benchmark"])
            pearson_r, pearson_p = stats.pearsonr(df["rate_rank"], df["bench_rank"])

            print(f"  vs {bench_name} (n={len(pairs)})")
            print(f"    Spearman ρ: {spearman_r:+.3f}  (p={spearman_p:.4f})")
            print(f"    Rank corr:  {pearson_r:+.3f}  (p={pearson_p:.4f})")

            df_sorted = df.sort_values("rate_rank")
            print(f"    {'Model':<30} {'Rate':>8} {'RateRk':>7} {'Bench':>8} {'BenchRk':>7}")
            for _, row in df_sorted.iterrows():
                print(f"    {row['model']:<30} {row['rate']:>8.3f} {int(row['rate_rank']):>7} {row['benchmark']:>8.3f} {int(row['bench_rank']):>7}")

            rank_results.append({
                "rate_type": rate_name,
                "benchmark": bench_name,
                "n": len(pairs),
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
            })

            df_plot = pd.DataFrame([
                {"model": shorten_model_name(m), "rate": r, "benchmark": b}
                for m, r, b in pairs
            ])
            output_path = f"plots/{rate_key}_vs_{bench_key}{suffix}.png"
            create_scatter_plot(df_plot, "rate", "benchmark", rate_name, bench_name, output_path)

    # Print rank-based summary
    print("\n" + "=" * 70)
    print(f"RANK-BASED SUMMARY{excluded_label}")
    print("=" * 70)
    print(f"{'Rate Type':<25}{'Benchmark':<10}{'n':<5}{'Spearman ρ':<12}")
    print("-" * 70)

    for r in rank_results:
        sig = "*" if r["spearman_p"] < 0.05 else ""
        print(f"{r['rate_type']:<25}{r['benchmark']:<10}{r['n']:<5}{r['spearman_r']:+.3f}{sig:<4}")

    print("-" * 70)
    print("* = p < 0.05")
    print("=" * 70)


def generate_latex_table(model_scores, all_elos, win_rates, benchmark_lookup, exclude_pattern=None):
    """Generate LaTeX tabular for RQ1 with model rankings (tabular only, no table wrapper)."""

    def should_include(model_id):
        if exclude_pattern is None:
            return True
        return not re.search(exclude_pattern, model_id)

    # Build combined dataframe
    rows = []
    for m in model_scores:
        model_id = m["id"]
        if not should_include(model_id):
            continue

        row = {"model": model_id}

        # TTG metrics
        if model_id in all_elos["solver"]:
            row["ttg_elo"] = all_elos["solver"][model_id]
        if model_id in win_rates:
            row["solver_win_rate"] = win_rates[model_id]["solver_win_rate"]
            row["proposer_win_rate"] = win_rates[model_id]["proposer_win_rate"]

        # Benchmarks
        for bench_key in ["hle", "arc_agi", "swe_bench_pro", "textquests", "gpqa_diamond", "avg"]:
            if model_id in benchmark_lookup.get(bench_key, {}):
                row[bench_key] = benchmark_lookup[bench_key][model_id]

        rows.append(row)

    df = pd.DataFrame(rows)

    # Compute ranks (1 = best), handling NaN values
    if "ttg_elo" in df.columns:
        df["ttg_elo_rank"] = df["ttg_elo"].rank(ascending=False)
    if "solver_win_rate" in df.columns:
        df["solver_rank"] = df["solver_win_rate"].rank(ascending=False)
    if "proposer_win_rate" in df.columns:
        df["proposer_rank"] = df["proposer_win_rate"].rank(ascending=False)
    for bench_key in ["hle", "arc_agi", "swe_bench_pro", "textquests", "avg"]:
        if bench_key in df.columns:
            df[f"{bench_key}_rank"] = df[bench_key].rank(ascending=False)

    # Sort by TTG Elo rank
    df = df.sort_values("ttg_elo_rank")

    n_models = len(df)

    def color_rank(rank, n):
        """Return LaTeX color command for rank."""
        if pd.isna(rank):
            return ""
        rank = int(rank)
        if rank <= 3:
            return "\\cellcolor{green!40}"
        elif rank > n - 3:
            return "\\cellcolor{red!40}"
        return ""

    bench_specs = [
        ("hle", "HLE"),
        ("arc_agi", "ARC-AGI"),
        ("swe_bench_pro", "SWE-BP"),
        ("textquests", "TQ"),
        ("gpqa_diamond", "GPQA-D"),
        ("avg", "Avg"),
    ]

    # Generate LaTeX (tabular only)
    n_bench = len(bench_specs)
    n_ttg_cols = 6  # 3 metrics x (val, rank)
    total_cols = n_ttg_cols + 2 * n_bench
    col_spec = "l|rr|rr|rr" + "|rr" * n_bench
    latex = []
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")
    bench_header = " & ".join([f"\\multicolumn{{2}}{{c{'|' if i < n_bench-1 else ''}}}{{\\textbf{{{bn}}}}}" for i, (_, bn) in enumerate(bench_specs)])
    latex.append(f"& \\multicolumn{{6}}{{c|}}{{\\textbf{{TTG}}}} & {bench_header} \\\\")
    cmidrules = ["\\cmidrule(lr){2-7}"]
    for i in range(n_bench):
        start = 8 + 2 * i
        cmidrules.append(f"\\cmidrule(lr){{{start}-{start+1}}}")
    latex.append(" ".join(cmidrules))
    bench_col_headers = " & ".join(["\\textbf{Acc} & \\textbf{Rk}"] * n_bench)
    latex.append(f"\\textbf{{Model}} & \\textbf{{Elo}} & \\textbf{{Rk}} & \\textbf{{Solv\\%}} & \\textbf{{Rk}} & \\textbf{{Prop\\%}} & \\textbf{{Rk}} & {bench_col_headers} \\\\")
    latex.append("\\midrule")

    for _, row in df.iterrows():
        model_id = row["model"]
        # Format model name with \allowbreak for long names
        model_name = model_id.replace("-", "-\\allowbreak ")

        def fmt_rank(val):
            return str(int(val)) if pd.notna(val) else "--"

        # TTG columns
        elo = f"{row['ttg_elo']:.0f}" if pd.notna(row.get('ttg_elo')) else "--"
        elo_rank = row.get('ttg_elo_rank')
        elo_color = color_rank(elo_rank, n_models)

        solver = f"{row['solver_win_rate']*100:.1f}" if pd.notna(row.get('solver_win_rate')) else "--"
        solver_rank = row.get('solver_rank')
        solver_color = color_rank(solver_rank, n_models)

        proposer = f"{row['proposer_win_rate']*100:.1f}" if pd.notna(row.get('proposer_win_rate')) else "--"
        proposer_rank = row.get('proposer_rank')
        proposer_color = color_rank(proposer_rank, n_models)

        bench_cells = []
        for bench_key, _ in bench_specs:
            val = row.get(bench_key)
            val_str = f"{val*100:.1f}" if pd.notna(val) else "--"
            rk = row.get(f"{bench_key}_rank")
            color = color_rank(rk, n_models)
            bench_cells.append(f"{val_str} & {color}{fmt_rank(rk)}")
        bench_part = " & ".join(bench_cells)

        line = f"{model_name} & {elo} & {elo_color}{fmt_rank(elo_rank)} & {solver} & {solver_color}{fmt_rank(solver_rank)} & {proposer} & {proposer_color}{fmt_rank(proposer_rank)} & {bench_part} \\\\"
        latex.append(line)

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")

    return "\n".join(latex)


def generate_correlation_table(model_scores, all_elos, win_rates, benchmark_lookup, exclude_pattern=None):
    """Generate LaTeX tabular for correlations (rq1-2.tex) with and without outlier rows."""

    bench_specs = [
        ("hle", "HLE"),
        ("arc_agi", "ARC-AGI"),
        ("swe_bench_pro", "SWE-BP"),
        ("textquests", "TQ"),
        ("gpqa_diamond", "GPQA-D"),
        ("avg", "Avg"),
    ]

    def compute_correlations_for_df(df):
        """Compute Spearman correlations for all metric-benchmark pairs."""
        results = {}
        for metric_name, metric_col in [("Elo", "ttg_elo"), ("Solver Win Rate", "solver_win_rate"), ("Proposer Win Rate", "proposer_win_rate")]:
            for bench_key, _ in bench_specs:
                if metric_col not in df.columns or bench_key not in df.columns:
                    results[(metric_name, bench_key)] = (float('nan'), float('nan'))
                    continue
                valid = df[[metric_col, bench_key]].dropna()
                if len(valid) >= 3:
                    rho, p = stats.spearmanr(valid[metric_col], valid[bench_key])
                    results[(metric_name, bench_key)] = (rho, p)
                else:
                    results[(metric_name, bench_key)] = (float('nan'), float('nan'))
        return results

    def build_df(include_all=True):
        """Build dataframe, optionally excluding outlier."""
        rows = []
        for m in model_scores:
            model_id = m["id"]
            if not include_all and exclude_pattern and re.search(exclude_pattern, model_id):
                continue

            row = {"model": model_id}
            if model_id in all_elos["solver"]:
                row["ttg_elo"] = all_elos["solver"][model_id]
            if model_id in win_rates:
                row["solver_win_rate"] = win_rates[model_id]["solver_win_rate"]
                row["proposer_win_rate"] = win_rates[model_id]["proposer_win_rate"]
            for bench_key, _ in bench_specs:
                if model_id in benchmark_lookup.get(bench_key, {}):
                    row[bench_key] = benchmark_lookup[bench_key][model_id]
            rows.append(row)
        return pd.DataFrame(rows)

    # Compute correlations for both cases
    df_all = build_df(include_all=True)
    df_no_outlier = build_df(include_all=False)

    corr_all = compute_correlations_for_df(df_all)
    corr_no_outlier = compute_correlations_for_df(df_no_outlier)

    # Determine outlier label from exclude_pattern (best-effort) and detect which models match
    excluded_models = []
    if exclude_pattern:
        excluded_models = [m["id"] for m in model_scores if re.search(exclude_pattern, m["id"])]
    outlier_label = ", ".join(excluded_models) if excluded_models else "outlier"

    # Generate LaTeX (tabular only)
    n_bench = len(bench_specs)
    col_spec = "l" + ("|cc" * n_bench)
    latex = []
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")
    bench_header = " & ".join([f"\\multicolumn{{2}}{{c{'|' if i < n_bench-1 else ''}}}{{\\textbf{{vs {bn}}}}}" for i, (_, bn) in enumerate(bench_specs)])
    latex.append(f"& {bench_header} \\\\")
    rho_p_header = " & ".join(["$\\rho$ & $p$"] * n_bench)
    latex.append(f"\\textbf{{TTG Metric}} & {rho_p_header} \\\\")
    latex.append("\\midrule")

    def fmt_corr_row(label, corr_dict):
        cells = []
        for bench_key, _ in bench_specs:
            rho, p = corr_dict[(metric_name, bench_key)]
            sig = "*" if pd.notna(p) and p < 0.05 else ""
            rho_str = f"{rho:+.2f}{sig}" if pd.notna(rho) else "--"
            p_str = f"{p:.3f}" if pd.notna(p) else "--"
            cells.append(f"{rho_str} & {p_str}")
        return f"{label} & " + " & ".join(cells) + " \\\\"

    # Add rows for each metric (all models first, then w/o outlier)
    for metric_name in ["Elo", "Solver Win Rate", "Proposer Win Rate"]:
        latex.append(fmt_corr_row(metric_name, corr_all))
        if excluded_models:
            latex.append(fmt_corr_row(f"\\quad (w/o {outlier_label})", corr_no_outlier))

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")

    return "\n".join(latex)


def proposer_failures_command(args):
    """Generate bar chart of proposer failure types by model."""
    from collections import defaultdict

    os.makedirs("plots", exist_ok=True)

    results = load_all_results()

    # Count proposer failures per model
    stats = defaultdict(lambda: {"incorrect_solution": 0, "solver_succeeded": 0})

    for r in results:
        proposer = r.get("inventer")
        outcome = r.get("outcome", "")

        if not proposer:
            continue

        if outcome in ("puzzle-wrong", "puzzle-wrong-ex"):
            stats[proposer]["incorrect_solution"] += 1
        elif outcome == "success":
            stats[proposer]["solver_succeeded"] += 1

    # Build dataframe for plotting
    rows = []
    for model, counts in stats.items():
        short_name = strip_model_id(model)
        rows.append({
            "model": short_name,
            "failure_type": "Incorrect solution",
            "count": counts["incorrect_solution"]
        })
        rows.append({
            "model": short_name,
            "failure_type": "Solver succeeded",
            "count": counts["solver_succeeded"]
        })

    df = pd.DataFrame(rows)

    # Sort models by total failures for better visualization
    model_totals = df.groupby("model")["count"].sum().sort_values(ascending=False)
    model_order = model_totals.index.tolist()

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("model:N", title=None, sort=model_order),
        y=alt.Y("count:Q", title="Number of rounds"),
        color=alt.Color("failure_type:N", title="Failure type",
                        scale=alt.Scale(domain=["Incorrect solution", "Solver succeeded"],
                                        range=["#e45756", "#4c78a8"]),
                        legend=alt.Legend(
                            orient="none",
                            legendX=350,
                            legendY=30,
                            direction="horizontal"
                        )),
        xOffset="failure_type:N"
    ).properties(
        width=600,
        height=400,
        title="Proposer Failures by Model"
    ).configure_title(
        fontSize=20
    ).configure_axis(
        titleFontSize=16,
        labelFontSize=18,
        labelAngle=-45
    ).configure_legend(
        titleFontSize=14,
        labelFontSize=12
    )

    output_path = "plots/proposer_failures.png"
    chart.save(output_path, scale_factor=3)
    print(f"Saved: {output_path}")

    # Print summary table
    print("\nProposer Failure Summary:")
    print(f"{'Model':<30} {'Incorrect':>12} {'Solver OK':>12} {'Total':>10}")
    print("-" * 66)
    for model in model_order:
        model_data = stats[[m for m in stats if strip_model_id(m) == model][0]]
        inc = model_data["incorrect_solution"]
        sol = model_data["solver_succeeded"]
        print(f"{model:<30} {inc:>12} {sol:>12} {inc + sol:>10}")


def correlation_command(args):
    """Compute correlations between TTG Elo scores and benchmarks."""
    model_scores = load_model_scores()
    exclude_pattern = args.exclude

    os.makedirs("plots", exist_ok=True)

    print("Computing Elo ratings...")
    all_elos = compute_all_elos()

    print("Computing win rates...")
    win_rates = compute_win_rates()

    elo_types = [
        ("match", "Match Elo", all_elos["match"]),
        ("proposer", "Proposer Elo", all_elos["proposer"]),
        ("solver", "Solver Elo", all_elos["solver"]),
    ]

    rate_types = [
        ("match_win_rate", "Match Win Rate"),
        ("proposer_win_rate", "Proposer Win Rate"),
        ("solver_win_rate", "Solver Win Rate"),
    ]

    benchmarks = [
        ("hle", "HLE"),
        ("arc_agi", "ARC-AGI"),
        ("swe_bench_pro", "SWE-Bench Pro"),
        ("textquests", "TextQuests"),
        ("gpqa_diamond", "GPQA Diamond"),
        ("avg", "Avg"),
    ]

    benchmark_lookup = {}
    for bench_key, bench_name in benchmarks:
        benchmark_lookup[bench_key] = {}
        for m in model_scores:
            if m.get(bench_key) is not None:
                benchmark_lookup[bench_key][m["id"]] = m[bench_key]

    # Run with all models
    run_correlation_analysis(elo_types, rate_types, benchmarks, benchmark_lookup, win_rates,
                             exclude_pattern=None, suffix="")

    # Run excluding outliers
    if exclude_pattern:
        run_correlation_analysis(elo_types, rate_types, benchmarks, benchmark_lookup, win_rates,
                                 exclude_pattern=exclude_pattern, suffix="_no_outlier")

    # Generate LaTeX tables
    print("\nGenerating LaTeX tables...")

    # Table with all models
    latex_all = generate_latex_table(model_scores, all_elos, win_rates, benchmark_lookup,
                                      exclude_pattern=None)
    with open("rq1.tex", "w") as f:
        f.write(latex_all)
        f.write("\n")
    print("  Saved: rq1.tex")

    # Table without outliers
    if exclude_pattern:
        latex_no_outlier = generate_latex_table(model_scores, all_elos, win_rates, benchmark_lookup,
                                                 exclude_pattern=exclude_pattern)
        with open("rq1_no_outlier.tex", "w") as f:
            f.write(latex_no_outlier)
            f.write("\n")
        print("  Saved: rq1_no_outlier.tex")

    # Correlation table (with and without outlier rows)
    latex_corr = generate_correlation_table(model_scores, all_elos, win_rates, benchmark_lookup,
                                             exclude_pattern=exclude_pattern)
    with open("rq1-2.tex", "w") as f:
        f.write(latex_corr)
        f.write("\n")
    print("  Saved: rq1-2.tex")

    # RQ2: Solver vs Proposer correlation
    print("\n" + "=" * 70)
    print("RQ2: Solver Win Rate vs Proposer Win Rate Correlation")
    print("=" * 70)

    solver_rates = []
    proposer_rates = []
    model_names = []
    for model_id, rates in win_rates.items():
        solver_rates.append(rates["solver_win_rate"])
        proposer_rates.append(rates["proposer_win_rate"])
        model_names.append(model_id)

    df_rq2 = pd.DataFrame({
        "model": model_names,
        "solver_win_rate": solver_rates,
        "proposer_win_rate": proposer_rates
    })
    df_rq2["solver_rank"] = df_rq2["solver_win_rate"].rank(ascending=False)
    df_rq2["proposer_rank"] = df_rq2["proposer_win_rate"].rank(ascending=False)

    # Spearman correlation
    rho, p = stats.spearmanr(df_rq2["solver_win_rate"], df_rq2["proposer_win_rate"])
    print(f"Spearman correlation: rho = {rho:+.3f}, p = {p:.4f}")

    # Summary stats
    avg_solver = df_rq2["solver_win_rate"].mean()
    avg_proposer = df_rq2["proposer_win_rate"].mean()
    print(f"\nAverage solver win rate:   {avg_solver*100:.1f}%")
    print(f"Average proposer win rate: {avg_proposer*100:.1f}%")

    # Per-model breakdown
    print(f"\n{'Model':<35} {'Solver%':>8} {'SolvRk':>7} {'Prop%':>8} {'PropRk':>7}")
    print("-" * 70)
    for _, row in df_rq2.sort_values("solver_rank").iterrows():
        print(f"{row['model']:<35} {row['solver_win_rate']*100:>7.1f}% {int(row['solver_rank']):>7} "
              f"{row['proposer_win_rate']*100:>7.1f}% {int(row['proposer_rank']):>7}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analysis tools for model scores")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # correlation subcommand
    corr_parser = subparsers.add_parser(
        "correlation",
        help="Compute correlations between TTG Elo scores and benchmarks"
    )
    corr_parser.add_argument(
        "--exclude",
        default="^gpt-5\\.5$",
        help="Regex pattern to exclude models as outliers (default: ^gpt-5\\.5$)"
    )
    corr_parser.set_defaults(func=correlation_command)

    # proposer-failures subcommand
    pf_parser = subparsers.add_parser(
        "proposer-failures",
        help="Generate bar chart of proposer failure types by model"
    )
    pf_parser.set_defaults(func=proposer_failures_command)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()

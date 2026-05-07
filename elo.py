#!/usr/bin/env python3
"""
Compute Elo ratings from duel results.

Supports three types of Elo:
- Match Elo: Based on complete duel outcomes (who won the match)
- Proposer Elo: Based on proposer performance in each round
- Solver Elo: Based on solver performance in each round
"""

import glob
import json
import math
import os
from collections import defaultdict


def load_all_results(pattern="runs/classic/*.jsonl"):
    """Load all duel results from JSONL files."""
    results = []
    for filepath in glob.glob(pattern):
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return results


def load_duels(pattern="runs/classic/*.jsonl"):
    """Load game results grouped by duel (match file)."""
    duels = {}
    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)
        parts = filename.replace(".jsonl", "").split("_vs_")
        if len(parts) != 2:
            continue
        model_a = parts[0]
        model_b_parts = parts[1].rsplit("_", 2)
        model_b = model_b_parts[0] if len(model_b_parts) >= 3 else parts[1]

        rounds = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rounds.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        if rounds:
            duels[filepath] = {
                "model_a": model_a,
                "model_b": model_b,
                "rounds": rounds
            }
    return duels


# =============================================================================
# Outcome extraction functions
# =============================================================================

def extract_round_outcomes(results):
    """
    Extract round-level outcomes (original behavior).

    Returns list of (winner, loser, is_draw) tuples.
    - "failure" = proposer wins (solver failed)
    - "puzzle-wrong" = solver wins (invalid puzzle)
    - "success" = draw
    """
    outcomes = []
    for r in results:
        proposer = r.get("inventer")
        solver = r.get("solver")
        outcome = r.get("outcome", "")

        if not proposer or not solver:
            continue

        if outcome in ("failure", "failure-ex"):
            outcomes.append((proposer, solver, False))
        elif outcome in ("puzzle-wrong", "puzzle-wrong-ex"):
            outcomes.append((solver, proposer, False))
        elif outcome == "success":
            outcomes.append((proposer, solver, True))

    return outcomes


def extract_proposer_outcomes(results):
    """
    Extract proposer-focused outcomes.

    For proposer skill: did the proposer stump the solver?
    Returns list of (proposer, solver, is_draw) where:
    - Win: solver failed to solve (failure)
    - Loss: puzzle was invalid (puzzle-wrong)
    - Draw: solver succeeded (success)
    """
    outcomes = []
    for r in results:
        proposer = r.get("inventer")
        solver = r.get("solver")
        outcome = r.get("outcome", "")

        if not proposer or not solver:
            continue

        if outcome in ("failure", "failure-ex"):
            outcomes.append((proposer, solver, False))  # Proposer wins
        elif outcome in ("puzzle-wrong", "puzzle-wrong-ex"):
            outcomes.append((solver, proposer, False))  # Proposer loses
        elif outcome == "success":
            outcomes.append((proposer, solver, True))   # Draw

    return outcomes


def extract_solver_outcomes(results):
    """
    Extract solver-focused outcomes.

    For solver skill: did the solver solve the puzzle?
    Returns list of (solver, proposer, is_draw) where:
    - Win: solved successfully OR puzzle was invalid
    - Loss: failed to solve
    - Draw: (none - solving is binary)
    """
    outcomes = []
    for r in results:
        proposer = r.get("inventer")
        solver = r.get("solver")
        outcome = r.get("outcome", "")

        if not proposer or not solver:
            continue

        if outcome == "success":
            outcomes.append((solver, proposer, False))  # Solver wins
        elif outcome in ("puzzle-wrong", "puzzle-wrong-ex"):
            outcomes.append((solver, proposer, False))  # Solver wins (invalid puzzle)
        elif outcome in ("failure", "failure-ex"):
            outcomes.append((proposer, solver, False))  # Solver loses

    return outcomes


def extract_match_outcomes(duels):
    """
    Extract match-level outcomes from complete duels.

    Each duel file becomes one match. Winner is whoever accumulated
    more points across all rounds.
    """
    outcomes = []

    for filepath, duel in duels.items():
        model_a = duel["model_a"]
        model_b = duel["model_b"]
        rounds = duel["rounds"]

        scores = defaultdict(float)
        for r in rounds:
            proposer = r.get("inventer")
            solver = r.get("solver")
            outcome = r.get("outcome", "")

            if not proposer or not solver:
                continue

            if outcome in ("failure", "failure-ex"):
                scores[proposer] += 1
            elif outcome in ("puzzle-wrong", "puzzle-wrong-ex"):
                scores[solver] += 1
            elif outcome == "success":
                scores[proposer] += 0.5
                scores[solver] += 0.5

        score_a = scores.get(model_a, 0)
        score_b = scores.get(model_b, 0)

        if score_a > score_b:
            outcomes.append((model_a, model_b, False))
        elif score_b > score_a:
            outcomes.append((model_b, model_a, False))
        else:
            outcomes.append((model_a, model_b, True))

    return outcomes


# =============================================================================
# Unified Elo computation
# =============================================================================

def compute_elo(outcomes, base_elo=1000, scale=400, lr=10.0, max_iters=5000, tol=1e-6):
    """
    Compute MLE Elo ratings from a list of outcomes.

    Args:
        outcomes: List of (winner, loser, is_draw) tuples
        base_elo: Base Elo for the first model (anchor)
        scale: Elo scale parameter (default 400)
        lr: Learning rate for gradient descent
        max_iters: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Dictionary mapping model names to Elo ratings
    """
    # Build pairwise statistics
    stats = defaultdict(lambda: {"wins_a": 0, "wins_b": 0, "draws": 0})

    for winner, loser, is_draw in outcomes:
        a, b = sorted([winner, loser])
        key = (a, b)

        if is_draw:
            stats[key]["draws"] += 1
        elif winner == a:
            stats[key]["wins_a"] += 1
        else:
            stats[key]["wins_b"] += 1

    # Get all unique models
    models = set()
    for a, b in stats.keys():
        models.add(a)
        models.add(b)
    models = sorted(models)

    if len(models) == 0:
        return {}
    if len(models) == 1:
        return {models[0]: base_elo}

    def elo_prob(elo_a, elo_b):
        return 1.0 / (1.0 + math.pow(10, (elo_b - elo_a) / scale))

    def neg_log_likelihood(elo_values):
        elo_dict = {model: elo_values[i] for i, model in enumerate(models)}
        nll = 0.0
        eps = 1e-10

        for (a, b), s in stats.items():
            elo_a, elo_b = elo_dict[a], elo_dict[b]
            p_a_wins = elo_prob(elo_a, elo_b)
            p_b_wins = 1 - p_a_wins

            if s["wins_a"] > 0:
                nll -= s["wins_a"] * math.log(max(p_a_wins, eps))
            if s["wins_b"] > 0:
                nll -= s["wins_b"] * math.log(max(p_b_wins, eps))
            if s["draws"] > 0:
                nll -= s["draws"] * 0.5 * (math.log(max(p_a_wins, eps)) +
                                            math.log(max(p_b_wins, eps)))
        return nll

    # Gradient descent with first model fixed at base_elo
    n_free = len(models) - 1
    x = [base_elo] * n_free
    eps = 1e-4

    for iteration in range(max_iters):
        def objective(free_elos):
            return neg_log_likelihood([base_elo] + list(free_elos))

        f0 = objective(x)
        grad = []
        for i in range(n_free):
            x_plus = x.copy()
            x_plus[i] += eps
            grad.append((objective(x_plus) - f0) / eps)

        max_grad = max(abs(g) for g in grad) if grad else 0
        if max_grad < tol:
            break

        for i in range(n_free):
            x[i] -= lr * grad[i]

        if iteration > 0 and iteration % 500 == 0:
            lr *= 0.9

    final_elos = [base_elo] + x
    return {model: elo for model, elo in zip(models, final_elos)}


def compute_pairwise_stats(outcomes):
    """Compute win/loss/draw statistics for each pair of models."""
    stats = defaultdict(lambda: {"wins_a": 0, "wins_b": 0, "draws": 0})

    for winner, loser, is_draw in outcomes:
        a, b = sorted([winner, loser])
        key = (a, b)

        if is_draw:
            stats[key]["draws"] += 1
        elif winner == a:
            stats[key]["wins_a"] += 1
        else:
            stats[key]["wins_b"] += 1

    return stats


# =============================================================================
# High-level functions to compute all Elo types
# =============================================================================

def compute_all_elos(pattern="runs/classic/*.jsonl"):
    """
    Compute all three types of Elo ratings.

    Returns:
        dict with keys 'match', 'proposer', 'solver', each mapping to Elo dict
    """
    results = load_all_results(pattern)
    duels = load_duels(pattern)

    match_outcomes = extract_match_outcomes(duels)
    proposer_outcomes = extract_proposer_outcomes(results)
    solver_outcomes = extract_solver_outcomes(results)

    return {
        "match": compute_elo(match_outcomes),
        "proposer": compute_elo(proposer_outcomes),
        "solver": compute_elo(solver_outcomes),
    }


def update_model_scores(elo_dict, scores_file="model-scores.json", key="ttg_elo"):
    """Update model-scores.json with computed Elo ratings."""
    try:
        with open(scores_file, 'r') as f:
            model_scores = json.load(f)
    except FileNotFoundError:
        print(f"Warning: {scores_file} not found, skipping update.")
        return

    for model_entry in model_scores:
        model_id = model_entry.get("id")
        if model_id and model_id in elo_dict:
            model_entry[key] = round(elo_dict[model_id], 1)

    with open(scores_file, 'w') as f:
        json.dump(model_scores, f, indent=2)
        f.write('\n')

    print(f"Updated {scores_file} with {key} scores.")


def print_elo_table(elo_dict, title="ELO RATINGS"):
    """Print Elo ratings as a ranked table."""
    if not elo_dict:
        print("No results found.")
        return

    ranked = sorted(elo_dict.items(), key=lambda x: -x[1])

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"{'Rank':<6}{'Model':<40}{'Elo':>10}")
    print("-" * 60)

    for rank, (model, elo) in enumerate(ranked, 1):
        print(f"{rank:<6}{model:<40}{elo:>10.1f}")

    print("=" * 60)


def main():
    results = load_all_results()
    duels = load_duels()
    print(f"Loaded {len(results)} rounds from {len(duels)} duels.")

    # Compute all Elo types
    all_elos = compute_all_elos()

    print_elo_table(all_elos["match"], "MATCH ELO (complete duels)")
    print_elo_table(all_elos["proposer"], "PROPOSER ELO (round-based)")
    print_elo_table(all_elos["solver"], "SOLVER ELO (round-based)")

    # Update model-scores.json with all three
    update_model_scores(all_elos["match"], key="ttg_match_elo")
    update_model_scores(all_elos["proposer"], key="ttg_proposer_elo")
    update_model_scores(all_elos["solver"], key="ttg_solver_elo")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Mine puzzles from arena.py JSON output files and test them against multiple models.

Usage:
    python mine_puzzles.py --models gpt-5.2,claude-opus-4-5-20251101 puzzle_results.jsonl [more_files.jsonl ...]
"""

import json
import argparse
import os

from llm_utils import (
    get_llm, invoke_with_tracking, checkSolution, findSolution, promptSolvePuzzle,
    token_usage, format_usage
)


def load_puzzles(json_files):
    """Load puzzles from JSONL files, filtering out invalid ones."""
    puzzles = []
    seen_codes = set()

    for filepath in json_files:
        file_index = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                outcome = record.get("outcome", "")
                if outcome in ("puzzle-wrong", "puzzle-wrong-ex"):
                    file_index += 1
                    continue

                code = record.get("code", "").strip()
                if not code:
                    file_index += 1
                    continue

                # Deduplicate by code
                if code in seen_codes:
                    file_index += 1
                    continue
                seen_codes.add(code)

                puzzles.append({
                    "code": code,
                    "inventer": record.get("inventer", "unknown"),
                    "sample_solution": record.get("sample_solution", ""),
                    "source_file": filepath,
                    "file_index": file_index,
                    # Preserve original solver info for potential reuse
                    "original_solver": record.get("solver", "unknown"),
                    "original_outcome": record.get("outcome", ""),
                    "original_reasoning": record.get("reasoning", ""),
                    "original_attempt": record.get("attempt", ""),
                })
                file_index += 1

    return puzzles


def load_tested_puzzles(output_file):
    """Load already-tested puzzle/model combinations from the output file.

    Returns a set of (source_file, file_index, model) tuples.
    """
    tested = set()
    if not output_file or not os.path.exists(output_file):
        return tested

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    key = (record.get("source_file"), record.get("file_index"), record.get("model"))
                    tested.add(key)
                except json.JSONDecodeError:
                    continue
    except IOError:
        pass

    return tested


def test_model_on_puzzle(model, model_name, puzzle):
    """Test a model on a puzzle. Returns (success, attempt, error, usage, reasoning, raw)."""
    try:
        content, response, usage = invoke_with_tracking(model, promptSolvePuzzle(puzzle["code"]), model_name)
        attempt = findSolution(content)
    except Exception as e:
        return False, None, f"Error getting solution: {e}", {"input_tokens": 0, "output_tokens": 0}, None, None

    try:
        result = checkSolution(puzzle["code"], attempt)
        return result, attempt, None, usage, content, response
    except Exception as e:
        return False, attempt, f"Error checking solution: {e}", usage, content, response


def main():
    parser = argparse.ArgumentParser(description='Mine puzzles from arena.py output and test against models')
    parser.add_argument('json_files', nargs='+', help='JSONL files from arena.py')
    parser.add_argument('--models', type=str, required=True,
                        help='Comma-separated list of models to test')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSONL file for results')
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(',')]
    print(f"Models to test: {model_names}")

    # Load puzzles
    puzzles = load_puzzles(args.json_files)
    print(f"Loaded {len(puzzles)} unique valid puzzles from {len(args.json_files)} file(s)")

    if not puzzles:
        print("No puzzles to test.")
        return

    # Initialize models
    models = {}
    for name in model_names:
        try:
            models[name] = get_llm(name)
            print(f"  Initialized {name}")
        except Exception as e:
            print(f"  Failed to initialize {name}: {e}")

    if not models:
        print("No models available.")
        return

    # Load already-tested puzzle/model combinations
    tested_combinations = load_tested_puzzles(args.output)
    if tested_combinations:
        print(f"Found {len(tested_combinations)} already-tested puzzle/model combinations")

    # Results tracking
    results = []
    scores = {name: {"solved": 0, "failed": 0, "error": 0, "skipped": 0} for name in models}

    # Open log file for streaming writes if specified (append mode)
    log_file = None
    if args.output:
        log_file = open(args.output, 'a', encoding='utf-8')

    # Test each puzzle with each model
    for i, puzzle in enumerate(puzzles):
        print(f"\n=== Puzzle {i+1}/{len(puzzles)} (by {puzzle['inventer']}, solved by {puzzle['original_solver']}) ===")
        print(f"Code preview: {puzzle['code'][:80]}...")

        puzzle_results = {
            "source_file": puzzle["source_file"],
            "file_index": puzzle["file_index"],
            "inventer": puzzle["inventer"],
            "original_solver": puzzle["original_solver"],
            "original_outcome": puzzle["original_outcome"],
            "code": puzzle["code"],
            "sample_solution": puzzle["sample_solution"],
            "model_results": {}
        }

        for model_name, model in models.items():
            # Check if this puzzle/model combination was already tested
            test_key = (puzzle["source_file"], puzzle["file_index"], model_name)
            if test_key in tested_combinations:
                print(f"  {model_name}: ALREADY TESTED (skipping)")
                continue

            # Check if we can skip solving - original solver matches this model
            if puzzle["original_solver"] == model_name:
                # Copy original result instead of solving again
                original_success = puzzle["original_outcome"] in ("solved", "correct")
                scores[model_name]["skipped"] += 1
                if original_success:
                    scores[model_name]["solved"] += 1
                else:
                    scores[model_name]["failed"] += 1
                print(f"  {model_name}: SKIPPED (original solver) - outcome={puzzle['original_outcome']}")

                puzzle_results["model_results"][model_name] = {
                    "success": original_success,
                    "attempt": puzzle["original_attempt"],
                    "error": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                    "reasoning": puzzle["original_reasoning"],
                    "skipped": True,
                }

                # Log to JSONL
                if log_file:
                    log_entry = {
                        "source_file": puzzle["source_file"],
                        "file_index": puzzle["file_index"],
                        "model": model_name,
                        "inventer": puzzle["inventer"],
                        "original_solver": puzzle["original_solver"],
                        "original_outcome": puzzle["original_outcome"],
                        "skipped": True,
                        "success": original_success,
                        "attempt": puzzle["original_attempt"],
                        "reasoning": puzzle["original_reasoning"],
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    }
                    log_file.write(json.dumps(log_entry) + "\n")
                    log_file.flush()
                continue

            success, attempt, error, usage, reasoning, raw = test_model_on_puzzle(model, model_name, puzzle)

            status = "SOLVED" if success else ("ERROR" if error else "FAILED")
            usage_str = format_usage(model_name, usage["input_tokens"], usage["output_tokens"])

            if success:
                scores[model_name]["solved"] += 1
                print(f"  {model_name}: {status} with {attempt} - {usage_str}")
            elif error:
                scores[model_name]["error"] += 1
                print(f"  {model_name}: {status} - {error} - {usage_str}")
            else:
                scores[model_name]["failed"] += 1
                print(f"  {model_name}: {status} (tried {attempt}) - {usage_str}")

            puzzle_results["model_results"][model_name] = {
                "success": success,
                "attempt": attempt,
                "error": error,
                "usage": usage,
                "reasoning": reasoning,
                "raw": raw,
                "skipped": False,
            }

            # Log to JSONL
            if log_file:
                log_entry = {
                    "source_file": puzzle["source_file"],
                    "file_index": puzzle["file_index"],
                    "model": model_name,
                    "inventer": puzzle["inventer"],
                    "original_solver": puzzle["original_solver"],
                    "original_outcome": puzzle["original_outcome"],
                    "skipped": False,
                    "success": success,
                    "attempt": attempt,
                    "error": error,
                    "reasoning": reasoning,
                    "usage": usage,
                }
                log_file.write(json.dumps(log_entry) + "\n")
                log_file.flush()

        results.append(puzzle_results)

    # Close log file
    if log_file:
        log_file.close()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for model_name in models:
        s = scores[model_name]
        total = s["solved"] + s["failed"] + s["error"]
        pct = (s["solved"] / total * 100) if total > 0 else 0
        usage = token_usage.get(model_name, {"input_tokens": 0, "output_tokens": 0})
        usage_str = format_usage(model_name, usage["input_tokens"], usage["output_tokens"])
        skipped_str = f" ({s['skipped']} skipped)" if s["skipped"] > 0 else ""
        print(f"{model_name}: {s['solved']}/{total} solved ({pct:.1f}%){skipped_str} - {usage_str}")

    if args.output:
        print(f"\nResults logged to {args.output}")


if __name__ == "__main__":
    main()

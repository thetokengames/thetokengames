import re
import sys
import json
import argparse
from datetime import datetime

from llm_utils import (
    get_llm, invoke_with_tracking, checkSolution, findSolution,
    MODEL_PRICING, token_usage, calculate_cost, format_usage,
    get_reasoning_tokens
)


def get_reasoning_text(raw):
    """Extract reasoning text from a raw response dict."""
    if not isinstance(raw, dict):
        return ""
    # OpenRouter: additional_kwargs.reasoning.content
    ak = raw.get("additional_kwargs") or {}
    reasoning_obj = ak.get("reasoning") or {}
    if isinstance(reasoning_obj, dict):
        reasoning_content = reasoning_obj.get("content") or []
        text = "".join(
            (item.get("text") or "") for item in reasoning_content
            if isinstance(item, dict)
        )
        if text:
            return text
    # Anthropic: thinking content blocks
    blocks = raw.get("content") or []
    if isinstance(blocks, list):
        text = "".join(
            (b.get("thinking") or "") for b in blocks
            if isinstance(b, dict) and b.get("type") == "thinking"
        )
        if text:
            return text
    return ""

def combine_usage(a, b):
    return {
        "input_tokens": a["input_tokens"] + b["input_tokens"],
        "output_tokens": a["output_tokens"] + b["output_tokens"],
    }

PAST_PUZZLE = """## Puzzle

### Proposer
This is {proposer} puzzle.

### Code and Explanation
```python
{code}
```
### Explanation
{expl}

### Outcome
{outcome}

"""

inventerPrompt = """Let's play a game.

# Rules
We take turns. One player, the proposer, gives the other a code puzzle with a Python function called `mystery` that takes a value x (either string or number) and returns a boolean. The other player, the solver, have to find a value x for which the function returns True. The proposer's job is to make the puzzle as hard as they can, i.e. it should be very difficult to find out which values the function returns True for. To prove that the puzzle is solvable, the proposer must give a sample solution. If the sample solution turns out to be wrong, the proposer is penalized.

# Instructions
You are the proposer. I am the solver.

Respond with a Python code block first, then explain your puzzle. Everything outside the code block is private -- I cannot see it.
The last line of your response should contain only: `SOLUTION: x` where x is the value of a sample solution for your puzzle. Ensure that `mystery(x)` is valid Python code given your x. For example, it is very important to use quotes if you believe `x` is a string.

Please make sure that your sample solution works -- you will be penalized if not. Do not give multiple code blocks or puzzles.
"""

def promptInventPuzzle(past_puzzles, player):
    if not withHistory:
        return inventerPrompt
    past = "".join(list(map(lambda x:
               PAST_PUZZLE.format(code=x[0],
                                  expl=(x[1]
                                        if x[3] == player
                                        else "(Hidden to you, since this is my puzzle.)"),
                                  outcome=x[2],
                                  proposer=("YOUR" if x[3] == player else "MY")),
               past_puzzles)))
    if past == "":
        past = "None, this is the first turn."
    return inventerPrompt + f"""
# History
Here are some puzzles from past turns of the game. You should NOT repeat them, but you can learn from them to ensure your puzzle is difficult enough.

{past}

# Reminder
Please ensure your response ends with this line: `SOLUTION: x` where x is the value of a sample solution for your puzzle.
"""

def promptSolvePuzzle(puzzle):
    return f"Here's a Python function that takes a value x and returns a boolean. Please give me a value for x such that mystery(x) is True. The last line of your response should be: `SOLUTION: x` where x is the value. Ensure that `mystery(x)` is valid Python code given your x. For example, if you believe `x` is a string, use quotes. Example: `SOLUTION: \"Hello, world!\"`\n\n```python\n{puzzle}\n```"

# Per-turn usage records for final summary
turn_usage_records = []

def generate_log_filenames(model_a_name, model_b_name):
    # Sanitize model names for filenames
    def sanitize(name):
        # Remove or replace characters that aren't filesystem-friendly
        return re.sub(r'[^\w\-.]', '_', name)

    sanitized_a = sanitize(model_a_name)
    sanitized_b = sanitize(model_b_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_name = f"{sanitized_a}_vs_{sanitized_b}_{timestamp}"
    json_filename = f"{base_name}.jsonl"
    text_filename = f"{base_name}.log"
    raw_filename = f"{base_name}.raw.txt"

    return json_filename, text_filename, raw_filename

def interpretInventerMsg(response):
    pattern = r'```python(.*?)```(.*?)SOLUTION: (.*?)(?:\n|$)'
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) != 1:
        print(response)
        raise ValueError("Inventer message format wrong.")
    return matches[0][0], matches[0][1], matches[0][2]

def runGameTurn(modelInventer, modelSolver, history, player, log_file, raw_file, inventer_name, solver_name):
    print(f"{inventer_name}'s turn to invent a puzzle...")
    print(f"{inventer_name}'s turn to invent a puzzle...", file=log_file)

    prompt = promptInventPuzzle(history, player)
    print('*** inventor prompt', inventer_name, file=raw_file)
    print(prompt, file=raw_file)
    print('*** END', file=raw_file)

    puzzle, puzzle_raw, inventer_usage = invoke_with_tracking(modelInventer, prompt, inventer_name)
    print('*** inventor', inventer_name, file=raw_file)
    print(puzzle, file=raw_file)
    print('*** END', file=raw_file)

    reasoning = get_reasoning_text(puzzle_raw)
    if reasoning:
        print('*** inventor reasoning', inventer_name, file=raw_file)
        print(reasoning, file=raw_file)
    print('*** END', file=raw_file)

    code, explanation, solution = interpretInventerMsg(puzzle)
    print(puzzle, "\nChecking sample solution ", solution)
    print(puzzle, "\nChecking sample solution ", solution, file=log_file)
    try:
        ex = False
        check = checkSolution(code, solution)
    except RuntimeError as e:
        check = False
        ex = True
        print(f"RuntimeError while checking sample -- penalizing!\nError: {e}")
        print(f"RuntimeError while checking sample -- penalizing!\nError: {e}", file=log_file)
    if not check:
        print("Sample solution is wrong -- this turn ends early.")
        print("Sample solution is wrong -- this turn ends early.", file=log_file)
        return "puzzle-wrong", [(code, explanation, "Sample solution wrong -- penalized", player)], {
            "inventer": inventer_name,
            "solver": solver_name,
            "code": code,
            "puzzle_raw": puzzle_raw,
            "solver_raw": None,
            "explanation": explanation,
            "solver_output": None,
            "sample_solution": solution,
            "solver_attempt": None,
            "outcome": "puzzle-wrong" if not ex else "puzzle-wrong-ex",
            "inventer_usage": inventer_usage,
            "solver_usage": None
        }
    print(f"Puzzle looks good; {solver_name}'s turn to solve...")
    print(f"Puzzle looks good; {solver_name}'s turn to solve...", file=log_file)
    solverOutput, solver_raw, solver_usage = invoke_with_tracking(modelSolver, promptSolvePuzzle(code), solver_name)

    print('*** solver', solver_name, file=raw_file)
    print(solverOutput, file=raw_file)
    reasoning = get_reasoning_text(solver_raw)
    if reasoning:
        print('*** solver reasoning', solver_name, file=raw_file)
        print(reasoning, file=raw_file)
    print('*** END', file=raw_file)

    attempt = findSolution(solverOutput)
    print(f"{solver_name} says {attempt}. Checking.")
    print(f"{solver_name} says {attempt}. Checking.", file=log_file)
    try:
        ex = False
        check = checkSolution(code, attempt)
    except RuntimeError as e:
        check = False
        ex = True
        print(f"RuntimeError while checking sample -- marking as wrong!\nError: {e}")
        print(f"RuntimeError while checking sample -- marking as wrong!\nError: {e}", file=log_file)
    if check:
        print("Success!")
        print("Success!", file=log_file)
        return "success", [(code, explanation, "Solved", player)], {
            "inventer": inventer_name,
            "solver": solver_name,
            "code": code,
            "puzzle_raw": puzzle_raw,
            "solver_raw": solver_raw,
            "explanation": explanation,
            "solver_output": solverOutput,
            "sample_solution": solution,
            "solver_attempt": attempt,
            "outcome": "success",
            "inventer_usage": inventer_usage,
            "solver_usage": solver_usage
        }
    else:
        print("Failure!")
        print("Failure!", file=log_file)
        return "failure", [(code, explanation, "Not solved", player)], {
            "inventer": inventer_name,
            "solver": solver_name,
            "code": code,
            "explanation": explanation,
            "puzzle_raw": puzzle_raw,
            "solver_raw": solver_raw,
            "solver_output": solverOutput,
            "sample_solution": solution,
            "solver_attempt": attempt,
            "outcome": "failure" if not ex else "failure-ex",
            "inventer_usage": inventer_usage,
            "solver_usage": solver_usage
        }

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run an LLM code puzzle arena competition')
parser.add_argument('--model-a', type=str, default='claude-opus-4-5-20251101',
                    help='Model A identifier (default: claude-opus-4-5-20251101)')
parser.add_argument('--model-b', type=str, default='gpt-5.2',
                    help='Model B identifier (default: gpt-5.2)')
parser.add_argument('--turns', type=int, default=5,
                    help='Number of turn pairs to run (default: 5)')
parser.add_argument('--json-log', type=str, default=None,
                    help='JSON log filename (default: auto-generated)')
parser.add_argument('--text-log', type=str, default=None,
                    help='Text log filename (default: auto-generated)')
parser.add_argument('--summary-log', type=str, default="summary.txt",
                    help='Summary log filename (default: "summary.txt")')
parser.add_argument('--raw-log', type=str, default=None,
                    help='Raw response log filename (default: auto-generated)')
parser.add_argument('--history', action=argparse.BooleanOptionalAction, default=True,
                    help='Play with history (default: True)')
args = parser.parse_args()

modelA = get_llm(args.model_a)
modelB = get_llm(args.model_b)

modelA_name = args.model_a
modelB_name = args.model_b

# Use provided log filenames or generate them
if args.json_log or args.text_log or args.raw_log:
    auto_json, auto_text, auto_raw = generate_log_filenames(modelA_name, modelB_name)
    json_log_file = args.json_log if args.json_log else auto_json
    text_log_file = args.text_log if args.text_log else auto_text
    raw_log_file = args.raw_log if args.raw_log else auto_raw
else:
    json_log_file, text_log_file, raw_log_file = generate_log_filenames(modelA_name, modelB_name)

log_file = open(text_log_file, 'w', encoding='utf-8')
json_file = open(json_log_file, 'w', encoding='utf-8')
raw_file = open(raw_log_file, 'w', encoding='utf-8')

scoreA = 0
penaltyA = 0
scoreB = 0
penaltyB = 0
history = []
puzzle_number = 0
failCount = 0
print(args.history)
withHistory = args.history

def tryRunGameTurn(modelInventer, modelSolver, history, player, log_file, raw_file, inventerName, solverName, n=0):
    try:
        return runGameTurn(modelInventer, modelSolver, history, player, log_file, raw_file, inventerName, solverName)
    except Exception as e:
        print(f"UNHANDLED EXCEPTION, error: {e}")
        print(f"UNHANDLED EXCEPTION, error: {e}", file=log_file)
        global failCount
        failCount += 1
        if n > 2:
            print(f"Giving up")
            print(f"Giving up", file=log_file)
            with open(args.summary_log, 'a') as file:
                print(f"{modelA_name} ./. {modelB_name} Giving up [see log at {text_log_file}]\n", file=file)
            sys.exit(1)
        else:
            return tryRunGameTurn(modelInventer, modelSolver, history, player, log_file, raw_file, inventerName, solverName, n+1)

def finishTurn(letter, turn_data):
    global puzzle_number

    print(f"New score after this turn: {scoreA-penaltyA} ({scoreA}-{penaltyA}) vs {scoreB-penaltyB} ({scoreB}-{penaltyB})")
    print(f"New score after this turn: {scoreA-penaltyA} ({scoreA}-{penaltyA}) vs {scoreB-penaltyB} ({scoreB}-{penaltyB})", file=log_file)

    # Print turn token usage
    inventer_name = turn_data["inventer"]
    solver_name = turn_data["solver"]
    inventer_usage = turn_data.get("inventer_usage") or {"input_tokens": 0, "output_tokens": 0}
    solver_usage = turn_data.get("solver_usage") or {"input_tokens": 0, "output_tokens": 0}

    inv_str = format_usage(inventer_name, inventer_usage["input_tokens"], inventer_usage["output_tokens"])
    print(f"  {inventer_name} (inventer): {inv_str}", file=log_file)
    print(f"  {inventer_name} (inventer): {inv_str}")

    if solver_usage["input_tokens"] > 0 or solver_usage["output_tokens"] > 0:
        sol_str = format_usage(solver_name, solver_usage["input_tokens"], solver_usage["output_tokens"])
        print(f"  {solver_name} (solver): {sol_str}", file=log_file)
        print(f"  {solver_name} (solver): {sol_str}")

    # Record for final summary
    turn_usage_records.append({
        "turn": puzzle_number,
        "inventer": inventer_name,
        "solver": solver_name,
        "inventer_usage": inventer_usage,
        "solver_usage": solver_usage,
        "outcome": turn_data["outcome"]
    })

    # Write JSON log for this turn
    turn_data["puzzle_number"] = puzzle_number
    turn_data["player"] = letter
    turn_data["score_a"] = scoreA
    turn_data["penalty_a"] = penaltyA
    turn_data["score_b"] = scoreB
    turn_data["penalty_b"] = penaltyB
    json_file.write(json.dumps(turn_data) + "\n")
    json_file.flush()
    log_file.flush()

    puzzle_number += 1

print("Before we start, let's ensure Python works.")
print("If this fails, try \"docker run  --rm   --network none   --memory=256m   --pids-limit=64   --cpus=1   --read-only -i  python:3.12 python --version\"")
assert(checkSolution("def mystery(x):\n\treturn x", "True"))

for i in range(args.turns):
    result, puzzle, turn_data = tryRunGameTurn(modelA, modelB, history, "a", log_file, raw_file, modelA_name, modelB_name)
    history += puzzle
    if result == "success":
        pass
    elif result == "failure":
        scoreA += 1
    elif result == "puzzle-wrong":
        penaltyA += 1
    else:
        raise ValueError("unknown outcome: {result}")
    finishTurn("a", turn_data)
    result, puzzle, turn_data = tryRunGameTurn(modelB, modelA, history, "b", log_file, raw_file, modelB_name, modelA_name)
    history += puzzle
    if result == "success":
        pass
    elif result == "failure":
        scoreB += 1
    elif result == "puzzle-wrong":
        penaltyB += 1
    else:
        raise ValueError("unknown outcome: {result}")
    finishTurn("b", turn_data)

print(f"Logs written to {text_log_file} and {json_log_file}")

# Print token usage summary (one line per turn)
print("\n=== Token Usage Summary ===")
for rec in turn_usage_records:
    inv = rec["inventer_usage"]
    sol = rec["solver_usage"]
    inv_cost = calculate_cost(rec["inventer"], inv["input_tokens"], inv["output_tokens"])
    sol_cost = calculate_cost(rec["solver"], sol["input_tokens"], sol["output_tokens"]) if sol else None
    total_cost = (inv_cost or 0) + (sol_cost or 0)
    cost_str = f"${total_cost:.2f}" if (inv_cost is not None or sol_cost is not None) else "?"
    total_in = inv["input_tokens"] + (sol["input_tokens"] if sol else 0)
    total_out = inv["output_tokens"] + (sol["output_tokens"] if sol else 0)
    print(f"Turn {rec['turn']}: {rec['inventer']} vs {rec['solver']} - {total_in:,} in / {total_out:,} out ({cost_str}) [{rec['outcome']}]")
    print(f"Turn {rec['turn']}: {rec['inventer']} vs {rec['solver']} - {total_in:,} in / {total_out:,} out ({cost_str}) [{rec['outcome']}]", file=log_file)

# Print totals by model
print("\n=== Totals by Model ===")
for model_name, usage in token_usage.items():
    print(f"{model_name}: {format_usage(model_name, usage['input_tokens'], usage['output_tokens'])}")
    print(f"{model_name}: {format_usage(model_name, usage['input_tokens'], usage['output_tokens'])}", file=log_file)

with open(args.summary_log, 'a') as file:
    file.write(f"{modelA_name} ./. {modelB_name}     ({scoreA}-{penaltyA}) vs ({scoreB}-{penaltyB})     [{failCount} failed, see {text_log_file}]\n")
    log_file.write(f"{modelA_name} ./. {modelB_name}     ({scoreA}-{penaltyA}) vs ({scoreB}-{penaltyB})     [{failCount} failed, see {text_log_file}]\n")

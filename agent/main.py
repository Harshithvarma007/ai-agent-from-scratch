"""CLI entry point: python -m agent "task" [--provider ollama] [--mode native] [--yolo]
(or `analyst-agent "task"` once the package is installed)."""
import argparse
import json
import os
import sys
import time

from . import __version__, tools
from .agent import run_agent
from .config import PROVIDERS, get_provider
from .tools import ALL_TOOLS, Registry

COLOR = {"thought": "\033[36m", "action": "\033[33m", "repeat_guard": "\033[31m", "end": "\033[0m"}


def make_trace_printer():
    def trace(kind, **info):
        if kind == "thought":
            print(f"{COLOR['thought']}[thought]{COLOR['end']} {info['content'][:300]}")
        elif kind == "action":
            print(f"{COLOR['action']}[action]{COLOR['end']} {info['name']}({info['args']})")
            print(f"[observation] {str(info['result'])[:300]}")
        elif kind == "rescued_tool_call_from_text":
            print(f"{COLOR['action']}[note]{COLOR['end']} tool call recovered from plain text, not tool_calls")
        elif kind == "repeat_guard":
            print(f"{COLOR['repeat_guard']}[loop guard]{COLOR['end']} refused to repeat {info['name']}({info['args']})")
    return trace


def make_confirm(yolo):
    def confirm(name, args):
        if yolo:
            return True
        reply = input(f"Allow {name}({args})? [y/N] ").strip().lower()
        return reply == "y"
    return confirm


def main(argv=None):
    p = argparse.ArgumentParser(prog="analyst-agent",
                                description="A data-analyst AI agent, from scratch, on free LLMs.")
    p.add_argument("task", nargs="?", default=(
        "There is a file at data/payments.csv. Load it with pandas and tell me the total "
        "eur_amount across all transactions, rounded to 2 decimal places."
    ))
    p.add_argument("--provider", default=os.environ.get("AGENT_PROVIDER", "ollama"),
                    choices=list(PROVIDERS))
    p.add_argument("--model", help="override the provider's default model (or set AGENT_MODEL)")
    p.add_argument("--mode", default="native", choices=["native", "react", "constrained"])
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--yolo", action="store_true", help="skip confirmation before dangerous tools")
    p.add_argument("--sandbox", default=tools.SANDBOX_MODE, choices=["subprocess", "docker"],
                    help="how run_python executes code; see sandbox_ladder.py")
    p.add_argument("--runs-dir", default=os.environ.get("AGENT_RUNS_DIR", "runs"),
                    help="where each run's full trajectory is saved as JSON")
    p.add_argument("--no-save", action="store_true", help="don't save the trajectory")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = p.parse_args(argv)

    try:
        provider = get_provider(args.provider)
    except ValueError as e:
        p.error(str(e))
    if args.model:
        provider.model = args.model
    tools.SANDBOX_MODE = args.sandbox

    mode = "react+constrained" if args.mode == "constrained" else args.mode
    if mode == "react+constrained" and not provider.supports_constrained:
        p.error("--mode constrained only works with --provider ollama")

    try:
        result = run_agent(provider, args.task, Registry(*ALL_TOOLS), mode=mode,
                           max_steps=args.max_steps, confirm=make_confirm(args.yolo),
                           trace=make_trace_printer())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except OSError as e:  # URLError/HTTPError are OSErrors: no server, bad key, network down
        hint = " Is Ollama running? Try `ollama serve`." if provider.name == "ollama" else ""
        print(f"error: could not reach {provider.name} at {provider.base_url}: {e}.{hint}",
              file=sys.stderr)
        return 2

    print(f"\n=== {result['outcome']} ===\n{result['answer']}")

    if not args.no_save:
        os.makedirs(args.runs_dir, exist_ok=True)
        run_path = os.path.join(args.runs_dir, f"{int(time.time())}.json")
        with open(run_path, "w") as f:
            json.dump({"provider": provider.name, "model": provider.model, "mode": mode, **result},
                      f, indent=2)
        print(f"(trajectory saved to {run_path})")
    return 0 if result["outcome"] == "final" else 1


if __name__ == "__main__":
    sys.exit(main())

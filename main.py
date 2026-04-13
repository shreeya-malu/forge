"""
main.py
Forge CLI entry point.
Run from terminal or Colab with:
  python main.py --prompt "Build a REST API for..." --mode cli
  python main.py --mode dashboard
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path


def run_cli(prompt: str, priority_weights: dict = None):
    """Run the full Forge pipeline in CLI mode with live stdout logging."""
    from core.graph import build_graph, get_initial_state
    from core.observability import init_wandb, init_langsmith, init_log
    import yaml

    cfg = yaml.safe_load(open(Path(__file__).parent / "config.yaml"))

    # Init observability
    init_log(cfg["output"]["log_file"])

    if os.environ.get("WANDB_API_KEY"):
        init_wandb(cfg["observability"]["wandb_project"], {
            "prompt": prompt[:200],
            "priority_weights": priority_weights or {},
        })

    if os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY"):
        init_langsmith(cfg["observability"]["langsmith_project"])

    initial_state = get_initial_state(prompt, priority_weights)
    graph = build_graph()

    print("\n" + "=" * 60)
    print("  FORGE — Engineering Intelligence System")
    print("=" * 60)
    print(f"\nPrompt: {prompt[:120]}...\n")

    final_state = None
    for step in graph.stream(initial_state):
        for node_name, state in step.items():
            phase = state.get("phase", "")
            confidence = state.get("current_confidence", 0)
            print(f"\n[{node_name}] phase={phase} confidence={confidence}%")
            final_state = state

    if final_state:
        summary = final_state.get("workflow_summary", {})
        print("\n" + "=" * 60)
        print("  WORKFLOW COMPLETE")
        print("=" * 60)
        print(json.dumps(summary, indent=2))

        # Show generated file list
        files = list(final_state.get("generated_files", {}).keys())
        print(f"\nGenerated {len(files)} files:")
        for f in files:
            print(f"  ./generated_project/{f}")

        print(f"\nSummary: ./outputs/summary.json")
        if final_state.get("security_report"):
            print(f"Security report: ./outputs/security_report.json")

    return final_state


def run_dashboard(share: bool = True, port: int = 7860):
    """Launch the Gradio dashboard."""
    from dashboard.app import launch
    print("\n🚀 Starting Forge Dashboard...")
    launch(share=share, port=port)


def main():
    parser = argparse.ArgumentParser(description="Forge — Engineering Intelligence System")
    parser.add_argument("--mode", choices=["cli", "dashboard"], default="dashboard",
                        help="Run mode: cli (batch) or dashboard (interactive UI)")
    parser.add_argument("--prompt", type=str, default="",
                        help="Project specification (required for CLI mode)")
    parser.add_argument("--priority-security", type=float, default=0.2)
    parser.add_argument("--priority-quality",  type=float, default=0.25)
    parser.add_argument("--priority-tests",    type=float, default=0.25)
    parser.add_argument("--priority-speed",    type=float, default=0.2)
    parser.add_argument("--priority-simplicity", type=float, default=0.1)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-share", action="store_true",
                        help="Disable Gradio public share link")
    args = parser.parse_args()

    if args.mode == "cli":
        if not args.prompt:
            print("Error: --prompt is required for CLI mode")
            sys.exit(1)
        weights = {
            "speed": args.priority_speed,
            "quality": args.priority_quality,
            "test_coverage": args.priority_tests,
            "security": args.priority_security,
            "simplicity": args.priority_simplicity,
        }
        run_cli(args.prompt, weights)
    else:
        run_dashboard(share=not args.no_share, port=args.port)


if __name__ == "__main__":
    main()

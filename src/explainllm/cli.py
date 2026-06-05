"""Command-line interface for ExplainLLM."""

import argparse
import sys
import json


def main():
    parser = argparse.ArgumentParser(
        prog="explainllm",
        description="LLM Token Relevance Calculator using DL-Backtrace patterns",
    )
    sub = parser.add_subparsers(dest="command")

    # --- run subcommand ---
    run = sub.add_parser("run", help="Compute token relevance for a prompt")
    run.add_argument("--prompt", required=True, help="Input prompt text")
    run.add_argument(
        "--model", default=None,
        help="HuggingFace model ID (default: meta-llama/Llama-3.2-1B-Instruct)",
    )
    run.add_argument(
        "--device", default="cpu",
        help="Device to run on: cpu, cuda, cuda:0, etc. (default: cpu)",
    )
    run.add_argument(
        "--dtype", default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Torch dtype (default: float32)",
    )
    run.add_argument(
        "--max-new-tokens", type=int, default=20,
        help="Maximum tokens to generate (default: 20)",
    )
    run.add_argument(
        "--method", default="combined",
        choices=["attention", "gradient", "rollout", "combined"],
        help="Relevance method (default: combined)",
    )
    run.add_argument(
        "--alpha", type=float, default=0.5,
        help="Attention weight in combined mode (default: 0.5)",
    )
    run.add_argument(
        "--export", metavar="FILE",
        help="Export results to JSON file",
    )
    run.add_argument(
        "--heatmap", metavar="FILE",
        help="Save relevance heatmap to file",
    )
    run.add_argument(
        "--step0-bar", metavar="FILE",
        help="Save step-0 relevance bar plot to file",
    )
    run.add_argument(
        "--arc-graph", metavar="FILE",
        help="Save arc diagram graph to file",
    )
    run.add_argument(
        "--visualize", action="store_true",
        help="Print text visualization to stdout",
    )
    run.add_argument(
        "--verbose", action="store_true",
        help="Print token-by-token generation info",
    )

    # --- version subcommand ---
    sub.add_parser("version", help="Show version")

    args = parser.parse_args()

    if args.command == "version":
        from explainllm._version import __version__
        print(f"ExplainLLM {__version__}")
        return

    if args.command != "run":
        parser.print_help()
        sys.exit(1)

    from explainllm.config import RelevanceConfig
    from explainllm.convenience import calculate_relevance

    config = RelevanceConfig(
        model_id=args.model,
        device=args.device,
        torch_dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        relevance_method=args.method,
        alpha=args.alpha,
    )

    print(f"Loading model: {config.model_id}")
    print(f"Device: {config.device}  |  Dtype: {config.torch_dtype}")
    print(f"Prompt: {args.prompt!r}")
    print()

    result = calculate_relevance(
        prompt=args.prompt,
        config=config,
        verbose=args.verbose,
    )

    print(f"\nGenerated: {result['generated_text']}")

    if args.visualize:
        from explainllm.viz.text import visualize_all_steps
        print()
        print(visualize_all_steps(result, top_k=5))

    if args.export:
        from explainllm.export import export_relevance_json
        export_relevance_json(result, args.export)

    if args.heatmap:
        from explainllm.viz.heatmap import plot_relevance_heatmap
        plot_relevance_heatmap(result, save_path=args.heatmap)

    if args.step0_bar:
        from explainllm.viz.heatmap import plot_single_step_heatmap
        plot_single_step_heatmap(result, step=0, save_path=args.step0_bar)

    if args.arc_graph:
        from explainllm.viz.graph import plot_relevance_graph
        plot_relevance_graph(result, save_path=args.arc_graph)


if __name__ == "__main__":
    main()

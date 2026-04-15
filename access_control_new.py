"""
Access Control DAG to Knowledge Graph - Main Entry Point

Converts Access Control DAG images into structured knowledge graphs using
vision models (e.g. gpt-5-nano, gpt-4o-mini) via the OpenAI API.

Configuration: set OPENAI_API_KEY in a `.env` file or the environment
(use a placeholder like "<OPENAI_API_KEY>" in docs only — never commit real keys).

Three primary modes (--method):
    1. extract_entities         - Identify nodes (entities) in the graph
    2. relation_classification  - Binary classification of relationships (needs entities)
    3. relation_extraction      - End-to-end nodes + edges from the image

Key Command-Line Options:
    --input PATH       Image file or directory (default: project `datasets/` folder)
    --output PATH      Output file or directory (default: project `experiments/` folder)
    --model MODEL      Vision model (default: gpt-5-nano; see --help for choices)
    --image_detail     Vision API detail: low (default) or high
    --method METHOD    extract_entities | relation_classification | relation_extraction | ...
    --workers N        Parallel workers for batch (default: 4). Use 1 for sequential.

Quick Start:
    export OPENAI_API_KEY="<OPENAI_API_KEY>"
    python access_control_new.py --method extract_entities
    python access_control_new.py --help

Full GitHub-style setup, examples with path placeholders, and CLI tables: README.md

Architecture:
    access_control_new.py -> src/cli.py -> src/core_processor.py -> src/processing_strategies.py
"""

import sys
from dataclasses import dataclass
from typing import Optional
from src.cli import main as cli_main

from src.config import DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_PATH, DEFAULT_MODEL

@dataclass
class ScriptArguments:
    """Arguments for the Access Control processing script (mirrors main CLI flags)."""
    input: str = DEFAULT_INPUT_PATH
    output: str = DEFAULT_OUTPUT_PATH
    method: str = "extract_entities"
    few_shot: str = "zero"
    with_legend: bool = True
    image_detail: str = "low"
    model: str = DEFAULT_MODEL  # e.g. gpt-5-nano, gpt-4o-mini
    workers: int = 4
    relation_source: str = "ground_truth"
    entities_input: Optional[str] = None
    gt_input: Optional[str] = None
    subset_size: Optional[int] = None
    comprehensive_eval: bool = False
    fuzzy_matching: bool = False

def run_with_args(args: ScriptArguments):
    """Run the main pipeline with provided arguments."""
    # Convert dataclass to sys.argv format for argparse in src/cli.py
    sys_args = [sys.argv[0]]
    sys_args.extend(["--input", args.input])
    sys_args.extend(["--output", args.output])
    sys_args.extend(["--method", args.method])
    sys_args.extend(["--few_shot", args.few_shot])
    sys_args.extend(["--image_detail", args.image_detail])
    sys_args.extend(["--model", args.model])
    sys_args.extend(["--workers", str(args.workers)])
    sys_args.extend(["--relation_source", args.relation_source])
    
    if args.with_legend:
        sys_args.append("--with_legend")
    else:
        sys_args.append("--no_legend")
        
    if args.entities_input:
        sys_args.extend(["--entities_input", args.entities_input])

    if args.gt_input:
        sys_args.extend(["--gt_input", args.gt_input])

    if args.subset_size is not None:
        sys_args.extend(["--subset_size", str(args.subset_size)])

    if args.comprehensive_eval:
        sys_args.append("--comprehensive_eval")

    if args.fuzzy_matching:
        sys_args.append("--fuzzy_matching")
        
    # Override sys.argv and call cli_main
    sys.argv = sys_args
    cli_main()

if __name__ == "__main__":
    # If no command line arguments are provided, use default ScriptArguments
    if len(sys.argv) == 1:
        default_args = ScriptArguments()
        run_with_args(default_args)
    else:
        # Otherwise, use the standard CLI parsing
        cli_main()

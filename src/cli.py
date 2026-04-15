"""
Command-Line Interface for Access Control DAG Processing

This module provides a clean command-line interface for the Access Control
processing system with simplified argument parsing and clear usage patterns.
"""

import argparse
from pathlib import Path
from typing import Optional

from .config import (
    ProcessingConfig, APIConfig, ImageConfig, DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_PATH,
    PROJECT_ROOT, DEFAULT_MODEL, SUPPORTED_VISION_MODELS
)
from .core_processor import AccessControlProcessor
from .file_utils import find_ground_truth_file
from .access_prompt import (
    generate_few_shot_examples_for_entity_extraction,
    generate_few_shot_examples_for_relation_classification,
    generate_few_shot_examples_for_path_generation,
    FEW_SHOT_JSON_PATH
)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert Access Control DAG images to Knowledge Graphs using Vision LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  extract_entities       Entities extraction (identify nodes)
  relation_classification  Binary classification of relationships (needs entities)
  relation_extraction    End-to-end: nodes + edges from image

Examples:
  python access_control_new.py --method extract_entities
  python access_control_new.py --method relation_classification --entities_input /path/to/entities
  python access_control_new.py --method relation_extraction --input graph.png --output result.json
        """
    )

    # Input/Output arguments
    parser.add_argument("--input", type=str,
                       default=DEFAULT_INPUT_PATH,
                       help="Input folder or image file (for images)")
    parser.add_argument("--output", type=str,
                       default=DEFAULT_OUTPUT_PATH,
                       help="Output folder or JSON file")
    parser.add_argument("--entities_input", type=str,
                       default=None,
                       help="""Entity input folder for relation_classification (optional).
                         If not provided, uses --relation_source to determine entity location.
                         For ground_truth: path to JSON folder (e.g., SubgraphsWithTriplesJSON/subgraphs_001/)
                         For predicted: path to extract_entities results folder""")
    parser.add_argument("--gt_input", type=str,
                       default=None,
                       help="Explicit ground truth directory for evaluation (optional)")

    # Legend arguments
    parser.add_argument("--with_legend", action="store_true",
                       default=True,
                       help="Process images with legend (default: True)")
    parser.add_argument("--no_legend", action="store_false",
                       dest="with_legend",
                       help="Process images without legend")

    # Method arguments (three modes: entities extraction, relation classification, relation extraction end-to-end)
    parser.add_argument("--method", type=str,
                       choices=["extract_entities", "relation_classification", "relation_extraction", "extract_relation", "enumerate_paths", "path_generation"],
                       default="extract_entities",
                       help="Processing method: extract_entities | relation_classification | relation_extraction (end-to-end)")

    # Few-shot mode
    parser.add_argument("--few_shot", type=str,
                       choices=["zero", "few"],
                       default="zero",
                       help="Few-shot mode: 'zero' or 'few'")

    # Relation source for relation classification
    parser.add_argument("--relation_source", type=str,
                       choices=["ground_truth", "predicted"],
                       default="ground_truth",
                       help="""Entity source for relation classification:
                         'ground_truth' - Use entities from ground truth JSON files (default)
                         'predicted' - Use entities from extract_entities results
                         Note: For 'predicted', run extract_entities first to generate prediction files""")

    # Subset testing for relation classification
    parser.add_argument("--subset_size", type=int,
                       default=None,
                       help="Randomly select N relations per graph for subset testing (default: None for full testing)")

    # Comprehensive evaluation
    parser.add_argument("--comprehensive_eval", action="store_true",
                       default=False,
                       help="Enable comprehensive evaluation covering ALL possible relations in the graph (Context7 approach). "
                            "Evaluates complete graph structure rather than just processed subset. "
                            "Provides true performance metrics for entire domain.")

    # Fuzzy entity matching for evaluation
    parser.add_argument("--fuzzy_matching", action="store_true",
                       default=False,
                       help="Enable fuzzy entity name matching in evaluation to handle OCR typos and spelling variations. "
                            "Uses similarity matching to reduce false negatives from entity name mismatches.")

    # Image quality/detail level
    parser.add_argument("--image_detail", type=str,
                       choices=["low", "high"],
                       default="low",
                       help="""Image detail level for vision API (default: low):
                         'low' - Cost-efficient (512px, ~2,833 tokens/image)
                         'high' - Higher quality (2048px with tiling, ~53,836 tokens/image)
                         Note: High detail is the default for better image analysis.""")

    # Parallel workers for batch processing
    parser.add_argument("--workers", type=int,
                       default=4,
                       help="Max parallel workers for batch processing (default: 4). Use 1 for sequential.")

    # Vision model selection
    parser.add_argument("--model", type=str,
                       choices=SUPPORTED_VISION_MODELS,
                       default=DEFAULT_MODEL,
                       help=f"Vision model for image analysis (default: {DEFAULT_MODEL}). "
                            "Use with --image_detail low/high for cost vs quality tradeoff.")

    return parser


def setup_few_shot_examples(method: str) -> Optional[list]:
    """
    Setup few-shot examples for Context7 sequential processing.

    Args:
        method: Processing method name

    Returns:
        List of few-shot examples or None
    """
    print(f"\n🔧 Generating two-shot examples for {method}...")

    if not Path(FEW_SHOT_JSON_PATH).exists():
        print(f"  ❌ Error: Few-shot ground truth not found at {FEW_SHOT_JSON_PATH}")
        return None

    try:
        if method == "extract_entities":
            examples = generate_few_shot_examples_for_entity_extraction()
        elif method == "relation_classification":
            examples = generate_few_shot_examples_for_relation_classification()
        elif method in ["enumerate_paths", "path_generation", "relation_extraction"]:
            examples = generate_few_shot_examples_for_path_generation()
        else:
            print(f"  Warning: Few-shot examples not supported for method '{method}'")
            return None

        if examples:
            print(f"  ✅ Generated {len(examples)} few-shot example messages (2 turns)\n")
            return examples
        else:
            print(f"  ❌ Failed to generate few-shot examples")
            return None

    except Exception as e:
        print(f"  ❌ Error generating few-shot examples: {e}")
        return None


def process_single_file(args, processor: AccessControlProcessor) -> None:
    """Process a single file with automatic evaluation."""
    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Find ground truth for evaluation
    gt_path = processor.find_ground_truth_file(args.input)

    if args.method == "extract_entities":
        result = processor.process_single_image(args.input, args.output)

        if result.success:
            print(f"✅ Successfully extracted {result.entities_extracted} entities")
            print(f"📄 Results saved to: {result.output_path}")

            # Try automatic evaluation
            if gt_path:
                from .evaluation import evaluate_entity_extraction
                metrics = evaluate_entity_extraction(result.output_path, gt_path)
                if metrics:
                    print(f"📊 Evaluation - Precision: {metrics.precision:.4f}, Recall: {metrics.recall:.4f}, F1: {metrics.f1:.4f}")
        else:
            print(f"❌ Processing failed: {result.error_message}")

    elif args.method == "enumerate_paths":
        result = processor.process_single_image(args.input, args.output)

        if result.success:
            metadata = result.metadata
            print(f"✅ Successfully enumerated paths")
            print(f"📊 Found {metadata['nodes']} nodes, {metadata['edges']} edges, {metadata['paths']} paths")
            print(f"📄 Results saved to: {result.output_path}")

            # Try automatic evaluation
            if gt_path:
                from .evaluation import evaluate_entity_extraction
                metrics = evaluate_entity_extraction(result.output_path, gt_path)
                if metrics:
                    print(f"📊 Entity Evaluation - Precision: {metrics.precision:.4f}, Recall: {metrics.recall:.4f}, F1: {metrics.f1:.4f}")
        else:
            print(f"❌ Processing failed: {result.error_message}")

    elif args.method == "relation_classification":
        # For relation classification, process single image with batch entity pairs
        print("ℹ️  Processing relation classification for single image...")
        print(f"   Using {processor.processing_config.relation_source} entities as source")
        
        if args.entities_input:
            print(f"   Entities from: {args.entities_input}")

        if gt_path:
            result = processor._process_relation_classification_batch(
                args.input,
                gt_path,
                args.output,
                entities_input_dir=args.entities_input
            )

            if result.success:
                print(f"✅ Successfully processed relations")
                print(f"📄 Results saved to: {result.output_path}")

                # Show evaluation if available
                if result.metadata.get("evaluation_metrics"):
                    metrics = result.metadata["evaluation_metrics"]
                    print(f"📊 Evaluation - Precision: {metrics.precision:.4f}, Recall: {metrics.recall:.4f}, F1: {metrics.f1:.4f}")
            else:
                print(f"❌ Processing failed: {result.error_message}")
        else:
            print(f"❌ No ground truth file found for evaluation")

    elif args.method in ["path_generation", "relation_extraction"]:
        # For path_generation/relation_extraction (end-to-end), pass ground truth for evaluation
        result = processor.process_single_image(
            args.input,
            args.output,
            ground_truth_path=gt_path
        )

        if result.success:
            print(f"✅ Successfully generated paths")
            print(f"📄 Results saved to: {result.output_path}")

            # Show evaluation if available
            if result.metadata.get("evaluation"):
                eval_data = result.metadata["evaluation"]
                combined = eval_data.get("combined_metrics", {})
                if combined:
                    print(f"📊 Evaluation - Precision: {combined.get('precision', 0.0):.4f}, Recall: {combined.get('recall', 0.0):.4f}, F1: {combined.get('f1', 0.0):.4f}")
        else:
            print(f"❌ Processing failed: {result.error_message}")

    else:
        result = processor.process_single_image(args.input, args.output)
        if result.success:
            print(f"✅ Processing completed successfully")
            print(f"📄 Results saved to: {result.output_path}")
        else:
            print(f"❌ Processing failed: {result.error_message}")


def process_batch_directory(args, processor: AccessControlProcessor) -> None:
    """Process a batch directory."""
    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input}")

    # Check if it's SubgraphsWithTriples dataset root
    # Only treat as dataset root if it contains both SubgraphsWithTriplesImages and SubgraphsWithTriplesJSON
    is_dataset_root = (
        input_path.name in ["SubgraphsWithTriplesImages", "SubgraphsWithTriplesJSON"] or
        (input_path / "SubgraphsWithTriplesImages").exists() or
        (input_path / "SubgraphsWithTriplesJSON").exists()
    )

    if is_dataset_root and ("SubgraphsWithTriples" in str(input_path) or "datasets" in str(input_path)):
        print(f"🔍 Detected dataset structure")
        results = processor.process_subgraphs_dataset(
            str(input_path),
            output_subdir=args.output
        )
        print(f"✅ Processed {len(results)} subsets")
    else:
        print(f"🔍 Processing directory: {input_path.name}")
        # Use args.output as the full output directory (no subdir nesting)
        processor.processing_config.output_path = args.output
        result = processor.process_batch(
            str(input_path),
            output_subdir=None
        )
        print(f"✅ Processed {result.summary['successful']}/{result.summary['total_images']} images")


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Normalize method: relation_extraction (user-facing) -> path_generation (internal)
    if getattr(args, "method", None) == "relation_extraction":
        args.method = "path_generation"

    # Create configuration
    processing_config = ProcessingConfig(
        input_path=args.input,
        output_path=args.output,
        method=args.method,
        few_shot_mode=args.few_shot,
        relation_source=args.relation_source,
        entities_input=args.entities_input,
        gt_input=args.gt_input,
        subset_size=args.subset_size,
        comprehensive_eval=args.comprehensive_eval,
        with_legend=args.with_legend,
        fuzzy_matching=args.fuzzy_matching,
        max_workers=args.workers
    )

    # Validate configuration
    processing_config.validate()

    # Setup few-shot examples if requested
    if processing_config.few_shot_mode == "few":
        processing_config.few_shot_examples = setup_few_shot_examples(processing_config.method)
        mode_desc = "Few-shot (Context7 sequential)"
    else:
        mode_desc = "Zero-shot"

    print(f"🚀 Access Control DAG Processor")
    print(f"📋 Mode: {mode_desc}")
    print(f"🎯 Method: {processing_config.method.replace('_', ' ').title()}")
    print(f"🖼️  Image Detail: {args.image_detail.upper()} ({'cost-efficient' if args.image_detail == 'low' else 'high-quality'})")
    print(f"🤖 Model: {args.model}")
    print(f"⚡ Workers: {args.workers} (parallel)" if args.workers > 1 else "⚡ Workers: 1 (sequential)")
    print(f"📂 Input: {args.input}")
    
    # Show entities_input for relation_classification
    if processing_config.method == "relation_classification" and args.entities_input:
        print(f"📂 Entities Input: {args.entities_input} ({processing_config.relation_source})")

    # Show output path with appropriate labeling
    input_path = Path(args.input)
    if input_path.is_dir():
        print(f"📂 Output Directory: {args.output}")
    else:
        print(f"📂 Output File: {args.output}")

    print("-" * 50)

    # Create API and processor
    api_config = APIConfig(model=args.model, image_detail=args.image_detail)
    processor = AccessControlProcessor(api_config, processing_config)

    # Route processing based on input type
    input_path = Path(args.input)

    # Output folder logic for directories: save under experiments
    if input_path.is_dir():
        if args.output == DEFAULT_OUTPUT_PATH:
            # Default: save under experiments/<method>/<input_folder_name>
            experiments_base = PROJECT_ROOT / "experiments"
            input_folder_name = input_path.name
            if "SubgraphsWithTriples" in str(input_path) and input_path.name == "SubgraphsWithTriplesImages":
                input_folder_name = "SubgraphsWithTriples"
            new_output = experiments_base / processing_config.method / input_folder_name
            args.output = str(new_output)
            processing_config.output_path = args.output
            print(f"📂 Output directory: {args.output}")
        else:
            processing_config.output_path = args.output
    else:
        # For single file, ensure output_path is set in config
        if args.output:
            processing_config.output_path = args.output
            print(f"📂 Setting output path for single file: {args.output}")

    try:
        if input_path.is_file():
            # Single file processing
            process_single_file(args, processor)
        elif input_path.is_dir():
            # Batch directory processing
            process_batch_directory(args, processor)
        else:
            raise ValueError(f"Input path does not exist: {args.input}")

        print("\n🎉 Processing completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()

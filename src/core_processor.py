"""Core Processing Engine for Access Control DAG Analysis."""

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from .config import (
    ProcessingConfig, APIConfig, ImageConfig, BatchProcessingResult,
    ProcessingResult, EvaluationMetrics, PROJECT_ROOT, DATASET_SUBSETS
)
from .processing_strategies import ProcessingStrategyFactory
from .file_utils import (
    find_ground_truth_file, discover_image_files, create_output_directory,
    load_json, save_json
)
from .evaluation import (
    evaluate_entity_extraction, evaluate_relation_classification,
    print_aggregated_results_with_micro_macro,
    aggregate_evaluation_metrics, aggregate_evaluation_metrics_with_micro_macro,
    aggregate_evaluation_metrics_with_relations,
    write_evaluation_results_csv,
    append_performance_results_csv,
)


class AccessControlProcessor:
    """
    Main processing engine for Access Control DAG analysis.

    Coordinates between different processing strategies and handles
    batch processing, evaluation, and results aggregation.
    """

    def __init__(
        self,
        api_config: APIConfig,
        processing_config: ProcessingConfig,
        image_config: Optional[ImageConfig] = None
    ):
        self.api_config = api_config
        self.processing_config = processing_config
        self.image_config = image_config or ImageConfig()

        # Set random seed for reproducible results
        random.seed(42)

        # Initialize strategy
        self.strategy = ProcessingStrategyFactory.create_strategy(
            processing_config.method,
            api_config,
            processing_config
        )

        # Cost and timing tracking
        self.usage_stats = {
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_tokens': 0,
            'api_calls': 0,
            'start_time': None,
            'end_time': None
        }

    def start_timing(self):
        """Start timing the processing."""
        self.usage_stats['start_time'] = time.time()

    def end_timing(self):
        """End timing the processing."""
        self.usage_stats['end_time'] = time.time()

    def add_usage(self, usage_info: Dict):
        """Add usage information from an API call."""
        if usage_info:
            self.usage_stats['total_prompt_tokens'] += usage_info.get('prompt_tokens', 0)
            self.usage_stats['total_completion_tokens'] += usage_info.get('completion_tokens', 0)
            self.usage_stats['total_tokens'] += usage_info.get('total_tokens', 0)
            self.usage_stats['api_calls'] += 1

    def calculate_cost(self) -> Dict[str, float]:
        """Calculate total cost based on token usage.

        Returns:
            Dictionary with cost breakdown
        """
        # GPT-4o-mini pricing (as of 2024)
        # Input tokens: $0.150 per 1M tokens
        # Output tokens: $0.600 per 1M tokens

        input_cost_per_million = 0.150
        output_cost_per_million = 0.600

        input_tokens = self.usage_stats['total_prompt_tokens']
        output_tokens = self.usage_stats['total_completion_tokens']

        input_cost = (input_tokens / 1_000_000) * input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * output_cost_per_million
        total_cost = input_cost + output_cost

        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': self.usage_stats['total_tokens'],
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost,
            'api_calls': self.usage_stats['api_calls']
        }

    def get_processing_time(self) -> float:
        """Get total processing time in seconds."""
        if self.usage_stats['start_time'] and self.usage_stats['end_time']:
            return self.usage_stats['end_time'] - self.usage_stats['start_time']
        return 0.0

    def print_cost_and_time_summary(self):
        """Print a summary of cost and time at the end of processing."""
        processing_time = self.get_processing_time()
        cost_info = self.calculate_cost()

        print("\n" + "="*60)
        print("🕒 PROCESSING SUMMARY")
        print("="*60)
        print(f"⏱️  Total Processing Time: {processing_time:.2f} seconds")
        print(f"🔄 API Calls Made: {cost_info['api_calls']}")
        print(f"📊 Token Usage:")
        print(f"   • Input tokens: {cost_info['input_tokens']:,}")
        print(f"   • Output tokens: {cost_info['output_tokens']:,}")
        print(f"   • Total tokens: {cost_info['total_tokens']:,}")
        print(f"💰 Estimated Cost:")
        print(f"   • Input cost: ${cost_info['input_cost']:.4f}")
        print(f"   • Output cost: ${cost_info['output_cost']:.4f}")
        print(f"   • Total cost: ${cost_info['total_cost']:.4f}")
        print("="*60)

    def process_single_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        fuzzy_matching: bool = False,
        **kwargs
    ) -> ProcessingResult:
        """
        Process a single image.

        Args:
            image_path: Path to the image file
            output_path: Optional output path
            **kwargs: Additional arguments for the strategy

        Returns:
            ProcessingResult object
        """
        # Start timing for single image processing
        self.start_timing()

        result = self.strategy.process(image_path, output_path=output_path, fuzzy_matching=fuzzy_matching, **kwargs)

        # Track usage and end timing
        if result.success and result.metadata.get("usage"):
            self.add_usage(result.metadata["usage"])

        self.end_timing()
        self.print_cost_and_time_summary()

        return result

    def _process_one_image_in_batch(
        self,
        image_file: Path,
        output_dir: Path,
        evaluate: bool,
    ) -> Dict[str, Any]:
        """Process a single image for batch; returns outcome dict (no side effects on usage_stats)."""
        try:
            gt_path = self.find_ground_truth_file(str(image_file))
            if not gt_path:
                return {
                    "success": False,
                    "image_name": image_file.name,
                    "reason": "No matching ground truth JSON file",
                }

            output_filename = f"{image_file.stem}_{self.processing_config.method}.json"
            output_path = output_dir / output_filename

            if self.processing_config.method == "relation_classification":
                result = self._process_relation_classification_batch(
                    str(image_file),
                    gt_path,
                    str(output_path),
                    entities_input_dir=self.processing_config.entities_input,
                )
            elif self.processing_config.method == "path_generation":
                result = self.process_single_image(
                    str(image_file),
                    str(output_path),
                    ground_truth_path=gt_path,
                    fuzzy_matching=self.processing_config.fuzzy_matching,
                )
            else:
                result = self.process_single_image(str(image_file), str(output_path))

            if not result.success:
                return {
                    "success": False,
                    "image_name": image_file.name,
                    "error": result.error_message or "Unknown error",
                }

            result_data = {
                "image": image_file.name,
                "output": str(output_path),
                "ground_truth": gt_path,
                "method": self.processing_config.method,
            }
            if self.processing_config.method == "extract_entities":
                result_data["entities_extracted"] = result.entities_extracted
            elif self.processing_config.method in ["enumerate_paths", "extract_relation"]:
                result_data.update(result.metadata)

            metrics = None
            per_relation_metrics = None
            if evaluate and gt_path:
                if self.processing_config.method == "relation_classification":
                    metrics = result.metadata.get("evaluation_metrics")
                    per_relation_metrics = result.metadata.get("per_relation_metrics")
                else:
                    metrics, per_relation_metrics = self._run_evaluation(str(output_path), gt_path)

                if metrics:
                    result_data["evaluated"] = True
                    result_data["metrics"] = {
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                        "f1": metrics.f1,
                        "confusion_matrix": metrics.confusion_matrix,
                    }
                    if per_relation_metrics:
                        result_data["per_relation_metrics"] = per_relation_metrics
                    # Keep full evaluation for performance_results.csv (path_generation)
                    if self.processing_config.method == "path_generation":
                        result_data["evaluation"] = result.metadata.get("evaluation") or result.metadata.get("result", {}).get("evaluation")
                else:
                    result_data["evaluated"] = False
            else:
                result_data["evaluated"] = False

            return {
                "success": True,
                "image_name": image_file.name,
                "result": result,
                "result_data": result_data,
                "metrics": metrics,
                "per_relation_metrics": per_relation_metrics,
            }
        except Exception as e:
            return {"success": False, "image_name": image_file.name, "error": str(e)}

    def process_batch(
        self,
        input_dir: str,
        output_subdir: Optional[str] = None,
        evaluate: bool = True
    ) -> BatchProcessingResult:
        """
        Process all images in a directory (optionally in parallel).

        Args:
            input_dir: Input directory containing images
            output_subdir: Optional output subdirectory
            evaluate: Whether to run evaluation

        Returns:
            BatchProcessingResult with processing summary
        """
        # Start timing
        self.start_timing()
        input_path = Path(input_dir)
        output_dir = create_output_directory(
            self.processing_config.output_path,
            output_subdir
        )

        # Discover images
        image_files = discover_image_files(input_path)
        if not image_files:
            raise ValueError(f"No image files found in {input_dir}")

        results = {
            "level": input_path.name,
            "method": self.processing_config.method,
            "timestamp": datetime.now().isoformat(),
            "processed": [],
            "failed": [],
            "evaluated": [],
            "summary": {
                "total_images": len(image_files),
                "successful": 0,
                "failed": 0,
                "evaluated": 0
            }
        }

        evaluation_metrics = []
        per_relation_metrics_list = []
        max_workers = self.processing_config.max_workers

        if max_workers <= 1:
            # Sequential
            for i, image_file in enumerate(image_files, 1):
                print(f"[{i}/{len(image_files)}] Processing {image_file.name}...")
                outcome = self._process_one_image_in_batch(image_file, output_dir, evaluate)
                if outcome["success"]:
                    if outcome["result"].metadata.get("usage"):
                        self.add_usage(outcome["result"].metadata["usage"])
                    results["processed"].append(outcome["result_data"])
                    results["summary"]["successful"] += 1
                    if outcome.get("metrics"):
                        evaluation_metrics.append(outcome["metrics"])
                        if outcome.get("per_relation_metrics"):
                            per_relation_metrics_list.append(outcome["per_relation_metrics"])
                        results["evaluated"].append(outcome["result_data"])
                else:
                    fail_entry = {"image": outcome["image_name"]}
                    if outcome.get("reason"):
                        fail_entry["reason"] = outcome["reason"]
                    else:
                        fail_entry["error"] = outcome.get("error", "Unknown")
                    results["failed"].append(fail_entry)
                    results["summary"]["failed"] += 1
        else:
            # Parallel
            outcomes: List[Optional[Dict[str, Any]]] = [None] * len(image_files)
            completed = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(self._process_one_image_in_batch, im, output_dir, evaluate): i
                    for i, im in enumerate(image_files)
                }
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    completed += 1
                    try:
                        outcome = future.result()
                        outcomes[idx] = outcome
                        status = "✓" if outcome.get("success") else "✗"
                        print(f"[{completed}/{len(image_files)}] Completed {outcome['image_name']} {status}")
                    except Exception as e:
                        outcomes[idx] = {"success": False, "image_name": image_files[idx].name, "error": str(e)}
                        print(f"[{completed}/{len(image_files)}] Completed {image_files[idx].name} ✗ (exception)")

            for outcome in outcomes:
                if outcome is None:
                    continue
                if outcome["success"]:
                    if outcome["result"].metadata.get("usage"):
                        self.add_usage(outcome["result"].metadata["usage"])
                    results["processed"].append(outcome["result_data"])
                    results["summary"]["successful"] += 1
                    if outcome.get("metrics"):
                        evaluation_metrics.append(outcome["metrics"])
                        if outcome.get("per_relation_metrics"):
                            per_relation_metrics_list.append(outcome["per_relation_metrics"])
                        results["evaluated"].append(outcome["result_data"])
                else:
                    fail_entry = {"image": outcome["image_name"]}
                    if outcome.get("reason"):
                        fail_entry["reason"] = outcome["reason"]
                    else:
                        fail_entry["error"] = outcome.get("error", "Unknown")
                    results["failed"].append(fail_entry)
                    results["summary"]["failed"] += 1

        # Calculate averages with MICRO and MACRO F1
        if evaluation_metrics:
            results["summary"]["micro_macro_averages"] = aggregate_evaluation_metrics_with_micro_macro(evaluation_metrics, per_relation_metrics_list)
            results["summary"]["averages"] = aggregate_evaluation_metrics(evaluation_metrics)
            results["summary"]["per_relation_averages"] = aggregate_evaluation_metrics_with_relations(evaluation_metrics, per_relation_metrics_list)
            
            # Explicitly print the results using the new function
            print_aggregated_results_with_micro_macro(evaluation_metrics, results["evaluated"], "FINAL AVERAGED", per_relation_metrics_list)
            
            # Also save to a separate text file for easy viewing
            output_txt_path = output_dir / f"{input_path.name}_{self.processing_config.method}_evaluation_report.txt"
            with open(output_txt_path, 'w') as f:
                # Capture stdout to file
                import sys
                from io import StringIO
                old_stdout = sys.stdout
                sys.stdout = mystdout = StringIO()
                
                print_aggregated_results_with_micro_macro(evaluation_metrics, results["evaluated"], "FINAL AVERAGED", per_relation_metrics_list)
                
                sys.stdout = old_stdout
                f.write(mystdout.getvalue())
            print(f"  📄 Saved full evaluation report to: {output_txt_path}")

            # Append evaluation results to CSV (dataset, model, type, tp, tn, fp, fn, recall, precision, macro_f1, micro_f1); type = entities | assign | permit | prohibit
            csv_path = output_dir / "evaluation_results.csv"
            micro_macro = results["summary"]["micro_macro_averages"]
            entities_val = None
            if self.processing_config.method in ["extract_entities", "enumerate_paths"]:
                micro_o = micro_macro.get("micro_overall") or {}
                f1 = micro_o.get("f1")
                if isinstance(f1, (int, float)):
                    entities_val = f1
            write_evaluation_results_csv(
                csv_path,
                dataset_name=input_path.name.replace("_wo_legend", ""),
                model_name=self.api_config.model,
                micro_macro_averages=micro_macro,
                entities_value=entities_val,
            )
            print(f"  📊 Appended results to: {csv_path}")

        # Save summary
        summary_filename = f"{input_path.name}_{self.processing_config.method}_summary.json"
        summary_path = output_dir / summary_filename
        save_json(results, str(summary_path))

        # Append one row to performance_results.csv (path_generation / relation_extraction)
        if self.processing_config.method in ("path_generation", "relation_extraction") and results["evaluated"]:
            performance_csv = PROJECT_ROOT / "performance_results.csv"
            legend_label = "with legend" if self.processing_config.with_legend else "without legend"
            # For CSV, use base dataset name (e.g. subgraphs_01 not subgraphs_01_wo_legend)
            dataset_name = input_path.name.replace("_wo_legend", "")
            append_performance_results_csv(
                performance_csv,
                model_name=self.api_config.model,
                dataset_name=dataset_name,
                legend_label=legend_label,
                evaluated_results=results["evaluated"],
            )
            print(f"  📊 Appended to: {performance_csv}")

        # End timing and print cost/time summary
        self.end_timing()
        self.print_cost_and_time_summary()

        return BatchProcessingResult(**results)

    def process_subgraphs_dataset(
        self,
        dataset_root: str,
        subsets: Optional[List[str]] = None,
        output_subdir: Optional[str] = None
    ) -> Dict[str, BatchProcessingResult]:
        """
        Process SubgraphsWithTriples dataset structure.

        Args:
            dataset_root: Root directory of the dataset
            subsets: Optional list of subsets to process
            output_subdir: Optional output subdirectory

        Returns:
            Dictionary mapping subset names to results
        """
        # Start timing for dataset processing
        self.start_timing()

        root_path = Path(dataset_root)

        # Resolve dataset root - handle various input path formats
        base_root = None

        # Check if we're directly in one of the main folders
        if root_path.name in ["SubgraphsWithTriplesImages", "SubgraphsWithTriplesJSON"]:
            base_root = root_path.parent
        # Check if we're in the root directory containing both folders
        elif (root_path / "SubgraphsWithTriplesImages").exists() and \
             (root_path / "SubgraphsWithTriplesJSON").exists():
            base_root = root_path
        # Check if we're in a subdirectory - walk up to find the root
        else:
            current = root_path
            for _ in range(5):  # Don't go up more than 5 levels
                if current.parent.name in ["SubgraphsWithTriplesImages", "SubgraphsWithTriplesJSON"]:
                    # We're in a subdirectory of one of the main folders
                    base_root = current.parent.parent
                    break
                current = current.parent

            if base_root is None:
                raise ValueError(f"Invalid dataset root: {dataset_root}. "
                               f"Expected path within dataset structure.")

        images_root = base_root / "SubgraphsWithTriplesImages"
        json_root = base_root / "SubgraphsWithTriplesJSON"

        if not images_root.exists() or not json_root.exists():
            raise ValueError(f"Dataset folders not found under {base_root}")

        # Determine subsets to process (only known subsets with matching ground truth)
        discovered = [d.name for d in images_root.iterdir() if d.is_dir() and not d.name.endswith("_wo_legend")]
        available_subsets = [s for s in DATASET_SUBSETS if s in discovered]
        target_subsets = subsets if subsets else available_subsets

        results = {}

        for subset in target_subsets:
            # Handle legend option for image directory
            actual_subset_name = subset if self.processing_config.with_legend else f"{subset}_wo_legend"
            images_dir = images_root / actual_subset_name
            json_dir = json_root / subset

            if not images_dir.exists() or not json_dir.exists():
                print(f"Skipping {subset}: missing images ({actual_subset_name}) or JSON directory")
                continue

            # Base path is in processing_config.output_path; use subset as subdir only
            subset_output = subset

            print(f"\n{'-'*60}")
            print(f"Processing subset: {subset} ({'with' if self.processing_config.with_legend else 'without'} legend)")
            print(f"Images: {images_dir}")
            print(f"Ground truth: {json_dir}")
            print(f"{'-'*60}")

            # Temporarily override JSON directory and entities input for this subset
            original_json_override = getattr(self, '_json_dir_override', None)
            self._json_dir_override = str(json_dir)

            original_entities_input = None
            if self.processing_config.entities_input:
                original_entities_input = self.processing_config.entities_input
                # Adjust entities_input to point to the subset directory
                entities_subset_dir = json_root / subset
                if entities_subset_dir.exists():
                    self.processing_config.entities_input = str(entities_subset_dir)

            try:
                results[subset] = self.process_batch(
                    str(images_dir),
                    output_subdir=subset_output,
                    evaluate=True
                )
            finally:
                # Restore original settings
                if original_json_override:
                    self._json_dir_override = original_json_override
                else:
                    delattr(self, '_json_dir_override')

                if original_entities_input is not None:
                    self.processing_config.entities_input = original_entities_input

        # Save overview summary
        overview = {
            "dataset_root": str(base_root),
            "method": self.processing_config.method,
            "processed_subsets": list(results.keys()),
            "timestamp": datetime.now().isoformat(),
            "with_legend": self.processing_config.with_legend
        }

        overview_name = f"datasets_{self.processing_config.method}_summary.json"
        overview_dir = create_output_directory(self.processing_config.output_path, None)
        overview_path = overview_dir / overview_name
        save_json(overview, str(overview_path))

        print(f"\nSaved datasets overview to: {overview_path}")

        # End timing and print cost/time summary
        self.end_timing()
        self.print_cost_and_time_summary()

        return results

    def _run_evaluation(self, output_path: str, ground_truth_path: str) -> Tuple[Optional[EvaluationMetrics], Optional[Dict[str, Dict]]]:
        """Run appropriate evaluation based on method."""
        try:
            if self.processing_config.method in ["extract_entities", "enumerate_paths"]:
                metrics = evaluate_entity_extraction(output_path, ground_truth_path, quiet=True)
                return metrics, None
            elif self.processing_config.method == "relation_classification":
                # Relation classification evaluation is already done during processing
                # Just return None to indicate no additional evaluation needed
                return None, None
            elif self.processing_config.method == "path_generation":
                # For path_generation, we need to load the result and extract metrics
                from .file_utils import load_json
                result_data = load_json(output_path)
                eval_data = result_data.get("evaluation")
                if eval_data:
                    # Create a synthetic EvaluationMetrics for combined metrics
                    combined = eval_data.get("combined_metrics", {})
                    metrics = EvaluationMetrics(
                        precision=combined.get("precision", 0.0),
                        recall=combined.get("recall", 0.0),
                        f1=combined.get("f1", 0.0),
                        accuracy=combined.get("accuracy", 0.0)
                    )
                    
                    # Prepare sub-metrics for aggregation
                    # We include them in rel_metrics with special keys to get macro averages
                    rel_metrics = eval_data.get("relation_metrics", {}).get("per_relation", {}).copy()
                    
                    # Add overall entity metrics
                    entity_m = eval_data.get("entity_metrics", {})
                    rel_metrics["entity_overall"] = {
                        "precision": entity_m.get("precision", 0.0),
                        "recall": entity_m.get("recall", 0.0),
                        "f1": entity_m.get("f1", 0.0),
                        "accuracy": entity_m.get("accuracy", 0.0),
                        "count": 1,
                        "confusion_matrix": {
                            "tp": entity_m.get("tp", 0),
                            "fp": entity_m.get("fp", 0),
                            "fn": entity_m.get("fn", 0),
                            "tn": 0
                        }
                    }
                    
                    # Add overall relation metrics
                    relation_m = eval_data.get("relation_metrics", {})
                    rel_metrics["relation_overall"] = {
                        "precision": relation_m.get("precision", 0.0),
                        "recall": relation_m.get("recall", 0.0),
                        "f1": relation_m.get("f1", 0.0),
                        "accuracy": relation_m.get("accuracy", 0.0),
                        "count": 1,
                        "confusion_matrix": {
                            "tp": relation_m.get("tp", 0),
                            "fp": relation_m.get("fp", 0),
                            "fn": relation_m.get("fn", 0),
                            "tn": 0
                        }
                    }
                    
                    return metrics, rel_metrics
                return None, None
            else:
                return None, None
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return None, None

    def _process_relation_classification_batch(
        self,
        image_path: str,
        ground_truth_path: str,
        output_path: str,
        entities_input_dir: Optional[str] = None
    ) -> ProcessingResult:
        """Process relation classification in batch mode for all triples.
        
        Args:
            image_path: Path to the image file
            ground_truth_path: Path to ground truth JSON (for expected results)
            output_path: Path to save classification results
            entities_input_dir: Optional directory containing entity files
            
        Returns:
            ProcessingResult with classification results
        """
        try:
            # Determine entity source
            if entities_input_dir:
                # Use explicit entities_input directory
                is_ground_truth_format = (self.processing_config.relation_source == "ground_truth")

                # Handle SubgraphsWithTriples dataset structure
                entities_dir_to_use = entities_input_dir
                if self.processing_config.is_subgraphs_dataset and is_ground_truth_format:
                    # For SubgraphsWithTriples ground truth, entities_input might be parent directory
                    # Try to find the correct subset subdirectory
                    image_path_obj = Path(image_path)
                    subset_name = None

                    # Find subset name from image path
                    if "subgraphs_" in str(image_path_obj):
                        parts = image_path_obj.parts
                        for part in parts:
                            if part.startswith("subgraphs_"):
                                subset_name = part
                                break

                    if subset_name:
                        subset_entities_dir = Path(entities_input_dir) / subset_name
                        if subset_entities_dir.exists():
                            entities_dir_to_use = str(subset_entities_dir)
                            print(f"  📂 Using subset entities directory: {entities_dir_to_use}")

                entities_json_path = self._find_entity_file_for_image(
                    image_path,
                    entities_dir_to_use,
                    is_ground_truth=is_ground_truth_format
                )

                if not entities_json_path:
                    return ProcessingResult(
                        success=False,
                        error_message=f"No entity file found in {entities_dir_to_use} for {Path(image_path).name}"
                    )

                # Load entities based on format
                if is_ground_truth_format:
                    entities_data = self._convert_ground_truth_to_entity_format(entities_json_path)
                else:
                    entities_data = load_json(entities_json_path)

            else:
                # No explicit entities_input - use relation_source to determine source
                if self.processing_config.relation_source == "ground_truth":
                    # Use ground truth entities directly from the JSON file
                    entities_data = self._convert_ground_truth_to_entity_format(ground_truth_path)

                elif self.processing_config.relation_source == "predicted":
                    # Find predicted entity extraction file using automatic search
                    entities_json_path = self._find_predicted_entities_file(image_path)
                    if not entities_json_path:
                        return ProcessingResult(
                            success=False,
                            error_message=f"No predicted entities file found for {image_path}"
                        )
                    entities_data = load_json(entities_json_path)
                else:
                    return ProcessingResult(
                        success=False,
                        error_message=f"Invalid relation_source: {self.processing_config.relation_source}"
                    )

            # Generate triples using extract_relation strategy
            from .processing_strategies import RelationExtractionStrategy, ProcessingStrategyFactory
            extract_strategy = ProcessingStrategyFactory.create_strategy(
                "extract_relation",
                self.api_config,
                self.processing_config
            )

            # Generate triples without saving
            triples_result = extract_strategy.process(
                entities_json_path=None,  # Not needed for generation
                ground_truth_json_path=ground_truth_path,  # For expected results
                output_path=None,  # Don't save triples
                entities_data=entities_data  # Pass entities directly
            )

            if not triples_result.success:
                return ProcessingResult(
                    success=False,
                    error_message=f"Failed to generate triples: {triples_result.error_message}"
                )

            triples = triples_result.metadata.get("triples", [])
            if not triples:
                return ProcessingResult(
                    success=False,
                    error_message="No triples generated from entities"
                )

            # Log triple generation statistics
            total_triples = len(triples)
            stats = triples_result.metadata
            assign_count = stats.get("assign_triples", 0)
            permit_count = stats.get("permit_triples", 0)
            prohibit_count = stats.get("prohibit_triples", 0)

            print(f"  📊 Generated {total_triples} entity pairs for relation testing:")
            print(f"     • Assign relations: {assign_count} pairs")
            print(f"     • Permit relations: {permit_count} pairs")
            print(f"     • Prohibit relations: {prohibit_count} pairs")
            print(f"  🔄 Starting relation classification for {total_triples} pairs...")

            # Run batch relation classification
            classification_result = self.strategy.process_batch(
                image_path,
                triples,
                output_path,
                entities_data=entities_data
            )

            # Add evaluation metadata
            if classification_result.success:
                successful_classifications = classification_result.metadata.get("successful_classifications", 0)
                total_triples = classification_result.metadata.get("total_triples", 0)

                print(f"  ✅ Classified {successful_classifications}/{total_triples} relations")

                # Try automatic evaluation if we have ground truth
                if ground_truth_path:
                    from .evaluation import evaluate_relation_classification_batch
                    overall_metrics, per_relation_metrics = evaluate_relation_classification_batch(output_path, triples)
                    if overall_metrics:
                        # Display confusion matrix for this figure
                        cm = overall_metrics.confusion_matrix
                        print(f"  🔢 Confusion Matrix - TP: {cm['tp']}, FP: {cm['fp']}, FN: {cm['fn']}, TN: {cm['tn']}")
                        # Handle f1 that can be string ("unavailable") or float
                        f1_display = overall_metrics.f1 if isinstance(overall_metrics.f1, str) else f"{overall_metrics.f1:.4f}"
                        print(f"  📊 MICRO-F1 - Precision: {overall_metrics.precision:.4f}, Recall: {overall_metrics.recall:.4f}, F1: {f1_display}")

                        # Display per-relation metrics
                        for rel_type, rel_metrics in per_relation_metrics.items():
                            count = rel_metrics['count']
                            rel_cm = rel_metrics.get('confusion_matrix', {})
                            rel_tp = rel_cm.get('tp', 0)
                            rel_fp = rel_cm.get('fp', 0)
                            rel_fn = rel_cm.get('fn', 0)
                            rel_tn = rel_cm.get('tn', 0)
                            # Handle f1 that can be string ("unavailable") or float
                            rel_f1 = rel_metrics['f1']
                            rel_f1_display = rel_f1 if isinstance(rel_f1, str) else f"{rel_f1:.4f}"
                            print(f"     {rel_type.title()}: P={rel_metrics['precision']:.4f}, R={rel_metrics['recall']:.4f}, F1={rel_f1_display} (TP:{rel_tp}, FP:{rel_fp}, FN:{rel_fn}, TN:{rel_tn}, n={count})")

                        classification_result.metadata["evaluation_metrics"] = overall_metrics
                        classification_result.metadata["per_relation_metrics"] = per_relation_metrics

            # Mark as successful and return new result
            print(f"  ✅ Processing completed successfully, returning success=True")
            return ProcessingResult(
                success=True,
                output_path=output_path,
                metadata=classification_result.metadata
            )

        except Exception as e:
            import traceback
            print(f"  ❌ Error in relation classification: {str(e)}")
            traceback.print_exc()
            return ProcessingResult(
                success=False,
                error_message=str(e)
            )

    def _find_predicted_entities_file_for_output(self, output_path: str) -> Optional[str]:
        """Find the corresponding predicted entities file for an output path."""
        from pathlib import Path
        output_path_obj = Path(output_path)
        output_dir = output_path_obj.parent
        output_name = output_path_obj.stem

        # Look for predicted entities file in the same directory
        # Pattern: [image_name]_extract_entities.json
        entities_file = output_dir / f"{output_name}_extract_entities.json"
        if entities_file.exists():
            return str(entities_file)

        # Try to find any _extract_entities.json file in the directory
        for json_file in output_dir.glob("*_extract_entities.json"):
            return str(json_file)

        return None

    def _find_entity_file_for_image(
        self,
        image_path: str,
        entities_dir: str,
        is_ground_truth: bool = True
    ) -> Optional[str]:
        """Find entity file that matches an image file.
        
        Args:
            image_path: Path to the image file
            entities_dir: Directory containing entity files
            is_ground_truth: True for ground truth JSON, False for predicted entities
            
        Returns:
            Path to the matching entity file, or None if not found
            
        File naming patterns:
        - Image: enterprise_clients_graph_policies_graph_part1__association_..._labeled.png
        - Ground truth: enterprise_clients_graph_policies_graph_part1__association_....json
        - Predicted: enterprise_clients_graph_policies_graph_part1__association_..._labeled_extract_entities.json
        """
        from pathlib import Path
        
        image_path_obj = Path(image_path)
        image_stem = image_path_obj.stem  # Without extension
        entities_dir_obj = Path(entities_dir)
        
        if not entities_dir_obj.exists():
            print(f"  ⚠ Entities directory does not exist: {entities_dir}")
            return None
        
        if is_ground_truth:
            # Ground truth: remove "_labeled" or "_labeled_b" suffix and add .json
            if image_stem.endswith("_labeled_b"):
                base_name = image_stem[:-10]  # Remove "_labeled_b"
            elif image_stem.endswith("_labeled"):
                base_name = image_stem[:-8]  # Remove "_labeled"
            else:
                base_name = image_stem

            entity_file = entities_dir_obj / f"{base_name}.json"
        else:
            # Predicted: keep "_labeled" and add "_extract_entities.json"
            entity_file = entities_dir_obj / f"{image_stem}_extract_entities.json"
        
        if entity_file.exists():
            return str(entity_file)
        
        # Try without "_labeled" for predicted as well (fallback)
        if not is_ground_truth and image_stem.endswith("_labeled"):
            base_name = image_stem[:-8]
            entity_file_alt = entities_dir_obj / f"{base_name}_extract_entities.json"
            if entity_file_alt.exists():
                return str(entity_file_alt)
        
        print(f"  ⚠ Entity file not found for {image_path_obj.name}")
        print(f"    Expected: {entity_file.name}")
        return None

    def _find_predicted_entities_file(self, image_path: str) -> Optional[str]:
        """Find predicted entity extraction file for an image.
        
        Searches for entity extraction results in various possible locations:
        1. Same directory structure as input (outputs/SubgraphsWithTriples/{subset}/)
        2. experiments directory
        3. outputs directory with various subset patterns
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Path to the predicted entities JSON file, or None if not found
        """
        from pathlib import Path
        image_path_obj = Path(image_path)
        image_stem = image_path_obj.stem
        
        # Extract subset name from image path (e.g., "subgraphs_01" from path)
        parent_dir = image_path_obj.parent.name
        grandparent_dir = image_path_obj.parent.parent.name if image_path_obj.parent.parent else ""
        
        # Determine subset name
        if parent_dir.startswith("subgraphs_"):
            subset_name = parent_dir
        elif grandparent_dir.startswith("subgraphs_"):
            subset_name = grandparent_dir
        else:
            subset_name = parent_dir
        
        # Build comprehensive list of possible paths
        outputs_base = PROJECT_ROOT / "outputs"
        experiments_base = PROJECT_ROOT / "experiments"
        
        possible_paths = [
            # outputs/datasets/{subset}/
            outputs_base / "datasets" / subset_name / f"{image_stem}_extract_entities.json",
            # outputs/{subset}/
            outputs_base / subset_name / f"{image_stem}_extract_entities.json",
            # experiments/datasets/{subset}/
            experiments_base / "datasets" / subset_name / f"{image_stem}_extract_entities.json",
            # experiments/{subset}/
            experiments_base / subset_name / f"{image_stem}_extract_entities.json",
            # outputs/Level_X_Graphs/
            outputs_base / parent_dir / f"{image_stem}_extract_entities.json",
        ]
        
        # Also try common subset names if we couldn't determine it
        for common_subset in DATASET_SUBSETS:
            possible_paths.extend([
                outputs_base / "SubgraphsWithTriples" / common_subset / f"{image_stem}_extract_entities.json",
                outputs_base / common_subset / f"{image_stem}_extract_entities.json",
            ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for path in possible_paths:
            if str(path) not in seen:
                seen.add(str(path))
                unique_paths.append(path)
        
        for path in unique_paths:
            if path.exists():
                print(f"  📂 Found predicted entities: {path}")
                return str(path)
        
        # Debug: print searched paths
        print(f"  ⚠ Could not find predicted entities file for {image_stem}")
        print(f"    Searched in: {[str(p.parent) for p in unique_paths[:3]]}")
        
        return None

    def _convert_ground_truth_to_entity_format(self, ground_truth_path: str) -> Dict:
        """Convert ground truth JSON to entity extraction format."""
        gt_data = load_json(ground_truth_path)

        nodes = []
        node_id_counter = 1

        if "policy_elements" in gt_data:
            policy_elements = gt_data["policy_elements"]

            # Add user_attributes
            if "user_attributes" in policy_elements:
                user_attrs = policy_elements["user_attributes"]
                if isinstance(user_attrs, list):
                    for attr in user_attrs:
                        nodes.append({
                            "node_id": f"n{node_id_counter}",
                            "type": "user_attributes",
                            "label": attr  # Use "label" to match predicted entities format
                        })
                        node_id_counter += 1

            # Add object_attributes
            if "object_attributes" in policy_elements:
                obj_attrs = policy_elements["object_attributes"]
                if isinstance(obj_attrs, list):
                    for attr in obj_attrs:
                        nodes.append({
                            "node_id": f"n{node_id_counter}",
                            "type": "object_attributes",
                            "label": attr  # Use "label" to match predicted entities format
                        })
                        node_id_counter += 1

            # Add policy_classes
            if "policy_classes" in policy_elements:
                policy_class = policy_elements["policy_classes"]
                if isinstance(policy_class, str):
                    nodes.append({
                        "node_id": f"n{node_id_counter}",
                        "type": "policy_classes",
                        "label": policy_class  # Use "label" to match predicted entities format
                    })

        return {
            "source_image": Path(ground_truth_path).name.replace('.json', '.png'),
            "method": "ground_truth_entities",
            "total_entities": len(nodes),
            "nodes": nodes
        }

    def find_ground_truth_file(self, image_path: str) -> Optional[str]:
        """Find ground truth file for an image."""
        # Use explicit gt_input if provided in config
        if self.processing_config.gt_input:
            return find_ground_truth_file(image_path, self.processing_config.gt_input)
            
        json_dir = getattr(self, '_json_dir_override', None)
        return find_ground_truth_file(image_path, json_dir)

"""Evaluation metrics for entity extraction, relation classification, and path generation.

Micro-F1: pools TP/FP/FN across all classes (biased toward frequent classes).
Macro-F1: averages per-class F1 (gives equal weight to rare classes).
Per-graph evaluation uses micro-F1; cross-graph aggregation uses macro-F1.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union

from .config import EvaluationMetrics
from .file_utils import normalize_entity_name
from .file_utils import load_json, save_json, create_output_directory


# ============================================================================
# EVALUATION RESULTS CSV EXPORT
# ============================================================================

EVALUATION_CSV_HEADERS = [
    "dataset", "model", "type",
    "tp", "tn", "fp", "fn",
    "recall", "precision", "macro_f1", "micro_f1",
]

# Header and separator for performance_results.csv (pipe-separated, same as run_evaluation.sh)
PERFORMANCE_RESULTS_CSV_HEADER = (
    "model name | dataset name | legend or not | entity_TP | entity_FP | entity_FN | "
    "entity_P_micro | entity_R_micro | entity_F1_micro | entity_P_macro | entity_R_macro | entity_F1_macro | "
    "rel_TP | rel_TN | rel_FP | rel_FN | rel_P_micro | rel_R_micro | rel_F1_micro | rel_P_macro | rel_R_macro | rel_F1_macro | overall_accuracy"
)
PERFORMANCE_RESULTS_SEP = " | "


def _safe_f1(value: Any) -> str:
    """Format F1 for CSV (may be float or 'unavailable')."""
    if isinstance(value, (int, float)):
        return f"{value:.6f}"
    return str(value)


def write_evaluation_results_csv(
    csv_path: Union[str, Path],
    dataset_name: str,
    model_name: str,
    micro_macro_averages: Dict[str, Any],
    entities_value: Union[str, float, None] = None,
) -> None:
    """
    Append evaluation result rows to a CSV file.

    Columns: dataset, model, type, tp, tn, fp, fn, recall, precision, macro_f1, micro_f1.
    type values: entities, assign, permit, prohibit.

    Writes:
    - One row with type="entities" (entity metrics) when entity_overall is in per_relation.
    - One row per relation type: assign, permit, prohibit.

    Args:
        csv_path: Path to CSV file (created with headers if missing).
        dataset_name: Name of the dataset (e.g. input directory name).
        model_name: Model used (e.g. gpt-5-nano).
        micro_macro_averages: Result from aggregate_evaluation_metrics_with_micro_macro().
        entities_value: Optional entity F1 or label (e.g. "n/a" when not applicable).
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    per_rel = micro_macro_averages.get("per_relation") or {}
    entity_overall = per_rel.get("entity_overall")

    file_exists = csv_path.exists()
    rows: List[List[str]] = []

    # Row with type="entities": from entity_overall when present, else from micro_overall + entities_value
    if entity_overall is not None:
        cm_e = entity_overall.get("total_confusion_matrix") or entity_overall.get("confusion_matrix") or {}
        tp_e = cm_e.get("tp", 0)
        tn_e = cm_e.get("tn", 0)
        fp_e = cm_e.get("fp", 0)
        fn_e = cm_e.get("fn", 0)
        rec_e = entity_overall.get("micro_recall", entity_overall.get("recall", 0.0))
        prec_e = entity_overall.get("micro_precision", entity_overall.get("precision", 0.0))
        micro_f1_e = entity_overall.get("micro_f1", entity_overall.get("f1", 0.0))
        macro_f1_e = entity_overall.get("macro_f1", entity_overall.get("f1", 0.0))
        rows.append([
            dataset_name,
            model_name,
            "entities",
            str(tp_e), str(tn_e), str(fp_e), str(fn_e),
            f"{rec_e:.6f}" if isinstance(rec_e, (int, float)) else str(rec_e),
            f"{prec_e:.6f}" if isinstance(prec_e, (int, float)) else str(prec_e),
            _safe_f1(macro_f1_e),
            _safe_f1(micro_f1_e),
        ])
    elif entities_value is not None:
        # Fallback for extract_entities / enumerate_paths: one row with type="entities" from micro_overall
        micro_o = micro_macro_averages.get("micro_overall") or {}
        macro_o = micro_macro_averages.get("macro_overall") or {}
        cm = micro_o.get("confusion_matrix") or {}
        rows.append([
            dataset_name,
            model_name,
            "entities",
            str(cm.get("tp", 0)), str(cm.get("tn", 0)), str(cm.get("fp", 0)), str(cm.get("fn", 0)),
            f"{micro_o.get('recall', 0.0):.6f}",
            f"{micro_o.get('precision', 0.0):.6f}",
            _safe_f1(macro_o.get("f1")),
            _safe_f1(micro_o.get("f1")),
        ])

    # Rows with type = assign, permit, prohibit
    for rel_type in sorted(per_rel.keys()):
        if rel_type in ("entity_overall", "relation_overall"):
            continue
        data = per_rel[rel_type]
        cm_rel = data.get("total_confusion_matrix") or data.get("confusion_matrix") or {}
        tp_r = cm_rel.get("tp", 0)
        tn_r = cm_rel.get("tn", 0)
        fp_r = cm_rel.get("fp", 0)
        fn_r = cm_rel.get("fn", 0)
        rec_r = data.get("micro_recall", data.get("recall", 0.0))
        prec_r = data.get("micro_precision", data.get("precision", 0.0))
        micro_f1_r = data.get("micro_f1", data.get("f1", 0.0))
        macro_f1_r = data.get("macro_f1", data.get("f1", 0.0))
        rows.append([
            dataset_name,
            model_name,
            rel_type,
            str(tp_r), str(tn_r), str(fp_r), str(fn_r),
            f"{rec_r:.6f}" if isinstance(rec_r, (int, float)) else str(rec_r),
            f"{prec_r:.6f}" if isinstance(prec_r, (int, float)) else str(prec_r),
            _safe_f1(macro_f1_r),
            _safe_f1(micro_f1_r),
        ])

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(EVALUATION_CSV_HEADERS)
        writer.writerows(rows)


def append_performance_results_csv(
    csv_path: Union[str, Path],
    model_name: str,
    dataset_name: str,
    legend_label: str,
    evaluated_results: List[Dict[str, Any]],
) -> None:
    """
    Append one row to performance_results.csv (pipe-separated format used by run_evaluation.sh).

    Aggregates entity and relation metrics from path_generation evaluated results,
    then writes: model name | dataset name | legend or not | entity_TP | ... | overall_accuracy.

    Args:
        csv_path: Path to performance_results.csv (e.g. PROJECT_ROOT / "performance_results.csv").
        model_name: Model used (e.g. gpt-5-mini).
        dataset_name: Dataset/subset name (e.g. subgraphs_01).
        legend_label: "with legend" or "without legend".
        evaluated_results: List of result dicts each with "evaluation" containing
            entity_metrics, relation_metrics, combined_metrics.
    """
    if not evaluated_results:
        return
    if not any((item.get("evaluation") or {}).get("entity_metrics") for item in evaluated_results):
        return
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate entity and relation counts and F1s across figures
    entity_tp = entity_fp = entity_fn = 0
    rel_tp = rel_tn = rel_fp = rel_fn = 0
    entity_p_macro_list, entity_r_macro_list, entity_f1_macro_list = [], [], []
    rel_p_macro_list, rel_r_macro_list, rel_f1_macro_list = [], [], []
    overall_acc_list = []

    for item in evaluated_results:
        ev = item.get("evaluation") or {}
        ent = ev.get("entity_metrics") or {}
        rel = ev.get("relation_metrics") or {}
        comb = ev.get("combined_metrics") or {}
        entity_tp += ent.get("tp", 0)
        entity_fp += ent.get("fp", 0)
        entity_fn += ent.get("fn", 0)
        rel_tp += rel.get("tp", 0)
        rel_tn += rel.get("tn", 0)
        rel_fp += rel.get("fp", 0)
        rel_fn += rel.get("fn", 0)
        if ent.get("precision") is not None:
            entity_p_macro_list.append(ent["precision"])
        if ent.get("recall") is not None:
            entity_r_macro_list.append(ent["recall"])
        if isinstance(ent.get("f1"), (int, float)):
            entity_f1_macro_list.append(ent["f1"])
        if rel.get("precision") is not None:
            rel_p_macro_list.append(rel["precision"])
        if rel.get("recall") is not None:
            rel_r_macro_list.append(rel["recall"])
        if isinstance(rel.get("f1"), (int, float)):
            rel_f1_macro_list.append(rel["f1"])
        if isinstance(comb.get("accuracy"), (int, float)):
            overall_acc_list.append(comb["accuracy"])
        elif isinstance(ent.get("f1"), (int, float)) and isinstance(rel.get("f1"), (int, float)):
            overall_acc_list.append((float(ent["f1"]) + float(rel["f1"])) / 2.0)

    # Micro (global) entity metrics
    e_den_p = entity_tp + entity_fp
    e_den_r = entity_tp + entity_fn
    entity_P_micro = entity_tp / e_den_p if e_den_p else 0.0
    entity_R_micro = entity_tp / e_den_r if e_den_r else 0.0
    entity_F1_micro = (
        2 * entity_P_micro * entity_R_micro / (entity_P_micro + entity_R_micro)
        if (entity_P_micro + entity_R_micro) else 0.0
    )
    entity_P_macro = sum(entity_p_macro_list) / len(entity_p_macro_list) if entity_p_macro_list else 0.0
    entity_R_macro = sum(entity_r_macro_list) / len(entity_r_macro_list) if entity_r_macro_list else 0.0
    entity_F1_macro = sum(entity_f1_macro_list) / len(entity_f1_macro_list) if entity_f1_macro_list else 0.0

    # Micro (global) relation metrics
    r_den_p = rel_tp + rel_fp
    r_den_r = rel_tp + rel_fn
    rel_P_micro = rel_tp / r_den_p if r_den_p else 0.0
    rel_R_micro = rel_tp / r_den_r if r_den_r else 0.0
    rel_F1_micro = (
        2 * rel_P_micro * rel_R_micro / (rel_P_micro + rel_R_micro)
        if (rel_P_micro + rel_R_micro) else 0.0
    )
    rel_P_macro = sum(rel_p_macro_list) / len(rel_p_macro_list) if rel_p_macro_list else 0.0
    rel_R_macro = sum(rel_r_macro_list) / len(rel_r_macro_list) if rel_r_macro_list else 0.0
    rel_F1_macro = sum(rel_f1_macro_list) / len(rel_f1_macro_list) if rel_f1_macro_list else 0.0

    overall_accuracy = sum(overall_acc_list) / len(overall_acc_list) if overall_acc_list else 0.0

    file_exists = csv_path.exists()
    with open(csv_path, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write(PERFORMANCE_RESULTS_CSV_HEADER + "\n")
        parts = [
            model_name,
            dataset_name,
            legend_label,
            str(entity_tp), str(entity_fp), str(entity_fn),
            f"{entity_P_micro:.4f}", f"{entity_R_micro:.4f}", f"{entity_F1_micro:.4f}",
            f"{entity_P_macro:.4f}", f"{entity_R_macro:.4f}", f"{entity_F1_macro:.4f}",
            str(rel_tp), str(rel_tn), str(rel_fp), str(rel_fn),
            f"{rel_P_micro:.4f}", f"{rel_R_micro:.4f}", f"{rel_F1_micro:.4f}",
            f"{rel_P_macro:.4f}", f"{rel_R_macro:.4f}", f"{rel_F1_macro:.4f}",
            f"{overall_accuracy:.4f}",
        ]
        f.write(PERFORMANCE_RESULTS_SEP.join(parts) + "\n")


# ============================================================================
# ENTITY EVALUATION
# ============================================================================

def evaluate_entity_extraction(
    prediction_path: str,
    ground_truth_path: str,
    quiet: bool = False
) -> Optional[EvaluationMetrics]:
    """
    Evaluate entity extraction against ground truth.

    Args:
        prediction_path: Path to prediction JSON file
        ground_truth_path: Path to ground truth JSON file
        quiet: If True, suppress console output

    Returns:
        EvaluationMetrics object or None if evaluation fails
    """
    try:
        # Load predictions
        pred_data = load_json(prediction_path)

        # Extract predicted entities
        pred_entities_with_type = _extract_predicted_entities(pred_data)

        # Load ground truth
        gt_data = load_json(ground_truth_path)

        # Extract ground truth entities
        gt_entities_with_type = _extract_ground_truth_entities(gt_data)

        # Calculate metrics
        metrics = _calculate_entity_metrics(
            pred_entities_with_type,
            gt_entities_with_type,
            quiet=quiet
        )

        if not quiet:
            _print_entity_evaluation_report(
                prediction_path,
                ground_truth_path,
                pred_entities_with_type,
                gt_entities_with_type,
                metrics
            )

        return metrics

    except Exception as e:
        if not quiet:
            print(f"❌ Error in entity evaluation: {e}")
        return None


def _extract_predicted_entities(pred_data: Dict) -> Set[Tuple[str, str]]:
    """Extract predicted entities with their types."""
    entities_with_type = set()

    # Try different formats for predictions
    for node in pred_data.get("nodes", []):
        if "content" in node:
            entity_name = node["content"]
            normalized = normalize_entity_name(entity_name)
            entity_type = node.get("type") or node.get("task_type", "unknown")
            # Normalize entity type
            if entity_type == "object_attribues":
                entity_type = "object_attributes"
            entities_with_type.add((normalized, entity_type))

    # Fallback for entities format
    if not entities_with_type:
        for entity in pred_data.get("entities", []):
            if "label" in entity:
                entity_name = entity["label"]
                normalized = normalize_entity_name(entity_name)
                entity_type = entity.get("type", "unknown")
                # Normalize entity type
                if entity_type == "object_attribues":
                    entity_type = "object_attributes"
                entities_with_type.add((normalized, entity_type))

    return entities_with_type


def _extract_ground_truth_entities(gt_data: Dict) -> Set[Tuple[str, str]]:
    """Extract ground truth entities with their types."""
    entities_with_type = set()

    if "policy_elements" in gt_data:
        policy_elements = gt_data["policy_elements"]

        for entity in policy_elements.get("user_attributes", []):
            normalized = normalize_entity_name(entity)
            entities_with_type.add((normalized, "user_attributes"))

        for entity in policy_elements.get("object_attributes", []):
            normalized = normalize_entity_name(entity)
            entities_with_type.add((normalized, "object_attributes"))

        policy_classes = policy_elements.get("policy_classes", [])
        if isinstance(policy_classes, str):
            policy_classes = [policy_classes]
        for entity in policy_classes:
            normalized = normalize_entity_name(entity)
            entities_with_type.add((normalized, "policy_classes"))
    elif "assignments" in gt_data or "associations" in gt_data:
        # Extract entities from assignments/associations
        all_gt_entity_names = set()

        # Extract from assignments
        for assignment in gt_data.get("assignments", []):
            if isinstance(assignment, dict):
                if "from" in assignment:
                    all_gt_entity_names.add(assignment["from"])
                if "to" in assignment:
                    all_gt_entity_names.add(assignment["to"])

        # Extract from associations
        for association in gt_data.get("associations", []):
            if isinstance(association, dict):
                if "from" in association:
                    all_gt_entity_names.add(association["from"])
                if "to" in association:
                    all_gt_entity_names.add(association["to"])

        # Use relaxed matching for assignments format
        for entity_name in all_gt_entity_names:
            normalized = normalize_entity_name(entity_name)
            entities_with_type.add((normalized, "unknown"))
    else:
        # Fallback for nodes format
        for node in gt_data.get("nodes", []):
            if "content" in node:
                entity_name = node["content"]
                normalized = normalize_entity_name(entity_name)
                entity_type = node.get("type", "unknown")
                entities_with_type.add((normalized, entity_type))

    return entities_with_type


def _calculate_entity_metrics(
    pred_entities: Set[Tuple[str, str]],
    gt_entities: Set[Tuple[str, str]],
    quiet: bool = False
) -> EvaluationMetrics:
    """Calculate precision, recall, and F1 for entity extraction."""
    # Match entities by (name, type) tuple
    # If ground truth has "unknown" type, match by name only
    true_positives_set = set()
    false_positives_set = pred_entities.copy()
    false_negatives_set = gt_entities.copy()

    # Build mapping of normalized names to types for ground truth
    gt_name_to_types = {}
    for normalized_name, entity_type in gt_entities:
        if normalized_name not in gt_name_to_types:
            gt_name_to_types[normalized_name] = set()
        gt_name_to_types[normalized_name].add(entity_type)

    # Match predictions to ground truth
    for pred_name, pred_type in pred_entities:
        if pred_name in gt_name_to_types:
            gt_types = gt_name_to_types[pred_name]
            # If ground truth has "unknown" type, match by name only
            if "unknown" in gt_types or pred_type in gt_types:
                true_positives_set.add((pred_name, pred_type))
                false_positives_set.discard((pred_name, pred_type))
                # Remove matching ground truth entry
                for gt_type in gt_types:
                    false_negatives_set.discard((pred_name, gt_type))

    true_positives = len(true_positives_set)
    false_positives = len(false_positives_set)
    false_negatives = len(false_negatives_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return EvaluationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix={
            "tp": true_positives,
            "fp": false_positives,
            "fn": false_negatives
        }
    )


def _print_entity_evaluation_report(
    pred_path: str,
    gt_path: str,
    pred_entities: Set[Tuple[str, str]],
    gt_entities: Set[Tuple[str, str]],
    metrics: EvaluationMetrics
) -> None:
    """Print detailed entity evaluation report."""
    print("\n" + "="*50)
    print("🔍 ENTITY EXTRACTION EVALUATION RESULTS")
    print("="*50)
    print(f"\nGround Truth File: {gt_path}")
    print(f"Prediction File: {pred_path}\n")

    print("Entity Counts:")
    print(f"  Ground Truth:     {len(gt_entities)}")
    print(f"  Predicted:        {len(pred_entities)}")
    print(f"  Correct (TP):     {len(gt_entities & pred_entities)}")
    print(f"  False Positives:  {len(pred_entities - gt_entities)}")
    print(f"  False Negatives:  {len(gt_entities - pred_entities)}")
    print("")

    print("Evaluation Metrics:")
    print(f"  Precision:  {metrics.precision:.4f}")
    print(f"  Recall:     {metrics.recall:.4f}")
    print(f"  F1 Score:   {metrics.f1:.4f}")
    print("=" * 50)


# ============================================================================
# RELATION CLASSIFICATION EVALUATION
# ============================================================================

def evaluate_relation_classification(
    predictions_path: str,
    triples_path: Optional[str] = None
) -> Tuple[EvaluationMetrics, Dict[str, int]]:
    """
    Evaluate binary relation classification predictions.

    Args:
        predictions_path: Path to predictions JSON file
        triples_path: Path to ground truth triples JSON file

    Returns:
        Tuple of (EvaluationMetrics, confusion_matrix)
    """
    # Load predictions
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)

    # If triples_path is provided, load ground truth triples
    ground_truth_triples = None
    if triples_path and Path(triples_path).exists():
        with open(triples_path, 'r') as f:
            ground_truth_triples = json.load(f)

    # Calculate metrics
    confusion_matrix = {
        'tp': 0,  # True Positives
        'tn': 0,  # True Negatives
        'fp': 0,  # False Positives
        'fn': 0   # False Negatives
    }

    correct_predictions = 0
    total_predictions = len(predictions)

    for pred in predictions:
        pred_exists = pred.get('exists', '').lower() in ['yes', 'true']
        # Fix: field name is 'groundtruth' not 'ground_truth'
        gt_exists = pred.get('groundtruth', pred.get('ground_truth', '')).lower() in ['yes', 'true']

        if pred_exists and gt_exists:
            confusion_matrix['tp'] += 1
        elif not pred_exists and not gt_exists:
            confusion_matrix['tn'] += 1
        elif pred_exists and not gt_exists:
            confusion_matrix['fp'] += 1
        elif not pred_exists and gt_exists:
            confusion_matrix['fn'] += 1

    # Calculate metrics
    tp = confusion_matrix['tp']
    tn = confusion_matrix['tn']
    fp = confusion_matrix['fp']
    fn = confusion_matrix['fn']

    # Check for all TN case (no positive predictions or ground truth)
    if tp == 0 and fp == 0 and fn == 0:
        precision = 0.0
        recall = 0.0
        f1 = "unavailable"
        accuracy = tn / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    else:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    metrics = EvaluationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        confusion_matrix=confusion_matrix
    )

    return metrics, confusion_matrix


def evaluate_relation_classification_comprehensive(
    entities_data: Dict,
    ground_truth_path: str,
    predictions_path: Optional[str] = None,
    predictions_list: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of relation classification covering ALL possible relations.

    This Context7 approach evaluates the complete graph structure rather than just
    a processed subset, providing true performance metrics for the entire domain.

    Args:
        entities_data: Entity data (predicted format with 'nodes')
        ground_truth_path: Path to ground truth JSON
        predictions_path: Path to predictions JSON (optional if predictions_list provided)
        predictions_list: Direct predictions list (optional if predictions_path provided)

    Returns:
        Dict with comprehensive evaluation metrics including:
        - Overall metrics (precision, recall, F1)
        - Per-relation-type metrics
        - Coverage analysis
        - Graph completeness metrics
    """
    from .entity_pair_generator import generate_all_relation_triples
    from .file_utils import load_json

    # Load ground truth and generate ALL possible triples
    gt_data = load_json(ground_truth_path)
    all_possible_triples = generate_all_relation_triples(entities_data, gt_data, entities_source="auto")

    # Load predictions if not provided directly
    if predictions_list is None and predictions_path:
        with open(predictions_path, 'r') as f:
            predictions_list = json.load(f)
        if not isinstance(predictions_list, list):
            predictions_list = [predictions_list]

    # Create prediction lookup for fast access
    prediction_lookup = {}
    if predictions_list:
        for pred in predictions_list:
            key = (pred.get('entity1', ''), pred.get('entity2', ''), pred.get('relation', ''))
            prediction_lookup[key] = pred

    # Initialize comprehensive metrics
    overall_cm = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    relation_type_metrics = {
        'assign': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'total': 0, 'predicted': 0},
        'permit': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'total': 0, 'predicted': 0},
        'prohibit': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'total': 0, 'predicted': 0}
    }

    # Evaluate each possible triple
    for triple in all_possible_triples:
        from_entity = triple['from_entity']
        to_entity = triple['to_entity']
        relation_type = triple['relationship']
        expected_exists = triple['expected_result'].lower() in ['yes', 'true']

        # Check if this triple was predicted
        pred = prediction_lookup.get((from_entity, to_entity, relation_type))
        if pred:
            # Triple was processed - use actual prediction
            pred_exists = pred.get('exists', '').lower() in ['yes', 'true']
            relation_type_metrics[relation_type]['predicted'] += 1
        else:
            # Triple was not processed - assume "No" prediction (conservative approach)
            pred_exists = False

        # Update confusion matrix
        if pred_exists and expected_exists:
            overall_cm['tp'] += 1
            relation_type_metrics[relation_type]['tp'] += 1
        elif not pred_exists and not expected_exists:
            overall_cm['tn'] += 1
            relation_type_metrics[relation_type]['tn'] += 1
        elif pred_exists and not expected_exists:
            overall_cm['fp'] += 1
            relation_type_metrics[relation_type]['fp'] += 1
        elif not pred_exists and expected_exists:
            overall_cm['fn'] += 1
            relation_type_metrics[relation_type]['fn'] += 1

        relation_type_metrics[relation_type]['total'] += 1

    # Calculate overall metrics
    def calculate_metrics(cm):
        tp, tn, fp, fn = cm['tp'], cm['tn'], cm['fp'], cm['fn']

        # Check for all TN case (no positive predictions or ground truth)
        if tp == 0 and fp == 0 and fn == 0:
            precision = 0.0
            recall = 0.0
            f1 = "unavailable"
            accuracy = tn / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'confusion_matrix': cm
        }

    results = {
        'overall': calculate_metrics(overall_cm),
        'by_relation_type': {},
        'coverage': {
            'total_possible_relations': len(all_possible_triples),
            'predicted_relations': len(prediction_lookup) if prediction_lookup else 0,
            'coverage_percentage': len(prediction_lookup) / len(all_possible_triples) if all_possible_triples else 0
        }
    }

    # Calculate per-relation-type metrics
    for rel_type, cm in relation_type_metrics.items():
        results['by_relation_type'][rel_type] = calculate_metrics(cm)
        results['by_relation_type'][rel_type]['coverage'] = cm['predicted'] / cm['total'] if cm['total'] > 0 else 0

    return results


def evaluate_relation_classification_batch(
    predictions_path: str,
    expected_triples: List[Dict]
) -> Tuple[Optional[EvaluationMetrics], Dict[str, Dict]]:
    """
    Evaluate batch relation classification predictions against expected triples.

    Args:
        predictions_path: Path to predictions JSON file (list of results)
        expected_triples: List of expected triples with ground truth

    Returns:
        Tuple of (overall EvaluationMetrics, per_relation_metrics_dict) or (None, {}) if evaluation fails
    """
    try:
        # Load predictions
        with open(predictions_path, 'r') as f:
            predictions = json.load(f)

        if not isinstance(predictions, list):
            predictions = [predictions]

        # Normalize entity names for consistent matching
        from .file_utils import normalize_entity_name
        
        # Create lookup for expected results (with normalized keys)
        expected_lookup = {}
        for triple in expected_triples:
            from_entity = normalize_entity_name(triple.get('from_entity', ''))
            to_entity = normalize_entity_name(triple.get('to_entity', ''))
            relation = triple.get('relationship', '').lower().strip()
            key = (from_entity, to_entity, relation)
            expected_lookup[key] = triple['expected_result'].lower() in ['yes', 'true']

        # Create lookup for predictions (with normalized keys)
        prediction_lookup = {}
        for pred in predictions:
            # Handle both entity1/entity2 and from_entity/to_entity field names
            from_entity = normalize_entity_name(pred.get('entity1', pred.get('from_entity', '')))
            to_entity = normalize_entity_name(pred.get('entity2', pred.get('to_entity', '')))
            relation = pred.get('relation', pred.get('relationship', '')).lower().strip()
            key = (from_entity, to_entity, relation)
            pred_exists = pred.get('exists', '').lower() in ['yes', 'true']
            prediction_lookup[key] = pred_exists

        # Calculate overall metrics
        confusion_matrix = {
            'tp': 0,  # True Positives
            'tn': 0,  # True Negatives
            'fp': 0,  # False Positives
            'fn': 0   # False Negatives
        }

        # Per-relation-type metrics
        relation_metrics = {
            'assign': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'count': 0},
            'permit': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'count': 0},
            'prohibit': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'count': 0}
        }

        # Iterate over EXPECTED triples to ensure all are evaluated (including missing predictions)
        for triple in expected_triples:
            from_entity = normalize_entity_name(triple.get('from_entity', ''))
            to_entity = normalize_entity_name(triple.get('to_entity', ''))
            relation = triple.get('relationship', '').lower().strip()
            key = (from_entity, to_entity, relation)
            
            gt_exists = expected_lookup.get(key, False)
            pred_exists = prediction_lookup.get(key, False)  # False if prediction missing

            # Update overall confusion matrix
            if pred_exists and gt_exists:
                confusion_matrix['tp'] += 1
                relation_metrics[relation]['tp'] += 1
            elif not pred_exists and not gt_exists:
                confusion_matrix['tn'] += 1
                relation_metrics[relation]['tn'] += 1
            elif pred_exists and not gt_exists:
                confusion_matrix['fp'] += 1
                relation_metrics[relation]['fp'] += 1
            elif not pred_exists and gt_exists:
                confusion_matrix['fn'] += 1
                relation_metrics[relation]['fn'] += 1

            relation_metrics[relation]['count'] += 1
        
        # Also count predictions that don't match any expected triple as FP
        for key, pred_exists in prediction_lookup.items():
            if key not in expected_lookup and pred_exists:
                # This prediction doesn't match any expected triple - count as FP
                from_entity, to_entity, relation = key
                confusion_matrix['fp'] += 1
                if relation in relation_metrics:
                    relation_metrics[relation]['fp'] += 1
                    relation_metrics[relation]['count'] += 1

        # Calculate overall metrics
        tp = confusion_matrix['tp']
        tn = confusion_matrix['tn']
        fp = confusion_matrix['fp']
        fn = confusion_matrix['fn']

        # Check for all TN case (no positive predictions or ground truth)
        if tp == 0 and fp == 0 and fn == 0:
            precision = 0.0
            recall = 0.0
            f1 = "unavailable"
            accuracy = tn / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        overall_metrics = EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
            confusion_matrix=confusion_matrix
        )

        # Calculate per-relation metrics
        per_relation_results = {}
        for rel_type, cm in relation_metrics.items():
            if cm['count'] > 0:
                tp_r = cm['tp']
                tn_r = cm['tn']
                fp_r = cm['fp']
                fn_r = cm['fn']

                # Check for all TN case per relation type
                if tp_r == 0 and fp_r == 0 and fn_r == 0:
                    precision_r = 0.0
                    recall_r = 0.0
                    f1_r = "unavailable"
                else:
                    precision_r = tp_r / (tp_r + fp_r) if (tp_r + fp_r) > 0 else 0.0
                    recall_r = tp_r / (tp_r + fn_r) if (tp_r + fn_r) > 0 else 0.0
                    f1_r = 2 * (precision_r * recall_r) / (precision_r + recall_r) if (precision_r + recall_r) > 0 else 0.0

                per_relation_results[rel_type] = {
                    'precision': precision_r,
                    'recall': recall_r,
                    'f1': f1_r,
                    'count': cm['count'],
                    'confusion_matrix': cm
                }

        return overall_metrics, per_relation_results

    except Exception as e:
        print(f"Error in batch relation evaluation: {e}")
        return None, {}


def save_evaluation_report(
    metrics: EvaluationMetrics,
    output_path: str,
    method: str = "entity_extraction"
) -> None:
    """
    Save evaluation results to file.

    Args:
        metrics: Evaluation metrics
        output_path: Output file path
        method: Evaluation method name
    """
    report_lines = []
    report_lines.append(f"=== {method.upper().replace('_', ' ')} EVALUATION SUMMARY ===")
    report_lines.append("")

    if metrics.confusion_matrix:
        cm = metrics.confusion_matrix
        report_lines.append("Confusion Matrix:")
        report_lines.append(f"  True Positives (TP):  {cm.get('tp', 0):>4}")
        report_lines.append(f"  True Negatives (TN):  {cm.get('tn', 0):>4}")
        report_lines.append(f"  False Positives (FP): {cm.get('fp', 0):>4}")
        report_lines.append(f"  False Negatives (FN): {cm.get('fn', 0):>4}")
        report_lines.append("")

    report_lines.append("Classification Metrics:")
    report_lines.append(f"   Precision:   {metrics.precision:.4f}")
    report_lines.append(f"   Recall:      {metrics.recall:.4f}")
    if isinstance(metrics.f1, str):
        report_lines.append(f"   F1 Score:    {metrics.f1}")
    else:
        report_lines.append(f"   F1 Score:    {metrics.f1:.4f}")
    if metrics.accuracy is not None:
        report_lines.append(f"   Accuracy:    {metrics.accuracy:.4f}")

    # Create outputs directory if it doesn't exist
    from .config import PROJECT_ROOT
    output_dir = create_output_directory(str(PROJECT_ROOT / "outputs"))
    report_path = output_dir / f"{Path(output_path).stem}_evaluation_report.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"📄 Detailed evaluation report saved to: {report_path}")


# ============================================================================
# BATCH EVALUATION HELPERS
# ============================================================================

def aggregate_evaluation_metrics(metrics_list: List[EvaluationMetrics]) -> Dict[str, float]:
    """
    Aggregate evaluation metrics across multiple files.

    Args:
        metrics_list: List of evaluation metrics

    Returns:
        Dictionary with averaged metrics
    """
    if not metrics_list:
        return {}

    # Handle F1 aggregation - exclude "unavailable" values
    valid_f1s = [m.f1 for m in metrics_list if isinstance(m.f1, (int, float))]
    avg_f1 = sum(valid_f1s) / len(valid_f1s) if valid_f1s else "unavailable"

    avg_metrics = {
        "precision": sum(m.precision for m in metrics_list) / len(metrics_list),
        "recall": sum(m.recall for m in metrics_list) / len(metrics_list),
        "f1": avg_f1,
    }

    if all(m.accuracy is not None for m in metrics_list):
        avg_metrics["accuracy"] = sum(m.accuracy for m in metrics_list) / len(metrics_list)

    return avg_metrics


def aggregate_evaluation_metrics_with_relations(
    metrics_list: List[EvaluationMetrics],
    per_relation_metrics_list: Optional[List[Dict[str, Dict]]] = None
) -> Dict[str, Any]:
    """
    Aggregate evaluation metrics including per-relation-type averages.

    Args:
        metrics_list: List of overall evaluation metrics
        per_relation_metrics_list: List of per-relation metrics dictionaries

    Returns:
        Dictionary with averaged overall and per-relation metrics
    """
    if not metrics_list:
        return {}

    # Overall averages - handle F1 "unavailable" case
    valid_f1s = [m.f1 for m in metrics_list if isinstance(m.f1, (int, float))]
    avg_f1 = sum(valid_f1s) / len(valid_f1s) if valid_f1s else "unavailable"

    overall_avg = {
        "precision": sum(m.precision for m in metrics_list) / len(metrics_list),
        "recall": sum(m.recall for m in metrics_list) / len(metrics_list),
        "f1": avg_f1,
    }

    if all(m.accuracy is not None for m in metrics_list):
        overall_avg["accuracy"] = sum(m.accuracy for m in metrics_list) / len(metrics_list)

    result = {"overall": overall_avg}

    # Per-relation-type averages if available
    if per_relation_metrics_list:
        relation_types = set()
        for rel_metrics in per_relation_metrics_list:
            relation_types.update(rel_metrics.keys())

        per_relation_avg = {}
        for rel_type in relation_types:
            # Collect metrics for this relation type across all files
            rel_precisions = []
            rel_recalls = []
            rel_f1s = []
            rel_counts = []

            for rel_metrics in per_relation_metrics_list:
                if rel_type in rel_metrics:
                    rel_data = rel_metrics[rel_type]
                    rel_precisions.append(rel_data['precision'])
                    rel_recalls.append(rel_data['recall'])
                    rel_f1s.append(rel_data['f1'])
                    rel_counts.append(rel_data.get('count', 0))

            if rel_precisions:
                # Handle case where some F1 values might be "unavailable"
                valid_f1s = [f for f in rel_f1s if isinstance(f, (int, float))]
                avg_f1 = sum(valid_f1s) / len(valid_f1s) if valid_f1s else "unavailable"

                per_relation_avg[rel_type] = {
                    "precision": sum(rel_precisions) / len(rel_precisions),
                    "recall": sum(rel_recalls) / len(rel_recalls),
                    "f1": avg_f1,
                    "avg_count": sum(rel_counts) / len(rel_counts)
                }

        result["per_relation"] = per_relation_avg

    return result


def aggregate_evaluation_metrics_with_micro_macro(
    metrics_list: List[EvaluationMetrics],
    per_relation_metrics_list: Optional[List[Dict[str, Dict]]] = None
) -> Dict[str, Any]:
    """
    Aggregate evaluation metrics with both MICRO-F1 and MACRO-F1 calculations.

    MICRO-F1: Global calculation across all figures (sums all TP/FP/FN/TN)
    MACRO-F1: Average of per-figure F1 scores

    Args:
        metrics_list: List of overall evaluation metrics
        per_relation_metrics_list: List of per-relation metrics dictionaries

    Returns:
        Dictionary with micro/macro metrics and per-relation averages
    """
    if not metrics_list:
        return {}

    # MICRO-F1: Sum all confusion matrix values across figures
    total_tp = sum(m.confusion_matrix.get('tp', 0) for m in metrics_list if m.confusion_matrix)
    total_fp = sum(m.confusion_matrix.get('fp', 0) for m in metrics_list if m.confusion_matrix)
    total_fn = sum(m.confusion_matrix.get('fn', 0) for m in metrics_list if m.confusion_matrix)
    total_tn = sum(m.confusion_matrix.get('tn', 0) for m in metrics_list if m.confusion_matrix)

    # Calculate MICRO metrics
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    micro_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0.0

    # MACRO-F1: Average of per-figure F1 scores
    valid_f1s = [m.f1 for m in metrics_list if isinstance(m.f1, (int, float))]
    macro_f1 = sum(valid_f1s) / len(valid_f1s) if valid_f1s else "unavailable"
    macro_precision = sum(m.precision for m in metrics_list) / len(metrics_list)
    macro_recall = sum(m.recall for m in metrics_list) / len(metrics_list)

    if all(m.accuracy is not None for m in metrics_list):
        macro_accuracy = sum(m.accuracy for m in metrics_list) / len(metrics_list)
    else:
        macro_accuracy = None

    result = {
        "micro_overall": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
            "accuracy": micro_accuracy,
            "confusion_matrix": {
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "tn": total_tn
            }
        },
        "macro_overall": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
            "accuracy": macro_accuracy
        }
    }

    # Per-relation-type averages if available
    if per_relation_metrics_list:
        relation_types = set()
        for rel_metrics in per_relation_metrics_list:
            relation_types.update(rel_metrics.keys())

        per_relation_avg = {}
        for rel_type in relation_types:
            # Collect metrics for this relation type across all files
            rel_precisions = []
            rel_recalls = []
            rel_f1s = []
            rel_accuracies = []
            rel_counts = []
            
            # For micro-averaging per relation type
            rel_tp_total = 0
            rel_fp_total = 0
            rel_fn_total = 0
            rel_tn_total = 0
            has_cm = False

            for rel_metrics in per_relation_metrics_list:
                if rel_type in rel_metrics:
                    rel_data = rel_metrics[rel_type]
                    rel_precisions.append(rel_data.get('precision', 0.0))
                    rel_recalls.append(rel_data.get('recall', 0.0))
                    rel_f1s.append(rel_data.get('f1', 0.0))
                    if 'accuracy' in rel_data:
                        rel_accuracies.append(rel_data['accuracy'])
                    rel_counts.append(rel_data.get('count', 0))
                    
                    if 'confusion_matrix' in rel_data:
                        cm = rel_data['confusion_matrix']
                        rel_tp_total += cm.get('tp', 0)
                        rel_fp_total += cm.get('fp', 0)
                        rel_fn_total += cm.get('fn', 0)
                        rel_tn_total += cm.get('tn', 0)
                        has_cm = True

            if rel_precisions:
                # Macro averages
                valid_f1s = [f for f in rel_f1s if isinstance(f, (int, float))]
                macro_f1 = sum(valid_f1s) / len(valid_f1s) if valid_f1s else "unavailable"
                macro_precision = sum(rel_precisions) / len(rel_precisions)
                macro_recall = sum(rel_recalls) / len(rel_recalls)
                macro_accuracy = sum(rel_accuracies) / len(rel_accuracies) if rel_accuracies else None

                per_relation_avg[rel_type] = {
                    "macro_precision": macro_precision,
                    "macro_recall": macro_recall,
                    "macro_f1": macro_f1,
                    "macro_accuracy": macro_accuracy,
                    "avg_count": sum(rel_counts) / len(rel_counts)
                }
                
                # Micro averages if confusion matrix was available
                if has_cm:
                    micro_p = rel_tp_total / (rel_tp_total + rel_fp_total) if (rel_tp_total + rel_fp_total) > 0 else 0.0
                    micro_r = rel_tp_total / (rel_tp_total + rel_fn_total) if (rel_tp_total + rel_fn_total) > 0 else 0.0
                    micro_f1 = 2 * (micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
                    micro_acc = (rel_tp_total + rel_tn_total) / (rel_tp_total + rel_tn_total + rel_fp_total + rel_fn_total) if (rel_tp_total + rel_tn_total + rel_fp_total + rel_fn_total) > 0 else 0.0
                    
                    per_relation_avg[rel_type].update({
                        "micro_precision": micro_p,
                        "micro_recall": micro_r,
                        "micro_f1": micro_f1,
                        "micro_accuracy": micro_acc,
                        "total_confusion_matrix": {
                            "tp": rel_tp_total,
                            "fp": rel_fp_total,
                            "fn": rel_fn_total,
                            "tn": rel_tn_total
                        }
                    })
                
                # Maintain backward compatibility for keys without macro/micro prefix
                per_relation_avg[rel_type].update({
                    "precision": macro_precision,
                    "recall": macro_recall,
                    "f1": macro_f1
                })

        result["per_relation"] = per_relation_avg

    return result


def aggregate_entity_metrics_with_micro_macro(
    metrics_list: List[EvaluationMetrics]
) -> Dict[str, Any]:
    """
    Aggregate entity evaluation metrics with both MICRO-F1 and MACRO-F1 calculations.

    MICRO-F1: Global calculation across all figures (sums all TP/FP/FN)
    MACRO-F1: Average of per-figure F1 scores

    Args:
        metrics_list: List of entity evaluation metrics (one per graph/figure)

    Returns:
        Dictionary with micro/macro metrics for entity evaluation
    """
    if not metrics_list:
        return {}

    # MICRO-F1: Sum all confusion matrix values across figures (TP/FP/FN only, no TN)
    total_tp = sum(m.confusion_matrix.get('tp', 0) for m in metrics_list if m.confusion_matrix)
    total_fp = sum(m.confusion_matrix.get('fp', 0) for m in metrics_list if m.confusion_matrix)
    total_fn = sum(m.confusion_matrix.get('fn', 0) for m in metrics_list if m.confusion_matrix)

    # Calculate MICRO metrics (no TN for entities)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    # For entities, accuracy is TP / (TP + FP + FN) since no TN defined
    micro_accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

    # MACRO-F1: Average of per-figure F1 scores
    valid_f1s = [m.f1 for m in metrics_list if isinstance(m.f1, (int, float))]
    macro_f1 = sum(valid_f1s) / len(valid_f1s) if valid_f1s else "unavailable"
    macro_precision = sum(m.precision for m in metrics_list) / len(metrics_list)
    macro_recall = sum(m.recall for m in metrics_list) / len(metrics_list)

    if all(m.accuracy is not None for m in metrics_list):
        macro_accuracy = sum(m.accuracy for m in metrics_list) / len(metrics_list)
    else:
        macro_accuracy = None

    result = {
        "micro_overall": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
            "accuracy": micro_accuracy,
            "confusion_matrix": {
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn
            }
        },
        "macro_overall": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
            "accuracy": macro_accuracy
        }
    }

    return result


def compute_overall_accuracy(entity_metrics: Dict[str, Any], relation_metrics: Dict[str, Any]) -> float:
    """
    Compute overall accuracy by combining entity and relation F1 scores.

    Args:
        entity_metrics: Dictionary with entity evaluation metrics
        relation_metrics: Dictionary with relation evaluation metrics

    Returns:
        Overall accuracy as weighted average of entity and relation F1 scores
    """
    entity_f1 = entity_metrics.get('f1', 0.0)
    relation_f1 = relation_metrics.get('f1', 0.0)

    # Handle "unavailable" F1 case
    if isinstance(entity_f1, str) and entity_f1 == "unavailable":
        entity_f1 = 0.0
    if isinstance(relation_f1, str) and relation_f1 == "unavailable":
        relation_f1 = 0.0

    # Convert to numeric if needed
    entity_f1 = float(entity_f1) if isinstance(entity_f1, (int, float)) else 0.0
    relation_f1 = float(relation_f1) if isinstance(relation_f1, (int, float)) else 0.0

    # Weighted average (equal weights for entities and relations)
    overall_accuracy = (entity_f1 * 0.5) + (relation_f1 * 0.5)

    return overall_accuracy


def print_aggregated_results_with_micro_macro(
    metrics_list: List[EvaluationMetrics],
    evaluated_results: List[Dict],
    label: str = "FINAL AVERAGED",
    per_relation_metrics_list: Optional[List[Dict[str, Dict]]] = None
) -> None:
    """
    Print comprehensive aggregated evaluation results with MICRO/MACRO F1 and per-figure metrics.

    Args:
        metrics_list: List of evaluation metrics
        evaluated_results: List of evaluated result dictionaries (contains per-figure metrics)
        label: Label for the results
        per_relation_metrics_list: List of per-relation metrics dictionaries
    """
    if not metrics_list:
        return

    # Get micro/macro averages
    micro_macro_metrics = aggregate_evaluation_metrics_with_micro_macro(metrics_list, per_relation_metrics_list)

    print(f"\n{'='*80}")
    print(f"📊 {label} RESULTS (across {len(metrics_list)} figures)")
    print(f"{'='*80}")

    # Per-figure results
    print("📋 PER-FIGURE RESULTS:")
    print("─" * 80)
    print(f"{'Image':<30} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}  {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("-" * 80)

    for result in evaluated_results:
        if result.get("evaluated", False):
            image_name = result.get("image", "unknown").replace("_labeled", "").replace("_labeled_b", "")
            metrics = result.get("metrics") or {}
            cm = metrics.get("confusion_matrix") or {}

            tp = cm.get('tp', 0)
            fp = cm.get('fp', 0)
            fn = cm.get('fn', 0)
            tn = cm.get('tn', 0)

            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            f1 = metrics.get('f1', 0)

            print(f"{image_name:<30} {tp:>4} {fp:>4} {fn:>4} {tn:>4}  {precision:>6.3f} {recall:>6.3f} {f1:>6.3f}")

    print("\n" + "="*80)

    # MICRO-F1 results (global across all figures)
    micro = micro_macro_metrics.get("micro_overall", {})
    print("🔬 MICRO-F1 RESULTS (Global calculation across all figures):")
    print("─" * 80)
    micro_cm = micro.get('confusion_matrix') or {}
    if micro_cm:
        print(f"   Total TP: {micro_cm.get('tp', 0):>4}")
        print(f"   Total FP: {micro_cm.get('fp', 0):>4}")
        print(f"   Total FN: {micro_cm.get('fn', 0):>4}")
        print(f"   Total TN: {micro_cm.get('tn', 0):>4}")
    print(f"   Precision: {micro['precision']:.4f}")
    print(f"   Recall:    {micro['recall']:.4f}")
    print(f"   F1:        {micro['f1']:.4f}")
    if micro.get('accuracy') is not None:
        print(f"   Accuracy:  {micro['accuracy']:.4f}")

    # MACRO-F1 results (average across figures)
    macro = micro_macro_metrics.get("macro_overall", {})
    print("\n📊 MACRO-F1 RESULTS (Average across all figures):")
    print("─" * 80)
    print(f"   Precision: {macro['precision']:.4f}")
    print(f"   Recall:    {macro['recall']:.4f}")
    f1_value = macro['f1']
    if isinstance(f1_value, str):
        print(f"   F1:        {f1_value}")
    else:
        print(f"   F1:        {f1_value:.4f}")
    if macro.get('accuracy') is not None:
        print(f"   Accuracy:  {macro['accuracy']:.4f}")

    # Per-relation metrics
    per_relation = micro_macro_metrics.get("per_relation", {})
    if per_relation:
        print("\n📈 PER-RELATION AVERAGES:")
        for rel_type, metrics in sorted(per_relation.items()):
            count = metrics['avg_count']
            f1_value = metrics['f1']
            cm = metrics.get('total_confusion_matrix', {})
            tp = cm.get('tp', 0)
            tn = cm.get('tn', 0)
            fp = cm.get('fp', 0)
            fn = cm.get('fn', 0)

            if isinstance(f1_value, str):
                print(f"   {rel_type.title()}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={f1_value} (TP={tp}, TN={tn}, FP={fp}, FN={fn}, avg_n={count:.1f})")
            else:
                print(f"   {rel_type.title()}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={f1_value:.4f} (TP={tp}, TN={tn}, FP={fp}, FN={fn}, avg_n={count:.1f})")

    print(f"{'='*80}\n")


def print_aggregated_results(
    metrics_list: List[EvaluationMetrics],
    label: str = "FINAL AVERAGED",
    per_relation_metrics_list: Optional[List[Dict[str, Dict]]] = None
) -> None:
    """
    Print comprehensive aggregated evaluation results.

    Args:
        metrics_list: List of evaluation metrics
        label: Label for the results
        per_relation_metrics_list: List of per-relation metrics dictionaries
    """
    if not metrics_list:
        return

    # Get comprehensive averages
    avg_metrics = aggregate_evaluation_metrics_with_relations(metrics_list, per_relation_metrics_list)

    print(f"\n{'='*80}")
    print(f"📊 {label} RESULTS (across {len(metrics_list)} files)")
    print(f"{'='*80}")

    # Overall metrics
    overall = avg_metrics.get("overall", {})
    print(f"   Precision: {overall['precision']:.4f}")
    print(f"   Recall:    {overall['recall']:.4f}")
    f1_value = overall['f1']
    if isinstance(f1_value, str):
        print(f"   F1:        {f1_value}")
    else:
        print(f"   F1:        {f1_value:.4f}")
    if "accuracy" in overall:
        print(f"   Accuracy:  {overall['accuracy']:.4f}")

    # Per-relation metrics
    per_relation = avg_metrics.get("per_relation", {})
    if per_relation:
        print(f"\n📈 PER-RELATION AVERAGES:")
        for rel_type, metrics in sorted(per_relation.items()):
            count = metrics['avg_count']
            f1_value = metrics['f1']
            if isinstance(f1_value, str):
                print(f"   {rel_type.title()}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={f1_value} (avg_n={count:.1f})")
            else:
                print(f"   {rel_type.title()}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={f1_value:.4f} (avg_n={count:.1f})")

    print(f"{'='*80}\n")


# ============================================================================
# PATH GENERATION EVALUATION (END-TO-END)
# ============================================================================

def evaluate_path_generation(
    prediction_data: Dict,
    ground_truth_path: str,
    quiet: bool = False,
    fuzzy_matching: bool = False
) -> Dict[str, Any]:
    """
    Evaluate path generation (end-to-end) against ground truth.
    
    This evaluates both:
    1. Entity extraction (nodes)
    2. Relation extraction (edges)
    
    Args:
        prediction_data: Prediction dictionary with 'nodes' and 'edges'
        ground_truth_path: Path to ground truth JSON file
        quiet: If True, suppress console output
        
    Returns:
        Dictionary with entity_metrics, relation_metrics, and combined_metrics
    """
    try:
        # Load ground truth
        gt_data = load_json(ground_truth_path)
        
        # 1. Evaluate Entity Extraction
        entity_metrics = _evaluate_path_generation_entities(prediction_data, gt_data, quiet)

        # 2. Evaluate Relation Extraction
        relation_metrics = _evaluate_path_generation_relations(prediction_data, gt_data, quiet, fuzzy_matching)

        # 3. Calculate Combined Metrics (without path evaluation)
        combined_metrics = _calculate_combined_metrics_with_paths(entity_metrics, relation_metrics, None)

        if not quiet:
            _print_path_generation_report(entity_metrics, relation_metrics, None, combined_metrics)

        return {
            "entity_metrics": entity_metrics,
            "relation_metrics": relation_metrics,
            "combined_metrics": combined_metrics
        }
        
    except Exception as e:
        if not quiet:
            print(f"❌ Error in path generation evaluation: {e}")
        return {
            "entity_metrics": None,
            "relation_metrics": None,
            "path_metrics": None,
            "combined_metrics": None,
            "error": str(e)
        }


def _entity_name_for_display(normalized_name: str) -> str:
    """Convert normalized entity name (may have spaces) to underscore form for display in JSON."""
    if not normalized_name:
        return normalized_name
    return "_".join(normalized_name.lower().split())


def _evaluate_path_generation_entities(
    prediction_data: Dict,
    gt_data: Dict,
    quiet: bool = False
) -> Dict[str, Any]:
    """Evaluate entity extraction from path generation."""
    # Extract predicted entities
    pred_entities = set()
    for node in prediction_data.get("nodes", []):
        content = node.get("content", "")
        if content:
            normalized = normalize_entity_name(content)
            entity_type = node.get("type", "unknown")
            pred_entities.add((normalized, entity_type))
    
    # Extract ground truth entities
    gt_entities = _extract_ground_truth_entities(gt_data)
    
    # Calculate metrics
    # True positives: entities in both prediction and ground truth
    tp_entities = pred_entities & gt_entities
    # False positives: entities in prediction but not in ground truth
    fp_entities = pred_entities - gt_entities
    # False negatives: entities in ground truth but not in prediction
    fn_entities = gt_entities - pred_entities
    
    tp = len(tp_entities)
    fp = len(fp_entities)
    fn = len(fn_entities)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    # Serialize entity lists with underscore form for display (matching uses normalized form internally)
    def _to_display_list(entities_set):
        return [[_entity_name_for_display(name), entity_type] for name, entity_type in entities_set]

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_count": len(pred_entities),
        "ground_truth_count": len(gt_entities),
        "tp_entities": _to_display_list(tp_entities),
        "fp_entities": _to_display_list(fp_entities),
        "fn_entities": _to_display_list(fn_entities),
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
    }


def _find_fuzzy_entity_match(pred_entity: str, gt_entities: Set[str], threshold: float = 0.8) -> Optional[str]:
    """
    Find a fuzzy match for a predicted entity name among ground truth entities.

    Args:
        pred_entity: Predicted entity name
        gt_entities: Set of ground truth entity names
        threshold: Similarity threshold (0-1)

    Returns:
        Best matching ground truth entity name, or None if no good match
    """
    from difflib import SequenceMatcher

    best_match = None
    best_score = 0.0

    for gt_entity in gt_entities:
        # Calculate similarity ratio
        score = SequenceMatcher(None, pred_entity.lower(), gt_entity.lower()).ratio()
        if score > best_score and score >= threshold:
            best_score = score
            best_match = gt_entity

    return best_match if best_score >= threshold else None


def _evaluate_path_generation_relations(
    prediction_data: Dict,
    gt_data: Dict,
    quiet: bool = False,
    fuzzy_matching: bool = False
) -> Dict[str, Any]:
    """Evaluate relation extraction from path generation."""
    # Extract predicted relations (edges) using centralized normalization
    from .file_utils import triple_from_prediction
    pred_relations = set()
    pred_relations_with_type = {}

    for edge in prediction_data.get("edges", []):
        subject, relation, object = triple_from_prediction(edge)

        if subject and object and relation != "unknown":
            rel_key = (subject, object, relation)
            pred_relations.add(rel_key)
            if relation not in pred_relations_with_type:
                pred_relations_with_type[relation] = set()
            pred_relations_with_type[relation].add((subject, object))
    
    # Extract ground truth relations using centralized normalization
    from .file_utils import triple_from_ground_truth
    gt_relations = set()
    gt_relations_with_type = {}

    # Assignments (assign)
    assignments = gt_data.get("assignments", {})
    if isinstance(assignments, dict):
        assignments = assignments.values()
    for assign in assignments:
        subject, relation, object = triple_from_ground_truth(assign, "assign")
        if subject and object:
            gt_relations.add((subject, object, relation))
            if relation not in gt_relations_with_type:
                gt_relations_with_type[relation] = set()
            gt_relations_with_type[relation].add((subject, object))

    # Associations (permit)
    associations = gt_data.get("associations", {})
    if isinstance(associations, dict):
        associations = associations.values()
    for assoc in associations:
        subject, relation, object = triple_from_ground_truth(assoc, "permit")
        if subject and object:
            gt_relations.add((subject, object, relation))
            if relation not in gt_relations_with_type:
                gt_relations_with_type[relation] = set()
            gt_relations_with_type[relation].add((subject, object))

    # Prohibitions (prohibit)
    prohibitions = gt_data.get("prohibitions", {})
    if isinstance(prohibitions, dict):
        prohibitions = prohibitions.values()
    for prohib in prohibitions:
        subject, relation, object = triple_from_ground_truth(prohib, "prohibit")
        if subject and object:
            gt_relations.add((subject, object, relation))
            if relation not in gt_relations_with_type:
                gt_relations_with_type[relation] = set()
            gt_relations_with_type[relation].add((subject, object))
    
    # Calculate initial metrics with exact matching
    tp_relations = pred_relations & gt_relations
    fp_relations = pred_relations - gt_relations
    fn_relations = gt_relations - pred_relations

    # Optional fuzzy matching to improve recall
    if fuzzy_matching:
        # Get all ground truth entity names for fuzzy matching
        gt_entity_names = set()
        for source, target, _ in gt_relations:
            gt_entity_names.add(source)
            gt_entity_names.add(target)

        # Try to match FP predictions with FN ground truth using fuzzy entity matching
        fuzzy_matches = set()
        remaining_fp = set()
        remaining_fn = set(fn_relations)

        for pred_rel in fp_relations:
            pred_source, pred_target, pred_rel_type = pred_rel

            # Look for fuzzy matches in remaining false negatives
            best_match = None
            for gt_rel in remaining_fn:
                gt_source, gt_target, gt_rel_type = gt_rel

                # Only match same relation type
                if pred_rel_type != gt_rel_type:
                    continue

                # Try fuzzy matching for source and target
                fuzzy_source = _find_fuzzy_entity_match(pred_source, gt_entity_names)
                fuzzy_target = _find_fuzzy_entity_match(pred_target, gt_entity_names)

                # Check if fuzzy matches align with ground truth
                if fuzzy_source == gt_source and fuzzy_target == gt_target:
                    best_match = gt_rel
                    break

            if best_match:
                fuzzy_matches.add((pred_rel, best_match))
                remaining_fn.discard(best_match)
            else:
                remaining_fp.add(pred_rel)

        # Update metrics with fuzzy matches
        tp_relations = tp_relations | {(pred_rel, gt_rel) for pred_rel, gt_rel in fuzzy_matches}
        fp_relations = remaining_fp
        fn_relations = remaining_fn

    tp = len(tp_relations)
    fp = len(fp_relations)
    fn = len(fn_relations)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    # Calculate per-relation-type metrics
    per_relation_metrics = {}
    for rel_type in ["assign", "permit", "prohibit"]:
        pred_set = pred_relations_with_type.get(rel_type, set())
        gt_set = gt_relations_with_type.get(rel_type, set())
        
        rel_tp = len(pred_set & gt_set)
        rel_fp = len(pred_set - gt_set)
        rel_fn = len(gt_set - pred_set)
        
        rel_precision = rel_tp / (rel_tp + rel_fp) if (rel_tp + rel_fp) > 0 else 0.0
        rel_recall = rel_tp / (rel_tp + rel_fn) if (rel_tp + rel_fn) > 0 else 0.0
        rel_f1 = 2 * (rel_precision * rel_recall) / (rel_precision + rel_recall) if (rel_precision + rel_recall) > 0 else 0.0
        rel_accuracy = rel_tp / (rel_tp + rel_fp + rel_fn) if (rel_tp + rel_fp + rel_fn) > 0 else 0.0
        
        per_relation_metrics[rel_type] = {
            "precision": rel_precision,
            "recall": rel_recall,
            "f1": rel_f1,
            "accuracy": rel_accuracy,
            "tp": rel_tp,
            "fp": rel_fp,
            "fn": rel_fn,
            "tn": 0,  # TN not computed for path generation relations
            "predicted_count": len(pred_set),
            "ground_truth_count": len(gt_set),
            "count": len(gt_set),  # Add count key for aggregation
            "confusion_matrix": {
                "tp": rel_tp,
                "fp": rel_fp,
                "fn": rel_fn,
                "tn": 0
            }
        }
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": 0,  # TN not computed for path generation relations
        "predicted_count": len(pred_relations),
        "ground_truth_count": len(gt_relations),
        "per_relation": per_relation_metrics,
        "tp_relations": list(tp_relations),
        "fp_relations": list(fp_relations),
        "fn_relations": list(fn_relations),
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": 0
        }
    }


def _evaluate_path_generation_paths(
    prediction_data: Dict,
    gt_data: Dict,
    quiet: bool = False
) -> Dict[str, Any]:
    """Evaluate path enumeration from path generation."""

    # Generate expected paths from ground truth
    try:
        gt_paths = _generate_paths_from_ground_truth(gt_data)
        gt_path_set = set()
        for path in gt_paths:
            # Convert path to tuple of normalized entity names
            path_tuple = tuple(normalize_entity_name(node) for node in path)
            gt_path_set.add(path_tuple)
    except Exception as e:
        if not quiet:
            print(f"Warning: Could not generate ground truth paths: {e}")
        gt_path_set = set()

    # For path generation, we generate paths from the predicted nodes and edges
    try:
        pred_paths = _generate_paths_from_prediction(prediction_data)
        pred_path_set = set()
        for path in pred_paths:
            # Convert path to tuple of normalized entity names
            path_tuple = tuple(normalize_entity_name(node) for node in path)
            pred_path_set.add(path_tuple)
    except Exception as e:
        if not quiet:
            print(f"Warning: Could not generate predicted paths: {e}")
        pred_path_set = set()

    # Calculate path metrics
    tp_paths = pred_path_set & gt_path_set
    fp_paths = pred_path_set - gt_path_set
    fn_paths = gt_path_set - pred_path_set

    tp = len(tp_paths)
    fp = len(fp_paths)
    fn = len(fn_paths)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_count": len(pred_path_set),
        "ground_truth_count": len(gt_path_set),
        "tp_paths": list(tp_paths),
        "fp_paths": list(fp_paths),
        "fn_paths": list(fn_paths)
    }


def _generate_paths_from_ground_truth(gt_data: Dict) -> List[List[str]]:
    """Generate all possible paths from user_attributes to policy_classes in ground truth."""
    paths = []

    # Extract entities by type
    user_attrs = []
    object_attrs = []
    policy_classes = []

    if "policy_elements" in gt_data:
        policy_elements = gt_data["policy_elements"]
        user_attrs = policy_elements.get("user_attributes", [])
        object_attrs = policy_elements.get("object_attributes", [])
        policy_classes = policy_elements.get("policy_classes", [])
        if isinstance(policy_classes, str):
            policy_classes = [policy_classes]

    # Build graph from assignments and associations
    graph = {}  # entity -> list of connected entities

    # Add assignments (assign relationships)
    assignments = gt_data.get("assignments", {})
    if isinstance(assignments, dict):
        assignments = assignments.values()
    for assign in assignments:
        source = assign.get("from", "")
        target = assign.get("to", "")
        if source and target:
            if source not in graph:
                graph[source] = []
            if target not in graph[source]:
                graph[source].append(target)

    # Add associations (permit relationships)
    associations = gt_data.get("associations", {})
    if isinstance(associations, dict):
        associations = associations.values()
    for assoc in associations:
        source = assoc.get("from", "")
        target = assoc.get("to", "")
        if source and target:
            if source not in graph:
                graph[source] = []
            if target not in graph[source]:
                graph[source].append(target)

    # Generate paths from each user_attribute to each policy_class
    for user_attr in user_attrs:
        for policy_class in policy_classes:
            paths_from_user = _find_paths_dfs(graph, user_attr, policy_class, visited=set())
            paths.extend(paths_from_user)

    return paths


def _generate_paths_from_prediction(prediction_data: Dict) -> List[List[str]]:
    """Generate all possible paths from user_attributes to policy_classes in prediction."""
    paths = []

    # Build entity type mapping from nodes
    entity_types = {}
    for node in prediction_data.get("nodes", []):
        content = node.get("content", "")
        node_type = node.get("type", "")
        if content:
            entity_types[content] = node_type

    # Extract entities by type
    user_attrs = [entity for entity, entity_type in entity_types.items() if entity_type == "user_attributes"]
    policy_classes = [entity for entity, entity_type in entity_types.items() if entity_type == "policy_classes"]

    # Build graph from edges
    graph = {}  # entity -> list of connected entities
    for edge in prediction_data.get("edges", []):
        source = edge.get("source_name", "")
        target = edge.get("target_name", "")
        if source and target:
            if source not in graph:
                graph[source] = []
            if target not in graph[source]:
                graph[source].append(target)

    # Generate paths from each user_attribute to each policy_class
    for user_attr in user_attrs:
        for policy_class in policy_classes:
            paths_from_user = _find_paths_dfs(graph, user_attr, policy_class, visited=set())
            paths.extend(paths_from_user)

    return paths


def _find_paths_dfs(graph: Dict[str, List[str]], start: str, end: str, visited: set = None) -> List[List[str]]:
    """Find all paths from start to end using DFS."""
    if visited is None:
        visited = set()

    visited.add(start)
    paths = []

    if start == end:
        return [[start]]

    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            sub_paths = _find_paths_dfs(graph, neighbor, end, visited.copy())
            for sub_path in sub_paths:
                paths.append([start] + sub_path)

    return paths


def _calculate_combined_metrics_with_paths(
    entity_metrics: Dict[str, Any],
    relation_metrics: Dict[str, Any],
    path_metrics: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Calculate combined metrics from entity and relation metrics (path metrics optional)."""
    if not entity_metrics or not relation_metrics:
        return {}

    # Weighted average (equal weight for entities and relations, path metrics ignored)
    combined_precision = (entity_metrics["precision"] + relation_metrics["precision"]) / 2
    combined_recall = (entity_metrics["recall"] + relation_metrics["recall"]) / 2
    combined_f1 = (entity_metrics["f1"] + relation_metrics["f1"]) / 2
    combined_accuracy = (entity_metrics["accuracy"] + relation_metrics["accuracy"]) / 2

    return {
        "precision": combined_precision,
        "recall": combined_recall,
        "f1": combined_f1,
        "accuracy": combined_accuracy,
        "entity_weight": 1/2,
        "relation_weight": 1/2
    }


def _print_path_generation_report(
    entity_metrics: Dict[str, Any],
    relation_metrics: Dict[str, Any],
    path_metrics: Optional[Dict[str, Any]],
    combined_metrics: Dict[str, Any]
) -> None:
    """Print path generation evaluation report."""
    print(f"\n{'='*80}")
    print("📊 PATH GENERATION EVALUATION (End-to-End)")
    print(f"{'='*80}")
    
    # Entity Extraction Results
    print(f"\n📋 ENTITY EXTRACTION:")
    print(f"   Predicted: {entity_metrics['predicted_count']}, Ground Truth: {entity_metrics['ground_truth_count']}")
    print(f"   TP: {entity_metrics['tp']}, FP: {entity_metrics['fp']}, FN: {entity_metrics['fn']}")
    print(f"   Precision: {entity_metrics['precision']:.4f}")
    print(f"   Recall:    {entity_metrics['recall']:.4f}")
    print(f"   F1:        {entity_metrics['f1']:.4f}")
    
    # Relation Extraction Results
    print(f"\n🔗 RELATION EXTRACTION:")
    print(f"   Predicted: {relation_metrics['predicted_count']}, Ground Truth: {relation_metrics['ground_truth_count']}")
    print(f"   TP: {relation_metrics['tp']}, TN: {relation_metrics.get('tn', 0)}, FP: {relation_metrics['fp']}, FN: {relation_metrics['fn']}")
    print(f"   Precision: {relation_metrics['precision']:.4f}")
    print(f"   Recall:    {relation_metrics['recall']:.4f}")
    print(f"   F1:        {relation_metrics['f1']:.4f}")
    
    # Per-relation metrics
    per_relation = relation_metrics.get("per_relation", {})
    if per_relation:
        print(f"\n   Per-Relation Type:")
        for rel_type in ["assign", "permit", "prohibit"]:
            if rel_type in per_relation:
                m = per_relation[rel_type]
                tn_count = m.get('tn', 0)
                print(f"     {rel_type.title()}: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f} "
                      f"(TP={m['tp']}, TN={tn_count}, FP={m['fp']}, FN={m['fn']})")

    # Path Enumeration Results (only if path_metrics is provided)
    if path_metrics:
        print(f"\n🛤️ PATH ENUMERATION:")
        print(f"   Predicted: {path_metrics['predicted_count']}, Ground Truth: {path_metrics['ground_truth_count']}")
        print(f"   TP: {path_metrics['tp']}, FP: {path_metrics['fp']}, FN: {path_metrics['fn']}")
        print(f"   Precision: {path_metrics['precision']:.4f}")
        print(f"   Recall:    {path_metrics['recall']:.4f}")
        print(f"   F1:        {path_metrics['f1']:.4f}")

    # Combined Metrics
    if combined_metrics:
        print(f"\n🎯 COMBINED METRICS:")
        print(f"   Precision: {combined_metrics['precision']:.4f}")
        print(f"   Recall:    {combined_metrics['recall']:.4f}")
        print(f"   F1:        {combined_metrics['f1']:.4f}")
    
    print(f"{'='*80}\n")

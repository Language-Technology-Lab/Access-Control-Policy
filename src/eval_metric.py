"""
Knowledge Graph Evaluation Metrics for Access Control Systems

This module provides comprehensive evaluation metrics for comparing predicted knowledge graphs
against ground truth annotations, inspired by the chemoTimelinesEval approach.

Evaluation Components:
1. Entity Extraction: Compare identified entities (users, roles, resources, policies)
2. Assignment Relations: Compare hierarchical relationships (user->role->resource chains)
3. Association Relations: Compare permission relationships with weights
4. Prohibition Relations: Compare denied permissions

Evaluation Modes:
- Strict: Exact matches only
- Relaxed: Allow partial matches and semantic equivalences
"""

import json
from typing import Dict, List, Set, Tuple, Any, Optional
from datetime import datetime

# Import normalize_entity_name from file_utils for consistency
try:
    from .file_utils import normalize_entity_name
except ImportError:
    # Fallback if import fails
    def normalize_entity_name(name: str) -> str:
        """Fallback normalize_entity_name"""
        if not name:
            return ""
        return " ".join(name.lower().split()).replace("_", " ")


class KnowledgeGraphEvaluator:
    """
    Evaluator for knowledge graph predictions against ground truth
    """

    def __init__(self):
        self.strict_mode = True
        self.relaxed_mode = False

    def _categorize_relationship_type(self, rel_type: str) -> str:
        """
        Categorize relationship type into assignments, associations, or prohibitions
        
        Args:
            rel_type: Relationship type string (normalized to lowercase)
            
        Returns:
            Category name: "assignments", "associations", or "prohibitions"
        """
        rel_type_lower = rel_type.lower()
        
        # Check for assignment relationships (hierarchical assignments)
        if "assign" in rel_type_lower:
            return "assignments"
        
        # Check for permission/access relationships (associations)
        permit_keywords = [
            "permit",
            "allow",
            "access",
            "read",
            "write",
            "execute",
            "grant",
            "associate",
            "association"
        ]
        if any(keyword in rel_type_lower for keyword in permit_keywords):
            return "associations"
        
        # Check for prohibition/denial relationships
        prohibit_keywords = ["prohibit", "deny", "block", "restrict", "forbid"]
        if any(keyword in rel_type_lower for keyword in prohibit_keywords):
            return "prohibitions"
        
        # Default to assignments for unknown types
        return "assignments"

    def extract_entities_from_gt(self, gt_data: Dict) -> Set[Tuple[str, str]]:
        """
        Direct entity extraction from ground truth data
        """
        entities = set()
        if "policy_elements" not in gt_data:
            if "nodes" in gt_data:
                for node in gt_data["nodes"]:
                    if "content" in node:
                        name = normalize_entity_name(node["content"])
                        type_ = node.get("type", "resource")
                        entities.add((name, type_))
            return entities

        pe = gt_data["policy_elements"]
        for attr in pe.get("user_attributes", []):
            entities.add((normalize_entity_name(attr), "user"))
        
        # Handle both typo "object_attribues" and correct "object_attributes"
        objs = pe.get("object_attributes", []) or pe.get("object_attribues", [])
        for attr in objs:
            entities.add((normalize_entity_name(attr), "resource"))
        
        pc = pe.get("policy_classes", [])
        if isinstance(pc, str): pc = [pc]
        for attr in pc:
            entities.add((normalize_entity_name(attr), "policy_classes"))

        return entities

    def extract_entities_from_pred(self, pred_data: Dict) -> Set[Tuple[str, str]]:
        """
        Direct entity extraction from prediction data

        Prediction format: nodes with "label": "entity_name" (new format) or "content": "entity_name" (legacy)

        Returns:
            Set of (entity_name, entity_type) tuples
        """
        entities = set()

        # Direct extraction from nodes
        if "nodes" in pred_data:
            for node in pred_data["nodes"]:
                # Try new format first (label field)
                if "label" in node:
                    entity_name = normalize_entity_name(node["label"])
                    entity_type = node.get("type", "unknown")
                    entities.add((entity_name, entity_type))
                # Fallback to legacy format (content field)
                elif "content" in node:
                    entity_name = normalize_entity_name(node["content"])
                    entity_type = node.get("type", "unknown")
                    entities.add((entity_name, entity_type))

        return entities

    def extract_relationships_from_gt(self, gt_data: Dict) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Extract relationships from ground truth data
        """
        relationships = {"assignments": [], "associations": [], "prohibitions": []}

        # Helper to process relationship dicts
        def add_rels(data_key, rel_type_default, category):
            if data_key in gt_data:
                data_val = gt_data[data_key]
                # Handle both dict (keys are IDs) and list formats
                items = data_val.values() if isinstance(data_val, dict) else data_val
                for item in items:
                    if isinstance(item, dict) and "from" in item and "to" in item:
                        f = normalize_entity_name(item["from"])
                        t = normalize_entity_name(item["to"])
                        weight = item.get("weight", [])
                        rt = rel_type_default if not weight else f"{rel_type_default}_{'_'.join(weight)}"
                        relationships[category].append((f, rt, t))

        add_rels("assignments", "assign", "assignments")
        add_rels("associations", "permit", "associations")
        add_rels("prohibitions", "prohibit", "prohibitions")

        # Process direct edges
        if "edges" in gt_data:
            for edge in gt_data["edges"]:
                f = normalize_entity_name(edge.get("from_entity") or edge.get("source_name"))
                t = normalize_entity_name(edge.get("to_entity") or edge.get("target_name"))
                rt = (edge.get("relationship") or edge.get("relation_type", "")).lower()
                if f and t and rt:
                    relationships[self._categorize_relationship_type(rt)].append((f, rt, t))

        return relationships

    def extract_relationships_from_pred(self, pred_data: Dict) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Extract relationships from prediction data

        Prediction format: edges with (source_name, relationship, target_name)

        Returns:
            Dictionary with keys: 'assignments', 'associations', 'prohibitions'
        """
        relationships = {
            "assignments": [],
            "associations": [],
            "prohibitions": []
        }

        # Direct relationship extraction from edges
        if "edges" in pred_data:
            for edge in pred_data["edges"]:
                # Check for relationship field (legacy) or relation_type field (new)
                rel_field = None
                if "relationship" in edge:
                    rel_field = "relationship"
                elif "relation_type" in edge:
                    rel_field = "relation_type"

                if "source_name" in edge and rel_field and "target_name" in edge:
                    from_entity = normalize_entity_name(edge["source_name"])
                    to_entity = normalize_entity_name(edge["target_name"])
                    rel_type = edge[rel_field].lower()

                    # Categorize relationship type
                    category = self._categorize_relationship_type(rel_type)
                    relationships[category].append((from_entity, rel_type, to_entity))

        return relationships

    def calculate_entity_metrics(self, gt_entities: Set[Tuple[str, str]], pred_entities: Set[Tuple[str, str]]) -> Tuple[float, float, float, int, int, int]:
        """
        Calculate precision, recall, and F1 for entity extraction

        Returns:
            (precision, recall, f1, tp, fp, fn) tuple
        """
        # For strict mode, require exact match of both name and type
        # For relaxed mode, allow type mismatches but require name match

        if self.strict_mode:
            true_pos = len(gt_entities & pred_entities)
            false_pos = len(pred_entities - gt_entities)
            false_neg = len(gt_entities - pred_entities)
        else:
            # Relaxed: match by name only
            gt_names = {name for name, _ in gt_entities}
            pred_names = {name for name, _ in pred_entities}

            true_pos = len(gt_names & pred_names)
            false_pos = len(pred_names - gt_names)
            false_neg = len(gt_names - pred_names)

        if true_pos + false_pos == 0:
            precision = 0.0
        else:
            precision = true_pos / (true_pos + false_pos)

        if true_pos + false_neg == 0:
            recall = 0.0
        else:
            recall = true_pos / (true_pos + false_neg)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1, true_pos, false_pos, false_neg

    def strict_eval(self, gold: List[Tuple[str, str, str]], pred: List[Tuple[str, str, str]]) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        """
        Strict evaluation: exact matches only
        
        A predicted relationship is a True Positive if it exactly matches a gold relationship.
        False Positives are predicted relationships not in gold.
        False Negatives are gold relationships not in predictions.
        
        Note: Relationship tuples are (from_entity, relation_type, to_entity).
        All three components must match exactly for a True Positive.
        
        Returns:
            true_pos, false_pos, false_neg lists
        """
        # Convert to sets for efficient lookup
        gold_set = set(gold)
        pred_set = set(pred)
        
        # True positives: predictions that exist in gold
        true_pos = [p for p in pred if p in gold_set]
        
        # False positives: predictions not in gold
        false_pos = [p for p in pred if p not in gold_set]
        
        # False negatives: gold items not in predictions
        false_neg = [g for g in gold if g not in pred_set]
        
        return true_pos, false_pos, false_neg

    def calculate_relationship_metrics(self, gt_rels: List[Tuple[str, str, str]], pred_rels: List[Tuple[str, str, str]]) -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F1 for relationship extraction
        """
        true_pos, false_pos, false_neg = self.strict_eval(gt_rels, pred_rels)

        if len(true_pos) + len(false_neg) == 0:
            precision, recall, f1 = 0.0, 0.0, 0.0
        elif len(true_pos) + len(false_pos) == 0:
            precision, recall, f1 = 0.0, 0.0, 0.0
        else:
            precision = len(true_pos) / (len(true_pos) + len(false_pos))
            recall = len(true_pos) / (len(true_pos) + len(false_neg))
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0

        return precision, recall, f1

    def load_relation_triples(self, triples_path: str) -> List[Dict]:
        """
        Load relation triples dataset for binary classification evaluation

        Args:
            triples_path: Path to relation_triple.json file

        Returns:
            List of relation triple dictionaries
        """
        with open(triples_path, 'r') as f:
            return json.load(f)

    def evaluate_binary_relation_predictions(self, predictions: List[Dict], triples_path: str) -> Dict[str, Any]:
        """
        Evaluate binary relation classification predictions against balanced dataset

        Args:
            predictions: List of prediction dictionaries with keys:
                        'entity1', 'entity2', 'relation', 'exists' (predicted Yes/No)
            triples_path: Path to ground truth relation_triple.json

        Returns:
            Dictionary with binary classification metrics
        """
        # Load ground truth triples
        gt_triples = self.load_relation_triples(triples_path)

        # Create lookup dictionary for ground truth (with normalized entity names)
        gt_lookup = {}
        for triple in gt_triples:
            # Normalize entity names for consistent matching
            from_entity = normalize_entity_name(triple['from_entity'])
            to_entity = normalize_entity_name(triple['to_entity'])
            relationship = triple['relationship'].lower().strip()  # Normalize relationship too
            key = (from_entity, to_entity, relationship)
            gt_lookup[key] = triple['expected_result']  # "Yes" or "No"

        # Initialize confusion matrix
        tp = tn = fp = fn = 0

        # Evaluate each prediction
        for pred in predictions:
            # Normalize entity names and relationship for consistent matching
            entity1 = normalize_entity_name(pred['entity1'])
            entity2 = normalize_entity_name(pred['entity2'])
            relation = pred['relation'].lower().strip()
            pred_key = (entity1, entity2, relation)
            pred_result = pred['exists']  # "Yes" or "No"

            # Get ground truth
            gt_result = gt_lookup.get(pred_key)

            if gt_result is None:
                # Prediction for triple not in ground truth dataset
                continue

            # Update confusion matrix
            if pred_result == "Yes" and gt_result == "Yes":
                tp += 1
            elif pred_result == "No" and gt_result == "No":
                tn += 1
            elif pred_result == "Yes" and gt_result == "No":
                fp += 1
            elif pred_result == "No" and gt_result == "Yes":
                fn += 1

        # Calculate metrics
        total_predictions = tp + tn + fp + fn

        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1 = 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Accuracy = (TP + TN) / Total
        accuracy = (tp + tn) / total_predictions if total_predictions > 0 else 0.0

        return {
            "confusion_matrix": {
                "tp": tp,  # True Positives
                "tn": tn,  # True Negatives
                "fp": fp,  # False Positives
                "fn": fn   # False Negatives
            },
            "metrics": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy
            },
            "counts": {
                "total_predictions": total_predictions,
                "total_ground_truth": len(gt_triples),
                "matched_predictions": total_predictions
            },
            "evaluation_timestamp": datetime.now().isoformat(),
            "evaluation_type": "binary_relation_classification"
        }

    def evaluate_single_graph(self, gt_data: Dict, pred_data: Dict) -> Dict[str, Any]:
        """
        Evaluate a single predicted graph against ground truth
        Following chemoTimelinesEval approach with detailed metrics

        Returns:
            Dictionary with evaluation results
        """
        results = {}

        # Extract entities
        gt_entities = self.extract_entities_from_gt(gt_data)
        pred_entities = self.extract_entities_from_pred(pred_data)

        # Calculate entity metrics (always use relaxed mode - match by name only, ignore task_type)
        original_strict_mode = self.strict_mode
        self.strict_mode = False  # Force relaxed mode for entities to ignore task_type
        entity_precision, entity_recall, entity_f1, entity_tp, entity_fp, entity_fn = self.calculate_entity_metrics(gt_entities, pred_entities)
        self.strict_mode = original_strict_mode  # Restore original mode
        results["entity_precision"] = entity_precision
        results["entity_recall"] = entity_recall
        results["entity_f1"] = entity_f1
        results["entity_tp"] = entity_tp
        results["entity_fp"] = entity_fp
        results["entity_fn"] = entity_fn

        # Extract relationships
        gt_relationships = self.extract_relationships_from_gt(gt_data)
        pred_relationships = self.extract_relationships_from_pred(pred_data)

        # Calculate relationship metrics for each type
        all_true_pos = {}
        all_false_pos = {}
        all_false_neg = {}
        local_f1 = {}
        local_precision = {}
        local_recall = {}

        for rel_type in ["assignments", "associations", "prohibitions"]:
            gt_rels = gt_relationships[rel_type]
            pred_rels = pred_relationships[rel_type]

            # Get detailed counts following chemoTimelinesEval
            true_pos, false_pos, false_neg = self.strict_eval(gt_rels, pred_rels)
            all_true_pos[rel_type] = len(true_pos)
            all_false_pos[rel_type] = len(false_pos)
            all_false_neg[rel_type] = len(false_neg)
            
            # Store detailed lists for error analysis
            results[f"{rel_type}_true_pos_list"] = true_pos
            results[f"{rel_type}_false_pos_list"] = false_pos
            results[f"{rel_type}_false_neg_list"] = false_neg

            # Calculate precision, recall, f1
            if len(true_pos) + len(false_neg) == 0:
                precision, recall, f1 = 0.0, 0.0, 0.0
            elif len(true_pos) + len(false_pos) == 0:
                precision, recall, f1 = 0.0, 0.0, 0.0
            else:
                precision = len(true_pos) / (len(true_pos) + len(false_pos))
                recall = len(true_pos) / (len(true_pos) + len(false_neg))
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0

            results[f"{rel_type}_precision"] = precision
            results[f"{rel_type}_recall"] = recall
            results[f"{rel_type}_f1"] = f1
            results[f"{rel_type}_tp"] = len(true_pos)
            results[f"{rel_type}_fp"] = len(false_pos)
            results[f"{rel_type}_fn"] = len(false_neg)

            local_precision[rel_type] = precision
            local_recall[rel_type] = recall
            local_f1[rel_type] = f1

        # Micro-averaged metrics across all relationship types
        total_tp = sum(all_true_pos.values())
        total_fp = sum(all_false_pos.values())
        total_fn = sum(all_false_neg.values())

        if total_tp + total_fp == 0:
            results["micro_precision"] = 0.0
        else:
            results["micro_precision"] = total_tp / (total_tp + total_fp)

        if total_tp + total_fn == 0:
            results["micro_recall"] = 0.0
        else:
            results["micro_recall"] = total_tp / (total_tp + total_fn)

        if results["micro_precision"] + results["micro_recall"] == 0:
            results["micro_f1"] = 0.0
        else:
            results["micro_f1"] = 2 * (results["micro_precision"] * results["micro_recall"]) / (results["micro_precision"] + results["micro_recall"])

        # Macro-averaged metrics
        if local_f1:
            results["macro_f1"] = sum(local_f1.values()) / len(local_f1)
            results["macro_precision"] = sum(local_precision.values()) / len(local_precision)
            results["macro_recall"] = sum(local_recall.values()) / len(local_recall)
        else:
            results["macro_f1"] = 0.0
            results["macro_precision"] = 0.0
            results["macro_recall"] = 0.0

        # Overall metrics (use micro F1 as primary)
        results["overall_precision"] = results["micro_precision"]
        results["overall_recall"] = results["micro_recall"]
        results["overall_f1"] = results["micro_f1"]

        # Metadata
        results["gt_entities_count"] = len(gt_entities)
        results["pred_entities_count"] = len(pred_entities)
        results["gt_relationships_count"] = sum(len(gt_relationships[rel_type]) for rel_type in ["assignments", "associations", "prohibitions"])
        results["pred_relationships_count"] = sum(len(pred_relationships[rel_type]) for rel_type in ["assignments", "associations", "prohibitions"])
        results["evaluation_timestamp"] = datetime.now().isoformat()
        results["evaluation_mode"] = "strict" if self.strict_mode else "relaxed"

        return results


def load_graph_data(data_path: str) -> Dict[str, Dict]:
    """
    Load graph data from JSON file

    Args:
        data_path: Path to JSON file containing graph data

    Returns:
        Dictionary mapping graph IDs to graph data
    """
    with open(data_path, "r") as f:
        data = json.load(f)

    # If it's a ground truth file with policy_elements, assignments, etc.
    if isinstance(data, dict) and "policy_elements" in data:
        return {"ground_truth_graph": data}

    # If it's a single graph, wrap it
    if isinstance(data, dict) and ("nodes" in data or "entities" in data):
        return {"predicted_graph": data}

    # If it's already a dictionary of graphs
    return data


def evaluate_binary_relation_predictions(predictions_path: str, triples_path: str) -> Dict[str, Any]:
    """
    Evaluate binary relation classification predictions against balanced dataset

    Args:
        predictions_path: Path to JSON file containing predictions
        triples_path: Path to ground truth relation_triple.json

    Returns:
        Dictionary with binary classification evaluation results
    """
    evaluator = KnowledgeGraphEvaluator()

    # Load predictions
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)

    # If predictions is a single dict, wrap it in a list
    if isinstance(predictions, dict):
        predictions = [predictions]

    return evaluator.evaluate_binary_relation_predictions(predictions, triples_path)


def evaluate_predictions(gt_path: str, pred_path: str, strict: bool = True) -> Dict[str, Any]:
    """
    Evaluate predictions against ground truth

    Args:
        gt_path: Path to ground truth JSON file
        pred_path: Path to predictions JSON file
        strict: Whether to use strict evaluation

    Returns:
        Dictionary with evaluation results
    """
    evaluator = KnowledgeGraphEvaluator()
    evaluator.strict_mode = strict
    evaluator.relaxed_mode = not strict

    # Load data
    gt_data = load_graph_data(gt_path)
    pred_data = load_graph_data(pred_path)

    # Simple single graph evaluation (most common case)
    gt_graph = None
    pred_graph = None

    # Find ground truth graph
    for graph_data in gt_data.values():
        if isinstance(graph_data, dict) and ("nodes" in graph_data or "policy_elements" in graph_data):
            gt_graph = graph_data
            break

    # Find prediction graph
    for graph_data in pred_data.values():
        if isinstance(graph_data, dict) and ("nodes" in graph_data or "entities" in graph_data):
            pred_graph = graph_data
            break

    results = {"individual_results": {}, "total_graphs_evaluated": 0}

    if gt_graph and pred_graph:
        graph_result = evaluator.evaluate_single_graph(gt_graph, pred_graph)
        graph_id = "graph_001"
        results["individual_results"][graph_id] = graph_result
        results["total_graphs_evaluated"] = 1

        # Use single graph results as overall results
        results.update({
            "micro_f1": graph_result["micro_f1"],
            "micro_precision": graph_result["micro_precision"],
            "micro_recall": graph_result["micro_recall"],
            "macro_f1": graph_result["macro_f1"],
            "macro_precision": graph_result["macro_precision"],
            "macro_recall": graph_result["macro_recall"],
            "overall_f1": graph_result["overall_f1"],
            "total_tp": graph_result["assignments_tp"] + graph_result["associations_tp"] + graph_result["prohibitions_tp"],
            "total_fp": graph_result["assignments_fp"] + graph_result["associations_fp"] + graph_result["prohibitions_fp"],
            "total_fn": graph_result["assignments_fn"] + graph_result["associations_fn"] + graph_result["prohibitions_fn"],
            "entity_tp": graph_result["entity_tp"],
            "entity_fp": graph_result["entity_fp"],
            "entity_fn": graph_result["entity_fn"],
        })
    else:
        # Set default values when no graphs found
        default_keys = ["micro_f1", "micro_precision", "micro_recall", "macro_f1", 
                        "macro_precision", "macro_recall", "overall_f1"]
        for key in default_keys:
            results[key] = 0.0
        results["total_tp"] = results["total_fp"] = results["total_fn"] = 0
        results["entity_tp"] = results["entity_fp"] = results["entity_fn"] = 0

    results["evaluation_mode"] = "strict" if strict else "relaxed"
    return results


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate Knowledge Graph predictions or Binary Relation Classification against ground truth"
    )

    parser.add_argument(
        "--gt_path",
        help="Path to ground truth JSON file (for graph evaluation)"
    )
    parser.add_argument(
        "--pred_path",
        help="Path to predictions JSON file (for graph evaluation)"
    )
    parser.add_argument(
        "--binary_predictions",
        help="Path to predictions JSON file (for binary relation classification)"
    )
    parser.add_argument(
        "--triples_path", default="data/GroundTruthGraphsJSON/relation_triple.json",
        help="Path to relation triples JSON file (default: data/GroundTruthGraphsJSON/relation_triple.json)"
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Use strict evaluation (exact matches only)"
    )
    parser.add_argument(
        "--relaxed", action="store_true",
        help="Use relaxed evaluation (allow partial matches)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for results (default: print to stdout)"
    )

    args = parser.parse_args()

    # Check evaluation mode
    graph_eval = args.gt_path and args.pred_path
    binary_eval = args.binary_predictions

    if graph_eval and binary_eval:
        parser.error("Cannot specify both graph evaluation (--gt_path, --pred_path) and binary evaluation (--binary_predictions)")
    if not graph_eval and not binary_eval:
        parser.error("Must specify either graph evaluation (--gt_path and --pred_path) or binary evaluation (--binary_predictions)")

    if graph_eval:
        if args.strict and args.relaxed:
            parser.error("Cannot specify both --strict and --relaxed")
        if not args.strict and not args.relaxed:
            args.strict = True  # Default to strict

    if graph_eval:
        print(f"Knowledge Graph Evaluation")
        print(f"Ground truth: {args.gt_path}")
        print(f"Predictions: {args.pred_path}")
        print(f"Evaluation mode: {'strict' if args.strict else 'relaxed'}")
        print()

        # Run graph evaluation
        results = evaluate_predictions(args.gt_path, args.pred_path, args.strict)
    else:
        print(f"Binary Relation Classification Evaluation")
        print(f"Predictions: {args.binary_predictions}")
        print(f"Ground truth triples: {args.triples_path}")
        print()

        # Run binary evaluation
        results = evaluate_binary_relation_predictions(args.binary_predictions, args.triples_path)

    # Print results
    print("=== EVALUATION RESULTS ===")

    if graph_eval:
        print(f"Total graphs evaluated: {results['total_graphs_evaluated']}")
        print(f"Evaluation mode: {results['evaluation_mode']}")
        print()

        # Overall metrics
        print("Overall Metrics:")
        print(f"  Micro F1: {results['micro_f1']:.4f}")
        print(f"  Macro F1: {results['macro_f1']:.4f}")
        print(f"  Micro Precision: {results['micro_precision']:.4f}")
        print(f"  Micro Recall: {results['micro_recall']:.4f}")
        print()

        # Counts
        print("Counts:")
        print(f"  Entity TP: {results.get('entity_tp', 0)}, FP: {results.get('entity_fp', 0)}, FN: {results.get('entity_fn', 0)}")
        print(f"  Relationship TP: {results['total_tp']}, FP: {results['total_fp']}, FN: {results['total_fn']}")
        print()
    else:
        print(f"Evaluation type: {results['evaluation_type']}")
        print(f"Total predictions evaluated: {results['counts']['total_predictions']}")
        print(f"Ground truth triples: {results['counts']['total_ground_truth']}")
        print()

        # Confusion matrix
        cm = results['confusion_matrix']
        print("Confusion Matrix:")
        print(f"  True Positives (TP): {cm['tp']}  - Predicted Yes, Ground Truth Yes")
        print(f"  True Negatives (TN): {cm['tn']}  - Predicted No, Ground Truth No")
        print(f"  False Positives (FP): {cm['fp']} - Predicted Yes, Ground Truth No")
        print(f"  False Negatives (FN): {cm['fn']} - Predicted No, Ground Truth Yes")
        print()

        # Binary classification metrics
        metrics = results['metrics']
        print("Binary Classification Metrics:")
        print(f"  Precision: {metrics['precision']:.4f}  (TP / (TP + FP))")
        print(f"  Recall:    {metrics['recall']:.4f}     (TP / (TP + FN))")
        print(f"  F1 Score:  {metrics['f1']:.4f}         (2 * P * R / (P + R))")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}   ((TP + TN) / Total)")
        print()

    if graph_eval and results.get('individual_results'):
        print("Individual Graph Results:")
        for graph_id, graph_result in results['individual_results'].items():
            print(f"  {graph_id}:")
            entity_tp = graph_result.get("entity_tp", 0)
            entity_fp = graph_result.get("entity_fp", 0)
            entity_fn = graph_result.get("entity_fn", 0)
            print(f"    Entity F1: {graph_result['entity_f1']:.4f}, TP={entity_tp}, FP={entity_fp}, FN={entity_fn}")
            print(f"    Micro F1: {graph_result['micro_f1']:.4f}")
            print(f"    Macro F1: {graph_result['macro_f1']:.4f}")

            # Show relationship type metrics
            for rel_type in ["assignments", "associations", "prohibitions"]:
                if f"{rel_type}_f1" in graph_result and graph_result[f"{rel_type}_f1"] > 0:
                    tp = graph_result.get(f"{rel_type}_tp", 0)
                    fp = graph_result.get(f"{rel_type}_fp", 0)
                    fn = graph_result.get(f"{rel_type}_fn", 0)
                    print(f"    {rel_type}: F1={graph_result[f'{rel_type}_f1']:.4f}, TP={tp}, FP={fp}, FN={fn}")
            print()

    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

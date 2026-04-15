"""Prompt engineering for entity extraction, relation classification, and path generation."""

import json
from collections import deque
from typing import Dict, List, Optional, Any, Tuple

from . import file_utils
from .file_utils import encode_image_to_base64

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# Few-shot example paths (relative to project root)
from .config import PROJECT_ROOT
FEW_SHOT_DIR = str(PROJECT_ROOT / "data" / "datasets" / "one-shot")
FEW_SHOT_BASE = "enterprise_clients_graph_policies_graph_part1__association_client_organization_b_to_application_services"
FEW_SHOT_JSON_PATH = str(PROJECT_ROOT / "data" / "datasets" / "one-shot" / f"{FEW_SHOT_BASE}.json")
FEW_SHOT_IMAGE_PATH = str(PROJECT_ROOT / "data" / "datasets" / "one-shot" / f"{FEW_SHOT_BASE}_labeled.png")
FEW_SHOT_IMAGE_B_PATH = str(PROJECT_ROOT / "data" / "datasets" / "one-shot" / f"{FEW_SHOT_BASE}_labeled_b.png")

# Node types Reference
NODE_TYPES = {
    "user_attributes": "blue/cyan nodes are characterized by job roles or titles, common names for user groups, or organizational units",
    "object_attributes": "green nodes representing resources, data, systems, or infrastructure subject to the control of users through user_attributes",
    "policy_classes": "red/orange represents node reachable by all other nodes (incoming edges only, no outgoing edges)"
}

# Relationship types and their visual characteristics
RELATIONSHIP_TYPES = {
    "assign": "solid black arrows (→) for hierarchical assignment. Can connect: user_attr ↔ user_attr, object_attr ↔ object_attr, or any → policy_class",
    "permit": "green arrows (→) for permission/association. ONLY from user_attributes to object_attributes",
    "prohibit": "red arrows (⇢) for prohibition/denial. ONLY from user_attributes to object_attributes"
}

# System prompt for vision tasks
SYSTEM_PROMPT_VISION = """You are an expert at analyzing images and understanding visual diagrams.
You can see and interpret images, graphs, charts, and visual representations.
You provide accurate, detailed analysis based on what you observe in the images."""

# ============================================================================
# PROMPT BUILDING BLOCKS
# ============================================================================

NODE_TYPE_INSTRUCTIONS = """Node Types (by background color):
- Blue/Cyan: "user_attributes" (roles, user groups, organizational units)
- Green: "object_attributes" (resources, systems, data)
- Red/Orange: "policy_classes" (sinks, no outgoing edges)"""

RELATIONSHIP_TYPE_INSTRUCTIONS = """Relation Types (by arrow style):
- "assign": Solid black arrow (→) - hierarchical assignment
- "permit": Solid green arrow (→) - permission/association
- "prohibit": Dashed red arrow (⇢) - prohibition/denial"""

ARROW_DIRECTION_INSTRUCTIONS = """Arrow Direction (CRITICAL):
- The arrowhead (▶) ALWAYS points to the TARGET node.
- The other end of the line is the SOURCE node.
- NEVER reverse direction - source is where arrow starts, target is where arrowhead points.
- Only consider DIRECT arrows (no intermediate nodes)."""

# Used only in path_generation template
ENTITY_LABEL_TIPS = """
Entity Labels (IMPORTANT):
- Copy entity labels EXACTLY as shown in the image.
- Common correct names: cloudshield (not cloudsited), enterprise_clients (not enterprises_clients), microservices (not microsservices), ai_machine_learning_models (not a_machine_learning_models).
- Check spelling carefully. Use underscores between words (e.g., cloudshield_users not cloudshield users)."""

OUTPUT_FORMAT_ENTITY = '''Output Format (Strict JSON only):
{{
  "nodes": [
    {{"node_id": "n1", "type": "user_attributes", "content": "label_name"}},
    {{"node_id": "n2", "type": "object_attributes", "content": "label_name"}},
    {{"node_id": "n3", "type": "policy_classes", "content": "label_name"}}
    ...
  ]
}}'''

OUTPUT_FORMAT_RELATION = '''Output Format (Strict JSON only):
{{
  "entity1": "from_entity",
  "entity1_type": "user_attributes" | "object_attributes" | "policy_classes",
  "entity2": "to_entity",
  "entity2_type": "user_attributes" | "object_attributes" | "policy_classes",
  "relation": "relation_type",
  "exists": "Yes" or "No",
  "confidence": "high", "medium", or "low",
  "explanation": "Brief visual evidence from the image (specify arrow color/style if found)",
  "subrelations": ["action1", "action2"]
}}'''

OUTPUT_FORMAT_PATH = '''Output Format (Strict JSON only):
{{
  "nodes": [
    {{"node_id": "n1", "type": "user_attributes", "content": "exact_label"}},
    {{"node_id": "n2", "type": "object_attributes", "content": "exact_label"}},
    {{"node_id": "n3", "type": "policy_classes", "content": "exact_label"}}
  ],
  "edges": [
    {{
      "from_id": "n1",
      "source_name": "from_label",
      "from_type": "user_attributes",
      "to_id": "n2",
      "target_name": "to_label",
      "to_type": "object_attributes",
      "relationship": "assign",
      "subrelations": []
    }}
  ]
}}'''

# ============================================================================
# PROMPT REGISTRY
# ============================================================================


class PromptRegistry:
    """Registry for prompt templates, enabling easy access and extension."""

    _templates: Dict[str, str] = {}

    @classmethod
    def register(cls, name: str, template: str) -> None:
        """Register a prompt template."""
        cls._templates[name] = template

    @classmethod
    def get(cls, name: str, **kwargs) -> str:
        """Get a prompt template and format it with provided arguments."""
        if name not in cls._templates:
            raise ValueError(f"Prompt template '{name}' not found")
        return cls._templates[name].format(**kwargs)

    @classmethod
    def list_templates(cls) -> List[str]:
        """List all registered template names."""
        return list(cls._templates.keys())


# Register prompt templates
PromptRegistry.register("entity_extraction", f"""Extract ALL entities from the Access Control DAG image.

Task:
1. Find every rectangular box with text.
2. Assign sequential IDs (n1, n2, n3, ...).
3. Classify each node:
""" + NODE_TYPE_INSTRUCTIONS + f"""
4. Extract the exact text label (no normalization).

{OUTPUT_FORMAT_ENTITY}""")

PromptRegistry.register("relation_classification", """Check if a direct "{relation_type}" arrow exists from "{from_entity}" to "{to_entity}".

""" + NODE_TYPE_INSTRUCTIONS + """

""" + RELATIONSHIP_TYPE_INSTRUCTIONS + """

""" + ARROW_DIRECTION_INSTRUCTIONS + """

Decision:
- Answer "Yes" ONLY if a direct arrow of the correct type points from the source to the target.
- Answer "No" if no direct arrow exists, the direction is reversed, or the style is wrong.

""" + OUTPUT_FORMAT_RELATION)

PromptRegistry.register("path_generation", """Extract all nodes and edges from this Access Control DAG image.

## Approach (work through systematically)
1. Identify ALL nodes: scan the entire image, list every rectangular box with text (typical graphs have 20-40+ nodes)
2. For each node, classify by background color: blue/cyan=user_attributes, green=object_attributes, red/orange=policy_classes
3. Trace EVERY arrow: black solid=assign, green=permit, red dashed=prohibit
4. For each arrow: identify source (line start) and target (arrowhead) - do not reverse direction
5. Output the complete JSON - do not truncate or skip any nodes/edges

""" + NODE_TYPE_INSTRUCTIONS + """

""" + RELATIONSHIP_TYPE_INSTRUCTIONS + """

""" + ARROW_DIRECTION_INSTRUCTIONS + """

Exhaustive Extraction Instructions:
- Extract ALL nodes. Do not stop until every rectangular box in the image is listed.
- Extract ALL edges. Trace every arrow. If you see an arrow, it must appear in the edges list.
- Typical graphs have 30-40+ entities and similar number of relations.
- For large or dense graphs: output valid JSON with as many nodes and edges as you can; partial output is acceptable. Do not return empty—always return at least nodes and any edges you can identify.

""" + ENTITY_LABEL_TIPS + """

Validation Checklist (before outputting):
1. permit/prohibit edges MUST have user_attributes as source and object_attributes as target
2. Arrow direction matches: source → target (arrowhead on target)
3. Entity names match image labels exactly (no typos)
4. Relation types follow the constraints above
5. Node count matches visible boxes (do not skip any)
6. Every arrow has a corresponding edge (do not miss any)

""" + OUTPUT_FORMAT_PATH + """

Return ONLY the JSON object.""")

# ============================================================================
# MESSAGE BUILDERS
# ============================================================================


def create_vision_message(text_content: str, image_base64: str, image_detail: str = "low") -> Dict[str, Any]:
    """
    Create a vision message with text and image content.

    Args:
        text_content: The text prompt to include
        image_base64: Base64 encoded image data
        image_detail: Image detail level - "low" (default, cost-efficient) or "high" (higher quality, more expensive)

    Returns:
        Message dictionary with text and image content
    """
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text_content},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}", "detail": image_detail}}
        ]
    }


def build_message_sequence(prompt_text: str, image_base64: str, few_shot_examples: Optional[List[Dict]] = None, image_detail: str = "low") -> List[Dict]:
    """
    Build a complete message sequence for vision tasks.

    Args:
        prompt_text: The main task prompt
        image_base64: Base64 encoded image
        few_shot_examples: Optional few-shot examples to include
        image_detail: Image detail level - "low" (default, cost-efficient) or "high" (higher quality, more expensive)

    Returns:
        Complete message sequence including system prompt, examples, and task
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT_VISION}]

    # Add few-shot examples if provided
    if few_shot_examples:
        messages.extend(few_shot_examples)

    # Add the actual task
    messages.append(create_vision_message(prompt_text, image_base64, image_detail))

    return messages


def get_entity_extraction_messages(image_base64: str, filename: str, few_shot_examples: Optional[List[Dict]] = None, image_detail: str = "low") -> List[Dict]:
    """
    Generate messages for entity extraction from Access Control DAG images.

    Args:
        image_base64: Base64 encoded image data
        filename: Image filename (unused in current implementation)
        few_shot_examples: Optional few-shot example messages for Context7 sequential processing
        image_detail: Image detail level - "low" (default, cost-efficient) or "high" (higher quality, more expensive)

    Returns:
        Complete message sequence for entity extraction task
    """
    prompt_text = PromptRegistry.get("entity_extraction")

    return build_message_sequence(prompt_text, image_base64, few_shot_examples, image_detail)


def get_relation_classification_messages(
    image_base64: str,
    triple_data: Dict[str, str],
    few_shot_examples: Optional[List[Dict]] = None,
    image_detail: str = "low"
) -> List[Dict]:
    """
    Generate messages for binary relation classification between two entities.

    Args:
        image_base64: Base64 encoded image data
        triple_data: Dictionary with 'from_entity', 'to_entity', and 'relationship' keys
        few_shot_examples: Optional few-shot example messages for Context7 sequential processing
        image_detail: Image detail level - "low" (default, cost-efficient) or "high" (higher quality, more expensive)

    Returns:
        Complete message sequence for relation classification task
    """
    from_entity = triple_data['from_entity']
    to_entity = triple_data['to_entity']
    relation_type = triple_data.get('relationship', 'assign')

    prompt = PromptRegistry.get("relation_classification",
                               from_entity=from_entity,
                               to_entity=to_entity,
                               relation_type=relation_type)

    return build_message_sequence(prompt, image_base64, few_shot_examples, image_detail)


def get_path_generation_messages(image_base64: str, filename: str, few_shot_examples: Optional[List[Dict]] = None, image_detail: str = "low") -> List[Dict]:
    """
    Generate messages for path generation and graph extraction from Access Control DAG images.

    Args:
        image_base64: Base64 encoded image data
        filename: Image filename (unused in current implementation)
        few_shot_examples: Optional few-shot example messages for Context7 sequential processing
        image_detail: Image detail level - "low" (default, cost-efficient) or "high" (higher quality, more expensive)

    Returns:
        Complete message sequence for path generation task
    """
    prompt_text = PromptRegistry.get("path_generation")

    return build_message_sequence(prompt_text, image_base64, few_shot_examples, image_detail)


# ============================================================================
# GRAPH / PATH HELPERS
# ============================================================================


def find_path(start_node_id: str, end_node_id: str, edges: List[Dict], entity_to_node_id: Dict) -> Optional[List[str]]:
    """
    Find a path between two nodes using BFS (Breadth-First Search).

    Args:
        start_node_id: Starting node ID
        end_node_id: Ending node ID
        edges: List of edge dictionaries with 'from_id' and 'to_id' keys
        entity_to_node_id: Mapping of entity names to node IDs (unused in current implementation)

    Returns:
        List of node IDs representing the path from start to end, or None if no path exists
    """
    # Build adjacency list from edges
    adj = {}
    for edge in edges:
        from_id = edge["from_id"]
        to_id = edge["to_id"]
        if from_id not in adj:
            adj[from_id] = []
        adj[from_id].append(to_id)

    # BFS to find path
    queue = deque([(start_node_id, [start_node_id])])
    visited = {start_node_id}

    while queue:
        current, path = queue.popleft()

        if current == end_node_id:
            return path

        if current in adj:
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

    return None


def generate_negative_examples(positives: List[Dict], entities: set, max_per_type: int = 10) -> List[Dict]:
    """
    Generate negative examples for each relation type that has positive examples.

    Args:
        positives: List of positive triple dictionaries
        entities: Set of all entity names
        max_per_type: Maximum negative examples per relation type

    Returns:
        List of negative triple dictionaries
    """
    # Get relation types that have positive examples
    relation_types_with_positives = set(p["relation"] for p in positives)
    pos_pairs = {(p["from_entity"], p["to_entity"]) for p in positives}

    negatives = []
    ent_list = list(entities)

    for relation_type in relation_types_with_positives:
        negatives_for_type = 0
        for from_entity in ent_list:
            for to_entity in ent_list:
                if from_entity != to_entity and (from_entity, to_entity) not in pos_pairs:
                    negatives.append({
                        "from_entity": from_entity,
                        "to_entity": to_entity,
                        "relation": relation_type
                    })
                    negatives_for_type += 1
                    if negatives_for_type >= max_per_type:
                        break
            if negatives_for_type >= max_per_type:
                break

    return negatives


def select_representative_examples(positives: List[Dict], negatives: List[Dict], max_types: int = 2) -> Tuple[List[Dict], List[Dict]]:
    """
    Select representative positive and negative examples with relation type balance.
    Prioritizes examples that may demonstrate tricky directionality cases for better
    few-shot learning of arrow direction identification.

    Args:
        positives: List of positive triple dictionaries
        negatives: List of negative triple dictionaries
        max_types: Maximum number of relation types to include

    Returns:
        Tuple of (selected_positives, selected_negatives)
    """
    # Group by relation type
    pos_by_type = {}
    neg_by_type = {}

    for p in positives:
        rel_type = p["relation"]
        pos_by_type.setdefault(rel_type, []).append(p)

    for n in negatives:
        rel_type = n["relation"]
        neg_by_type.setdefault(rel_type, []).append(n)

    # Prioritize getting examples from all relation types (assign, permit, prohibit)
    # instead of just the first max_types alphabetically
    available_types = set(pos_by_type.keys())
    preferred_types = ["assign", "permit", "prohibit"]  # Priority order

    selected_pos = []
    selected_neg = []

    # First, try to get one example from each preferred relation type
    for rel_type in preferred_types:
        if rel_type in pos_by_type and pos_by_type[rel_type]:
            # Prioritize examples that might be directionality-challenging:
            # - Longer entity names (potentially more complex relationships)
            # - Or select a middle example instead of first for variety
            pos_examples = pos_by_type[rel_type]
            if len(pos_examples) > 1:
                # Select example with longest combined entity names for potential complexity
                selected_example = max(pos_examples,
                                     key=lambda x: len(x["from_entity"]) + len(x["to_entity"]))
            else:
                selected_example = pos_examples[0]
            selected_pos.append(selected_example)

        if rel_type in neg_by_type and neg_by_type[rel_type]:
            # For negatives, select examples that might be confusing directionally
            neg_examples = neg_by_type[rel_type]
            if len(neg_examples) > 1:
                # Select a middle example to avoid always picking the first
                selected_neg.append(neg_examples[len(neg_examples)//2])
            else:
                selected_neg.append(neg_examples[0])

    # If we still need more examples, fill with remaining relation types
    remaining_slots = max_types - len(selected_pos)
    if remaining_slots > 0:
        remaining_types = [rt for rt in sorted(pos_by_type.keys()) if rt not in preferred_types][:remaining_slots]
        for rel_type in remaining_types:
            if pos_by_type[rel_type]:
                pos_examples = pos_by_type[rel_type]
                selected_example = pos_examples[0]  # Simple selection for remaining
                selected_pos.append(selected_example)

            if rel_type in neg_by_type and neg_by_type[rel_type]:
                neg_examples = neg_by_type[rel_type]
                selected_neg.append(neg_examples[0] if neg_examples else None)

    return selected_pos, selected_neg


def load_ground_truth(json_path: str) -> Dict:
    """
    Load ground truth data from JSON file.

    Args:
        json_path: Path to the ground truth JSON file

    Returns:
        Parsed JSON data as dictionary
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_positive_triples(ground_truth: Dict) -> List[Dict[str, str]]:
    """
    Extract all positive relationship triples from ground truth data.

    Args:
        ground_truth: Ground truth dictionary with assignments, associations, prohibitions

    Returns:
        List of triple dictionaries with 'from_entity', 'to_entity', 'relation' keys
    """
    positives = []

    # Extract assignments
    if "assignments" in ground_truth:
        assignments_raw = ground_truth["assignments"]
        assignments_list = assignments_raw.values() if isinstance(assignments_raw, dict) else assignments_raw
        for item in assignments_list:
            positives.append({
                "from_entity": item.get("from"),
                "to_entity": item.get("to"),
                "relation": "assign"
            })

    # Extract associations (permit relationships)
    if "associations" in ground_truth:
        associations_raw = ground_truth["associations"]
        associations_list = associations_raw.values() if isinstance(associations_raw, dict) else associations_raw
        for item in associations_list:
            positives.append({
                "from_entity": item.get("from"),
                "to_entity": item.get("to"),
                "relation": "permit"
            })

    # Extract prohibitions
    if "prohibitions" in ground_truth:
        prohibitions_raw = ground_truth["prohibitions"]
        prohibitions_list = prohibitions_raw.values() if isinstance(prohibitions_raw, dict) else prohibitions_raw
        for item in prohibitions_list:
            positives.append({
                "from_entity": item.get("from"),
                "to_entity": item.get("to"),
                "relation": "prohibit"
            })

    return positives


def extract_all_entities(ground_truth: Dict) -> set:
    """
    Extract all unique entity names from ground truth data.

    Args:
        ground_truth: Ground truth dictionary

    Returns:
        Set of all entity names
    """
    entities = set()

    if "policy_elements" in ground_truth:
        pe = ground_truth["policy_elements"]
        for key in ["user_attributes", "object_attributes"]:
            attrs = pe.get(key, [])
            if isinstance(attrs, list):
                entities.update(attrs)

        policy_class = pe.get("policy_classes")
        if isinstance(policy_class, str):
            entities.add(policy_class)
        elif isinstance(policy_class, list):
            entities.update(policy_class)

    return entities


# ============================================================================
# FEW-SHOT EXAMPLE GENERATION
# ============================================================================


def create_message_pair(
    method: str,
    image_base64: str,
    expected_output: str,
    triple_data: Optional[Dict] = None,
    image_detail: str = "low",
) -> List[Dict]:
    """
    Create a user-assistant message pair for few-shot examples.

    Args:
        method: One of "entity_extraction", "relation_classification", "path_generation"
        image_base64: Base64 encoded image
        expected_output: Expected JSON output string
        triple_data: For relation_classification only: dict with from_entity, to_entity, relation
        image_detail: Image detail level - "low" (default) or "high"

    Returns:
        List containing user message and assistant response
    """
    triple_data = triple_data or {}
    if method == "relation_classification":
        prompt = PromptRegistry.get(
            "relation_classification",
            from_entity=triple_data["from_entity"],
            to_entity=triple_data["to_entity"],
            relation_type=triple_data.get("relation", "assign"),
        )
    elif method == "entity_extraction":
        prompt = PromptRegistry.get("entity_extraction")
    elif method == "path_generation":
        prompt = PromptRegistry.get("path_generation")
    else:
        raise ValueError(f"Unknown method for create_message_pair: {method}")

    return [
        create_vision_message(prompt, image_base64, image_detail),
        {"role": "assistant", "content": expected_output},
    ]


def generate_few_shot_examples_for_entity_extraction(
    image_without_labels_path: str = FEW_SHOT_IMAGE_PATH,
    image_with_labels_path: str = FEW_SHOT_IMAGE_B_PATH,
    ground_truth_json_path: str = FEW_SHOT_JSON_PATH
) -> List[Dict]:
    """
    Generate two-shot examples for entity extraction using multiple conversation turns.

    Uses the same ground truth output for both images to demonstrate consistency.

    Args:
        image_without_labels_path: Path to image without label overlays
        image_with_labels_path: Path to image with label overlays
        ground_truth_json_path: Path to ground truth JSON file

    Returns:
        List of message dictionaries for few-shot learning
    """
    # Load and convert ground truth
    ground_truth = load_ground_truth(ground_truth_json_path)
    expected_output = file_utils.convert_ground_truth_to_entity_extraction_format(ground_truth)
    expected_output_json = json.dumps(expected_output, indent=2)

    # Encode images
    image_without_labels_base64 = encode_image_to_base64(image_without_labels_path)
    image_with_labels_base64 = encode_image_to_base64(image_with_labels_path)

    # Create message pairs for each image
    messages = []
    messages.extend(create_message_pair(
        "entity_extraction", image_without_labels_base64, expected_output_json, triple_data={}
    ))
    messages.extend(create_message_pair(
        "entity_extraction", image_with_labels_base64, expected_output_json, triple_data={}
    ))

    return messages


def generate_few_shot_examples_for_relation_classification(
    image_without_labels_path: str = FEW_SHOT_IMAGE_PATH,
    image_with_labels_path: str = FEW_SHOT_IMAGE_B_PATH,
    ground_truth_json_path: str = FEW_SHOT_JSON_PATH,
    example_relations: Optional[List[Dict]] = None
) -> List[Dict]:
    """
    Generate four-shot examples for relation classification (positive and negative examples per image).

    Uses deterministic selection of representative examples for reproducibility.
    Ensures negative examples match the relation types of positive examples.

    Args:
        image_without_labels_path: Path to image without label overlays
        image_with_labels_path: Path to image with label overlays
        ground_truth_json_path: Path to ground truth JSON file
        example_relations: Optional pre-selected relations (unused in current implementation)

    Returns:
        List of message dictionaries for few-shot learning
    """
    # Load ground truth and extract data
    ground_truth = load_ground_truth(ground_truth_json_path)
    positives = extract_positive_triples(ground_truth)
    entities = extract_all_entities(ground_truth)

    if not positives:
        raise ValueError("No positive examples found in ground truth")

    # Generate and select examples
    negatives = generate_negative_examples(positives, entities)
    selected_pos, selected_neg = select_representative_examples(positives, negatives)

    # Fallback handling
    if not selected_pos:
        selected_pos = [positives[0]]
    if not selected_neg:
        # Create fallback negative with same relation type as first positive
        first_pos = selected_pos[0]
        fallback_neg = {
            "from_entity": list(entities)[0] if entities else "unknown_entity",
            "to_entity": list(entities)[1] if len(entities) > 1 else "unknown_entity",
            "relation": first_pos["relation"]
        }
        selected_neg = [fallback_neg]

    # Encode images
    image_without_labels_base64 = encode_image_to_base64(image_without_labels_path)
    image_with_labels_base64 = encode_image_to_base64(image_with_labels_path)

    # Create message scenarios: (image, relation_data) pairs
    scenarios = [
        (image_without_labels_base64, selected_pos[0]),
        (image_without_labels_base64, selected_neg[0]),
        (image_with_labels_base64, selected_pos[1] if len(selected_pos) > 1 else selected_pos[0]),
        (image_with_labels_base64, selected_neg[1] if len(selected_neg) > 1 else selected_neg[0])
    ]

    # Build message sequence
    messages = []
    for img_base64, rel_data in scenarios:
        expected_output = file_utils.convert_ground_truth_to_relation_classification_format(
            ground_truth,
            rel_data["from_entity"],
            rel_data["to_entity"],
            rel_data["relation"]
        )
        expected_output_json = json.dumps(expected_output, indent=2)

        messages.extend(create_message_pair(
            "relation_classification", img_base64, expected_output_json, triple_data=rel_data
        ))

    return messages


def generate_few_shot_examples_for_path_generation(
    image_without_labels_path: str = FEW_SHOT_IMAGE_PATH,
    image_with_labels_path: str = FEW_SHOT_IMAGE_B_PATH,
    ground_truth_json_path: str = FEW_SHOT_JSON_PATH
) -> List[Dict]:
    """
    Generate two-shot examples for path generation using multiple conversation turns.

    Uses the same ground truth output for both images to demonstrate consistency.

    Args:
        image_without_labels_path: Path to image without label overlays
        image_with_labels_path: Path to image with label overlays
        ground_truth_json_path: Path to ground truth JSON file

    Returns:
        List of message dictionaries for few-shot learning
    """
    # Load and convert ground truth
    ground_truth = load_ground_truth(ground_truth_json_path)
    expected_output = file_utils.convert_ground_truth_to_path_generation_format(ground_truth)
    expected_output_json = json.dumps(expected_output, indent=2)

    # Encode images
    image_without_labels_base64 = encode_image_to_base64(image_without_labels_path)
    image_with_labels_base64 = encode_image_to_base64(image_with_labels_path)

    # Create message pairs for each image
    messages = []
    messages.extend(create_message_pair(
        "path_generation", image_without_labels_base64, expected_output_json, triple_data={}
    ))
    messages.extend(create_message_pair(
        "path_generation", image_with_labels_base64, expected_output_json, triple_data={}
    ))

    return messages

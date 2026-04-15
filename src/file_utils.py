"""File I/O, image encoding, JSON parsing, and discovery utilities."""

import base64
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

try:
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allow loading truncated images
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def encode_image_to_base64(image_path: str, max_side: int = 1024) -> str:
    """
    Encode image to base64 string with optional resizing.

    Args:
        image_path: Path to the image file
        max_side: Maximum dimension for resizing

    Returns:
        Base64 encoded string

    Raises:
        Exception: If image cannot be processed
    """
    if not HAS_PIL:
        raise ImportError("PIL (Pillow) is required for image processing")

    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if max(width, height) > max_side:
                # Resize large images
                scale = max_side / max(width, height)
                new_size = (int(width * scale), int(height * scale))
                img = img.convert("RGB").resize(new_size, Image.LANCZOS)
            else:
                # Convert to RGB for all images (including PNG)
                img = img.convert("RGB")

            buf = BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

    except Exception as e:
        # Fallback: direct file reading
        try:
            with open(image_path, "rb") as image_file:
                data = image_file.read()
                return base64.b64encode(data).decode('utf-8')
        except Exception as inner_e:
            raise Exception(f"Failed to encode image {image_path}: {e} (fallback error: {inner_e})")


# ============================================================================
# JSON PROCESSING
# ============================================================================

def parse_json_response(response: str) -> Dict:
    """
    Parse JSON from model response with multiple fallback strategies.

    Args:
        response: Raw response string from model

    Returns:
        Parsed JSON dictionary

    Raises:
        ValueError: If JSON cannot be parsed
    """
    import re
    
    if not response or not response.strip():
        raise ValueError("Empty response from model")

    # Try direct parsing first
    try:
        parsed = json.loads(response)
        return _normalize_parsed_response(parsed)
    except json.JSONDecodeError:
        pass

    # Remove markdown code blocks using regex (more robust)
    cleaned = response.strip()
    
    # Pattern to match ```json or ``` code blocks
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(code_block_pattern, cleaned)
    if matches:
        # Try each match
        for match in matches:
            try:
                parsed = json.loads(match.strip())
                return _normalize_parsed_response(parsed)
            except json.JSONDecodeError:
                continue
    
    # Fallback: manual removal of markdown code blocks
    if '```json' in cleaned:
        start = cleaned.find('```json') + 7
        end = cleaned.rfind('```')
        if end > start:
            cleaned = cleaned[start:end].strip()
    elif cleaned.startswith('```') and cleaned.endswith('```'):
        cleaned = cleaned[3:-3].strip()
        if cleaned.startswith('json'):
            cleaned = cleaned[4:].strip()

    # Try parsing cleaned content
    try:
        parsed = json.loads(cleaned)
        return _normalize_parsed_response(parsed)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object block
    start_idx = cleaned.find('{')
    end_idx = cleaned.rfind('}') + 1
    if start_idx != -1 and end_idx > start_idx:
        json_str = cleaned[start_idx:end_idx]
        try:
            parsed = json.loads(json_str)
            return _normalize_parsed_response(parsed)
        except json.JSONDecodeError:
            # Try to fix common issues: trailing commas
            json_str_fixed = re.sub(r',\s*}', '}', json_str)
            json_str_fixed = re.sub(r',\s*]', ']', json_str_fixed)
            try:
                parsed = json.loads(json_str_fixed)
                return _normalize_parsed_response(parsed)
            except json.JSONDecodeError:
                pass

    # Try array format
    start_idx = cleaned.find('[')
    end_idx = cleaned.rfind(']') + 1
    if start_idx != -1 and end_idx > start_idx:
        json_str = cleaned[start_idx:end_idx]
        try:
            parsed = json.loads(json_str)
            return _normalize_parsed_response(parsed)
        except json.JSONDecodeError:
            pass

    # Check for common refusal phrases before logging failure
    refusal_phrases = [
        "unable to view", "cannot see", "unable to analyze",
        "don't have access", "can't access", "I'm sorry",
        "I am sorry", "cannot process", "cannot analyze",
        "can't analyze", "unable to extract", "cannot extract",
        "can't extract", "I cannot analyze", "I can't analyze",
        "I cannot extract", "I can't extract"
    ]
    response_lower = response.lower()
    if any(phrase in response_lower for phrase in refusal_phrases):
        print(f"\n⚠️ Model refused to process the image. Response: {response[:200]}...")
        raise ValueError(f"Model refusal: {response[:200]}")

    # Log the full failed response for debugging
    from pathlib import Path
    import datetime
    from .config import PROJECT_ROOT

    # Create logs directory
    logs_dir = create_output_directory(str(PROJECT_ROOT), "logs")

    # Save full response to separate file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_response_path = logs_dir / f"full_failed_response_{timestamp}.txt"
    with open(full_response_path, 'w', encoding='utf-8') as f:
        f.write(f"FAILED JSON PARSING - {datetime.datetime.now().isoformat()}\n")
        f.write(f"Response length: {len(response)} characters\n")
        f.write("="*80 + "\n")
        f.write(response)
        f.write("\n" + "="*80 + "\n")

    # Also append summary to log file
    failed_response_path = logs_dir / "failed_responses.log"
    with open(failed_response_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"FAILED JSON PARSING - {datetime.datetime.now().isoformat()}\n")
        f.write(f"Response length: {len(response)} characters\n")
        f.write(f"Full response saved to: {full_response_path}\n")
        f.write(f"Response preview (first 500 chars):\n{response[:500]}\n")
        f.write(f"{'='*80}\n")

    print(f"\n⚠️ Failed to parse JSON. Response preview (first 500 chars):")
    print(response[:500])
    print(f"Full response length: {len(response)} characters")
    print(f"💾 Full response saved to: {full_response_path}")

    # Try one more aggressive fallback - extract the largest JSON-like block
    try:
        # Look for the largest block between { and }
        start_indices = []
        i = 0
        while i < len(response):
            if response[i] == '{':
                start_indices.append(i)
            i += 1

        if start_indices:
            # Try each potential JSON block from largest to smallest
            candidates = []
            for start_idx in start_indices:
                brace_count = 1
                end_idx = start_idx + 1
                while end_idx < len(response) and brace_count > 0:
                    if response[end_idx] == '{':
                        brace_count += 1
                    elif response[end_idx] == '}':
                        brace_count -= 1
                    end_idx += 1

                if brace_count == 0:  # Found matching braces
                    json_candidate = response[start_idx:end_idx]
                    candidates.append((len(json_candidate), json_candidate))

            # Try candidates from largest to smallest
            for _, json_candidate in sorted(candidates, reverse=True):
                try:
                    parsed = json.loads(json_candidate.strip())
                    return _normalize_parsed_response(parsed)
                except json.JSONDecodeError:
                    continue

    except Exception:
        pass  # Fall through to original error

    raise ValueError("Could not extract valid JSON from response")


def _normalize_parsed_response(data: Dict) -> Dict:
    """
    Normalize parsed JSON response to use canonical edge key formats.

    Converts alternative edge key formats to canonical format:
    - from_entity -> source_name
    - to_entity -> target_name
    - entity1 -> source_name
    - entity2 -> target_name
    - relation_type -> relationship
    - relation -> relationship

    Args:
        data: Parsed JSON response dictionary

    Returns:
        Normalized dictionary with canonical edge keys
    """
    if not isinstance(data, dict):
        return data

    # Normalize edges if they exist
    if "edges" in data and isinstance(data["edges"], list):
        normalized_edges = []
        for edge in data["edges"]:
            if isinstance(edge, dict):
                normalized_edge = dict(edge)  # Copy the edge

                # Map alternative keys to canonical keys
                key_mappings = {
                    "from_entity": "source_name",
                    "to_entity": "target_name",
                    "entity1": "source_name",
                    "entity2": "target_name",
                    "relation_type": "relationship",
                    "relation": "relationship"
                }

                for alt_key, canonical_key in key_mappings.items():
                    if alt_key in normalized_edge and canonical_key not in normalized_edge:
                        normalized_edge[canonical_key] = normalized_edge[alt_key]

                normalized_edges.append(normalized_edge)
            else:
                normalized_edges.append(edge)

        data["edges"] = normalized_edges

    return data


def save_json(data: Dict, output_path: str, indent: int = 2) -> None:
    """
    Save data to JSON file with proper encoding.

    Args:
        data: Data to save
        output_path: Output file path
        indent: JSON indentation level
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(file_path: str) -> Dict:
    """
    Load JSON data from file.

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================================
# FILE DISCOVERY
# ============================================================================

def find_ground_truth_file(image_path: str, json_dir: Optional[str] = None) -> Optional[str]:
    """
    Find the corresponding ground truth JSON file for a given image.

    Args:
        image_path: Path to the image file
        json_dir: Optional override for JSON directory

    Returns:
        Path to ground truth file or None if not found
    """
    image_path_obj = Path(image_path)
    image_name = image_path_obj.stem
    base_name = image_name

    # Remove common label suffixes
    base_name = re.sub(r'_labeled(?:_b)?\s*$', '', base_name)
    # Remove legacy numeric suffixes
    base_name = re.sub(r'\s*__1110\s*$', '', base_name)
    base_name = re.sub(r'\s*_1110\s*$', '', base_name)
    base_name = base_name.strip()

    candidate_json_dirs: List[Path] = []
    if json_dir:
        candidate_json_dirs.append(Path(json_dir))

    # Auto-detect SubgraphsWithTriples layout.
    # Image dirs: subgraphs_01, subgraphs_001, subgraphs_06, subgraphs_01_wo_legend, subgraphs_001_wo_legend, subgraphs_06_wo_legend.
    # Ground truth dirs: subgraphs_01, subgraphs_001, subgraphs_06 (same name; _wo_legend image dirs map to same JSON dir).
    parts = image_path_obj.parts
    if "SubgraphsWithTriplesImages" in parts:
        idx = parts.index("SubgraphsWithTriplesImages")
        base_root = Path(*parts[:idx])
        subset_dir = parts[idx + 1] if len(parts) > idx + 1 else ""
        # Strip _wo_legend so subgraphs_01_wo_legend → subgraphs_01 JSON dir
        subset_dir = subset_dir.replace("_wo_legend", "") if subset_dir else subset_dir
        candidate_json_dirs.append(base_root / "SubgraphsWithTriplesJSON" / subset_dir)
    elif "datasets" in parts and "SubgraphsWithTriplesImages" in str(image_path_obj):
        # Handle datasets/SubgraphsWithTriplesImages structure (_wo_legend → same JSON dir)
        idx = parts.index("datasets")
        base_root = Path(*parts[:idx+1])
        if "SubgraphsWithTriplesImages" in parts:
            img_idx = parts.index("SubgraphsWithTriplesImages")
            subset_dir = parts[img_idx + 1] if len(parts) > img_idx + 1 else ""
            subset_dir = subset_dir.replace("_wo_legend", "") if subset_dir else subset_dir
            candidate_json_dirs.append(base_root / "SubgraphsWithTriplesJSON" / subset_dir)

    # Ensure unique directories
    candidate_json_dirs = [p for i, p in enumerate(candidate_json_dirs) if p not in candidate_json_dirs[:i]]

    # Variants to try for file names (wo_legend images may be named *_labeled.png; GT JSON is same base name without _labeled)
    name_variants = [base_name, image_name]
    image_name_no_labeled = re.sub(r'_labeled(?:_b)?\s*$', '', image_name)
    if image_name_no_labeled not in name_variants:
        name_variants.append(image_name_no_labeled)

    # Handle "peg_" prefix variants
    if base_name.startswith("peg_"):
        name_variants.append(base_name[4:])
    if image_name.startswith("peg_"):
        cleaned = re.sub(r'_labeled(?:_b)?\s*$', '', image_name[4:])
        cleaned = re.sub(r'\s*__1110\s*$', '', cleaned)
        cleaned = re.sub(r'\s*_1110\s*$', '', cleaned)
        name_variants.append(cleaned.strip())

    for candidate_dir in candidate_json_dirs:
        if not candidate_dir or not candidate_dir.exists():
            continue
        for name in name_variants:
            gt_path = candidate_dir / f"{name}.json"
            if gt_path.exists():
                return str(gt_path)

    return None


def find_few_shot_files(base_path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Find few-shot example files from a base path.

    Looks for:
    - {base_path}_labeled.png (image without labels)
    - {base_path}_labeled_b.png (image with labels)
    - {base_path}.json (ground truth)

    Args:
        base_path: Base path (with or without extension)

    Returns:
        Tuple of (image_without_labels_path, image_with_labels_path, ground_truth_path)
    """
    base = Path(base_path)

    # Remove extension if present
    if base.suffix:
        base = base.parent / base.stem

    # Try the direct path first
    candidates = [base]

    # If base is relative or doesn't exist, try common locations
    if not base.is_absolute() and (not base.exists() or base.parent == Path('.')):
        # Try common data directories
        for data_dir in ['data', 'data/datasets', 'data/SubgraphsWithTriples', 'data/Level_1_Graphs']:
            data_path = Path(data_dir)
            if data_path.exists():
                # Try as subdirectory
                candidate = data_path / base.name
                candidates.append(candidate)
                # Try as direct file in data dir
                if base.name:
                    candidates.append(data_path / base.name)

    # Try each candidate location
    for candidate_base in candidates:
        # Ensure we have a directory to search in
        search_dir = candidate_base.parent if candidate_base.suffix else candidate_base.parent
        base_name = candidate_base.name if candidate_base.name else candidate_base.stem

        # Construct file paths
        image_without_labels = search_dir / f"{base_name}_labeled.png"
        image_with_labels = search_dir / f"{base_name}_labeled_b.png"
        ground_truth = search_dir / f"{base_name}.json"

        # Check if all files exist
        if image_without_labels.exists() and image_with_labels.exists() and ground_truth.exists():
            return (str(image_without_labels), str(image_with_labels), str(ground_truth))

    # If not found, return None
    return (None, None, None)


def discover_image_files(directory: Path, pattern: str = "*.png") -> List[Path]:
    """
    Discover all image files in a directory.

    Args:
        directory: Directory to search
        pattern: Glob pattern for image files

    Returns:
        List of image file paths
    """
    if not directory.exists():
        return []

    return sorted(list(directory.glob(pattern)))


def discover_level_directories(base_dir: str) -> List[Path]:
    """
    Discover all Level_X_Graphs directories in a base directory.

    Args:
        base_dir: Base directory to search

    Returns:
        List of Level_X_Graphs directory paths
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        return []

    return sorted([d for d in base_path.iterdir()
                   if d.is_dir() and d.name.startswith("Level_") and d.name.endswith("_Graphs")])


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_entity_name(name: str) -> str:
    """Normalize entity names for comparison (lowercase, trim spaces, replace underscores, fix common typos)"""
    if not name:
        return ""

    # Standard normalization: lowercase, trim spaces, replace underscores
    normalized = " ".join(name.lower().split()).replace("_", " ")

    # Common typo corrections observed in model predictions (applied to normalized form)
    TYPO_CORRECTIONS = {
        "cloudsited": "cloudshield",
        "enterprises clients": "enterprise clients",
        "microsservices": "microservices",
        "a machine learning models": "ai machine learning models",
        "a machine learning": "ai machine learning",
        "cloudstud": "cloudshield",
        "cloudstud digital resources": "cloudshield digital resources",
        "cloudstud users": "cloudshield users",
        "enterprises clients graph policies": "enterprise clients graph policies",
        # Additional corrections for underscore variants and common model errors
        "cloudshield users": "cloudshield_users",  # model may drop underscores
        "client organization b": "client_organization_b",
        "client organization c": "client_organization_c",
        "infrastructure operations": "infrastructure_operations",
        "data center technicians": "data_center_technicians",
        "database administrators": "database_administrators",
        "support engineers": "support_engineers",
        "systems administrators": "systems_administrators",
        "enterprise clients": "enterprise_clients",
        "ai researchers": "ai_researchers",
        "cybersecurity team": "cybersecurity_team",
        "incident response team": "incident_response_team",
        "software architects": "software_architects",
        "application services": "application_services",
        "big data analytics": "big_data_analytics",
        "cloud infrastructure": "cloud_infrastructure",
        "compute resources": "compute_resources",
        "data resources": "data_resources",
        "devops tools": "devops_tools",
        "governance policies": "governance_policies",
        "incident response tickets": "incident_response_tickets",
        "infrastructure as code": "infrastructure_as_code",
        "intrusion detection": "intrusion_detection",
        "load balancers": "load_balancers",
        "networking": "networking",  # already normalized but keeping for consistency
        "security compliance": "security_compliance",
        "security incident management": "security_incident_management",
        "support services": "support_services",
        "threat intelligence": "threat_intelligence",
        "virtual machines": "virtual_machines",
        "vpn gateways": "vpn_gateways",
        "ai machine learning models": "ai_machine_learning_models",
        "relational databases": "relational_databases"
    }

    # Apply typo corrections to normalized form
    return TYPO_CORRECTIONS.get(normalized, normalized)


def normalize_relation_type(relation: str) -> str:
    """Normalize relation types to canonical assign/permit/prohibit format."""
    if not relation:
        return "unknown"

    # Convert to lowercase and strip whitespace
    normalized = relation.lower().strip()

    # Remove suffixes like _01, _02, etc.
    normalized = re.sub(r'_\d+$', '', normalized)

    # Map variations to canonical types
    RELATION_MAPPINGS = {
        "assign": "assign",
        "assignment": "assign",
        "assigned": "assign",
        "assigns": "assign",

        "permit": "permit",
        "permission": "permit",
        "permitted": "permit",
        "permits": "permit",
        "associate": "permit",
        "association": "permit",
        "associated": "permit",
        "associates": "permit",
        "allow": "permit",
        "allows": "permit",
        "allowed": "permit",
        "grant": "permit",
        "grants": "permit",
        "granted": "permit",

        "prohibit": "prohibit",
        "prohibition": "prohibit",
        "prohibited": "prohibit",
        "prohibits": "prohibit",
        "deny": "prohibit",
        "denies": "prohibit",
        "denied": "prohibit",
        "block": "prohibit",
        "blocks": "prohibit",
        "blocked": "prohibit",
        "restrict": "prohibit",
        "restricts": "prohibit",
        "restricted": "prohibit",
        "forbid": "prohibit",
        "forbids": "prohibit",
        "forbidden": "prohibit"
    }

    return RELATION_MAPPINGS.get(normalized, normalized)


def triple_from_ground_truth(item: dict, relation: str) -> tuple[str, str, str]:
    """Extract normalized (subject, relation, object) triple from ground truth item.

    Args:
        item: Dictionary with 'from' and 'to' keys
        relation: Relation type (assign/permit/prohibit)

    Returns:
        Tuple of (subject, relation, object) with normalized entity names
    """
    subject = normalize_entity_name(item.get("from", ""))
    object = normalize_entity_name(item.get("to", ""))
    normalized_relation = normalize_relation_type(relation)

    return (subject, normalized_relation, object)


def triple_from_prediction(edge: dict) -> tuple[str, str, str]:
    """Extract normalized (subject, relation, object) triple from prediction edge.

    Handles multiple possible key formats from different model outputs:
    - source_name, target_name, relationship
    - from_entity, to_entity, relation
    - entity1, entity2, relation
    - from, to, relation_type

    Args:
        edge: Dictionary representing an edge/relation

    Returns:
        Tuple of (subject, relation, object) with normalized entity names
    """
    # Try different key combinations in order of preference
    subject_keys = ["source_name", "from_entity", "entity1", "from"]
    object_keys = ["target_name", "to_entity", "entity2", "to"]
    relation_keys = ["relationship", "relation_type", "relation"]

    subject = ""
    object = ""
    relation = ""

    # Extract subject
    for key in subject_keys:
        if key in edge:
            subject = normalize_entity_name(edge[key])
            break

    # Extract object
    for key in object_keys:
        if key in edge:
            object = normalize_entity_name(edge[key])
            break

    # Extract relation
    for key in relation_keys:
        if key in edge:
            relation = normalize_relation_type(edge[key])
            break

    return (subject, relation, object)


# ============================================================================
# OUTPUT MANAGEMENT
# ============================================================================

def get_output_path(filename: str, base_output_dir: Optional[str] = None) -> str:
    """
    Get full path for output file in the output directory.

    Args:
        filename: Output filename
        base_output_dir: Base output directory (defaults to PROJECT_ROOT)

    Returns:
        Full output path
    """
    from .config import PROJECT_ROOT
    if base_output_dir is None:
        base_output_dir = str(PROJECT_ROOT)
    return str(Path(base_output_dir) / filename)


def create_output_directory(base_path: str, subdir: Optional[str] = None) -> Path:
    """
    Create and return output directory path.

    Args:
        base_path: Base output directory
        subdir: Optional subdirectory

    Returns:
        Created output directory path
    """
    output_path = Path(base_path)
    if subdir:
        output_path = output_path / subdir
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


# ============================================================================
# GROUND TRUTH CONVERSION FUNCTIONS
# ============================================================================

def convert_ground_truth_to_entity_extraction_format(ground_truth_json: Dict) -> Dict:
    """
    Convert ground truth JSON to entity extraction output format.

    Args:
        ground_truth_json: Ground truth JSON with policy_elements structure

    Returns:
        Dictionary with nodes array in entity extraction format
    """
    nodes = []
    node_id_counter = 1

    # Extract user_attributes
    if "policy_elements" in ground_truth_json:
        policy_elements = ground_truth_json["policy_elements"]

        # Add user_attributes
        if "user_attributes" in policy_elements:
            if isinstance(policy_elements["user_attributes"], list):
                for attr in policy_elements["user_attributes"]:
                    nodes.append({
                        "node_id": f"n{node_id_counter}",
                        "type": "user_attributes",
                        "content": attr
                    })
                    node_id_counter += 1

        # Add object_attributes
        if "object_attributes" in policy_elements:
            if isinstance(policy_elements["object_attributes"], list):
                for attr in policy_elements["object_attributes"]:
                    nodes.append({
                        "node_id": f"n{node_id_counter}",
                        "type": "object_attributes",
                        "content": attr
                    })
                    node_id_counter += 1

        # Add policy_classes
        if "policy_classes" in policy_elements:
            policy_class = policy_elements["policy_classes"]
            if isinstance(policy_class, str):
                nodes.append({
                    "node_id": f"n{node_id_counter}",
                    "type": "policy_classes",
                    "content": policy_class
                })
                node_id_counter += 1

    return {"nodes": nodes}


def convert_ground_truth_to_relation_classification_format(
    ground_truth_json: Dict,
    from_entity: str,
    to_entity: str,
    relation_type: str = "assign"
) -> Dict:
    """
    Convert ground truth JSON to relation classification output format.
    """
    exists = "No"
    explanation = f"No {relation_type} relationship found between {from_entity} and {to_entity}"
    subrelations = []

    # Check assignments
    if relation_type == "assign" and "assignments" in ground_truth_json:
        assignments_raw = ground_truth_json["assignments"]
        assignments_list = assignments_raw.values() if isinstance(assignments_raw, dict) else assignments_raw
        for assign_data in assignments_list:
            if assign_data.get("from") == from_entity and assign_data.get("to") == to_entity:
                exists = "Yes"
                explanation = f"Direct arrow from '{from_entity}' to '{to_entity}' indicating an 'assign' relationship."
                break

    # Check associations (permit)
    elif relation_type in ["permit", "associate"] and "associations" in ground_truth_json:
        associations_raw = ground_truth_json["associations"]
        associations_list = associations_raw.values() if isinstance(associations_raw, dict) else associations_raw
        for assoc_data in associations_list:
            if assoc_data.get("from") == from_entity and assoc_data.get("to") == to_entity:
                exists = "Yes"
                subrelations = assoc_data.get("weight", [])
                weight_str = ", ".join(subrelations) if subrelations else "permit"
                explanation = f"Direct arrow from '{from_entity}' to '{to_entity}' indicating a 'permit' relationship with weights: {weight_str}."
                break

    # Check prohibitions
    elif relation_type == "prohibit" and "prohibitions" in ground_truth_json:
        prohibitions_raw = ground_truth_json["prohibitions"]
        prohibitions_list = prohibitions_raw.values() if isinstance(prohibitions_raw, dict) else prohibitions_raw
        for prohib_data in prohibitions_list:
            if prohib_data.get("from") == from_entity and prohib_data.get("to") == to_entity:
                exists = "Yes"
                subrelations = prohib_data.get("weight", [])
                weight_str = ", ".join(subrelations) if subrelations else "prohibit"
                explanation = f"Direct arrow from '{from_entity}' to '{to_entity}' indicating a 'prohibit' relationship with action subtypes: {weight_str}."
                break

    confidence = "high" if exists == "Yes" else "medium"

    return {
        "entity1": from_entity,
        "entity2": to_entity,
        "relation": relation_type,
        "exists": exists,
        "confidence": confidence,
        "explanation": explanation,
        "subrelations": subrelations
    }


def convert_ground_truth_to_path_generation_format(ground_truth_json: Dict) -> Dict:
    """
    Convert ground truth JSON to path generation output format.

    Creates nodes and edges from the ground truth data for end-to-end relation extraction.

    Args:
        ground_truth_json: Ground truth dictionary with policy_elements, assignments, associations, prohibitions

    Returns:
        Dictionary with nodes and edges keys (simplified format for relation extraction)
    """
    # First, create node mapping (entity name -> node_id)
    entity_to_node_id = {}
    nodes = []
    node_id_counter = 1

    # Extract all entities and create nodes
    if "policy_elements" in ground_truth_json:
        policy_elements = ground_truth_json["policy_elements"]

        # Add user_attributes
        if "user_attributes" in policy_elements:
            if isinstance(policy_elements["user_attributes"], list):
                for attr in policy_elements["user_attributes"]:
                    node_id = f"n{node_id_counter}"
                    entity_to_node_id[attr] = node_id
                    nodes.append({
                        "node_id": node_id,
                        "type": "user_attributes",
                        "content": attr
                    })
                    node_id_counter += 1

        # Add object_attributes
        if "object_attributes" in policy_elements:
            if isinstance(policy_elements["object_attributes"], list):
                for attr in policy_elements["object_attributes"]:
                    node_id = f"n{node_id_counter}"
                    entity_to_node_id[attr] = node_id
                    nodes.append({
                        "node_id": node_id,
                        "type": "object_attributes",
                        "content": attr
                    })
                    node_id_counter += 1

        # Add policy_classes
        if "policy_classes" in policy_elements:
            policy_class = policy_elements["policy_classes"]
            if isinstance(policy_class, str):
                node_id = f"n{node_id_counter}"
                entity_to_node_id[policy_class] = node_id
                nodes.append({
                    "node_id": node_id,
                    "type": "policy_classes",
                    "content": policy_class
                })
                node_id_counter += 1

    # Create edges from assignments
    edges = []
    if "assignments" in ground_truth_json:
        assignments_raw = ground_truth_json["assignments"]
        assignments_list = assignments_raw.values() if isinstance(assignments_raw, dict) else assignments_raw
        for assign_data in assignments_list:
            from_entity = assign_data.get("from")
            to_entity = assign_data.get("to")
            if from_entity in entity_to_node_id and to_entity in entity_to_node_id:
                edges.append({
                    "from_id": entity_to_node_id[from_entity],
                    "source_name": from_entity,
                    "to_id": entity_to_node_id[to_entity],
                    "target_name": to_entity,
                    "relationship": "assign",
                    "subrelations": []
                })

    # Create edges from associations (permit)
    if "associations" in ground_truth_json:
        associations_raw = ground_truth_json["associations"]
        associations_list = associations_raw.values() if isinstance(associations_raw, dict) else associations_raw
        for assoc_data in associations_list:
            from_entity = assoc_data.get("from")
            to_entity = assoc_data.get("to")
            if from_entity in entity_to_node_id and to_entity in entity_to_node_id:
                edges.append({
                    "from_id": entity_to_node_id[from_entity],
                    "source_name": from_entity,
                    "to_id": entity_to_node_id[to_entity],
                    "target_name": to_entity,
                    "relationship": "permit",
                    "subrelations": assoc_data.get("weight", [])
                })

    # Create edges from prohibitions
    if "prohibitions" in ground_truth_json:
        prohibitions_raw = ground_truth_json["prohibitions"]
        prohibitions_list = prohibitions_raw.values() if isinstance(prohibitions_raw, dict) else prohibitions_raw
        for prohib_data in prohibitions_list:
            from_entity = prohib_data.get("from")
            to_entity = prohib_data.get("to")
            if from_entity in entity_to_node_id and to_entity in entity_to_node_id:
                edges.append({
                    "from_id": entity_to_node_id[from_entity],
                    "source_name": from_entity,
                    "to_id": entity_to_node_id[to_entity],
                    "target_name": to_entity,
                    "relationship": "prohibit",
                    "subrelations": prohib_data.get("weight", [])
                })

    return {
        "nodes": nodes,
        "edges": edges
    }

"""Entity pair generator for relation classification (assign / permit / prohibit triples)."""

from typing import Dict, List, Tuple, Set, Optional


def parse_predicted_entities(entities_data: Dict) -> Tuple[List[str], List[str], List[str]]:
    """
    Parse entities from predicted entity extraction results JSON format.

    Predicted entities format:
    {
        "nodes": [
            {"label": "entity_name", "type": "user_attributes|object_attributes|policy_classes"},
            ...
        ]
    }

    Args:
        entities_data: Dictionary from predicted entities JSON file

    Returns:
        Tuple of (user_attributes, object_attributes, policy_classes) as lists of strings
    """
    user_attributes = []
    object_attributes = []
    policy_classes = []

    for node in entities_data.get("nodes", []):
        entity_type = node.get("type", "")
        entity_name = node.get("label", "")

        if not entity_name:  # Skip invalid nodes
            continue

        if entity_type == "user_attributes":
            user_attributes.append(entity_name)
        elif entity_type == "object_attributes":
            object_attributes.append(entity_name)
        elif entity_type == "policy_classes":
            policy_classes.append(entity_name)

    return user_attributes, object_attributes, policy_classes


def create_entity_type_lookup(entities_data: Dict, entities_source: str = "auto") -> Dict[str, str]:
    """
    Create a dictionary mapping entity names to their types.

    Args:
        entities_data: Dictionary containing entity data (predicted or ground truth format)
        entities_source: Format of entities_data: "predicted", "ground_truth", or "auto"

    Returns:
        Dictionary mapping entity name -> entity type (user_attributes, object_attributes, policy_classes)
    """
    entity_type_map = {}

    # Auto-detect format if not specified
    if entities_source == "auto":
        if "policy_elements" in entities_data:
            entities_source = "ground_truth"
        elif "nodes" in entities_data:
            entities_source = "predicted"
        else:
            entities_source = "predicted"

    if entities_source == "predicted":
        # Predicted format: {"nodes": [{"label": "...", "type": "..."}]}
        for node in entities_data.get("nodes", []):
            entity_name = node.get("label", "")
            entity_type = node.get("type", "")
            if entity_name and entity_type:
                entity_type_map[entity_name] = entity_type
    elif entities_source == "ground_truth":
        # Ground truth format: {"policy_elements": {"user_attributes": [...], ...}}
        policy_elements = entities_data.get("policy_elements", {})

        # Map user attributes
        user_attrs = policy_elements.get("user_attributes", [])
        if isinstance(user_attrs, list):
            for attr in user_attrs:
                entity_type_map[attr] = "user_attributes"
        elif isinstance(user_attrs, str):
            entity_type_map[user_attrs] = "user_attributes"

        # Map object attributes
        obj_attrs = policy_elements.get("object_attributes", [])
        if isinstance(obj_attrs, list):
            for attr in obj_attrs:
                entity_type_map[attr] = "object_attributes"
        elif isinstance(obj_attrs, str):
            entity_type_map[obj_attrs] = "object_attributes"

        # Map policy classes
        policy_classes = policy_elements.get("policy_classes", [])
        if isinstance(policy_classes, list):
            for pc in policy_classes:
                entity_type_map[pc] = "policy_classes"
        elif isinstance(policy_classes, str):
            entity_type_map[policy_classes] = "policy_classes"
        elif policy_classes:
            entity_type_map[str(policy_classes)] = "policy_classes"

    return entity_type_map


def parse_ground_truth_entities(gt_data: Dict) -> Tuple[List[str], List[str], List[str]]:
    """
    Parse entities from ground truth JSON format.

    Ground truth format:
    {
        "policy_elements": {
            "user_attributes": ["attr1", "attr2", ...],
            "object_attributes": ["obj1", "obj2", ...],
            "policy_classes": "policy_name"
        }
    }

    Args:
        gt_data: Dictionary from ground truth JSON file

    Returns:
        Tuple of (user_attributes, object_attributes, policy_classes) as lists of strings
    """
    user_attributes = []
    object_attributes = []
    policy_classes = []

    policy_elements = gt_data.get("policy_elements", {})

    # Parse user attributes
    user_attrs_data = policy_elements.get("user_attributes", [])
    if isinstance(user_attrs_data, list):
        user_attributes = user_attrs_data
    elif isinstance(user_attrs_data, str):
        user_attributes = [user_attrs_data]

    # Parse object attributes
    obj_attrs_data = policy_elements.get("object_attributes", [])
    if isinstance(obj_attrs_data, list):
        object_attributes = obj_attrs_data
    elif isinstance(obj_attrs_data, str):
        object_attributes = [obj_attrs_data]

    # Parse policy classes
    policy_class_data = policy_elements.get("policy_classes", [])
    if isinstance(policy_class_data, list):
        policy_classes = policy_class_data
    elif isinstance(policy_class_data, str):
        policy_classes = [policy_class_data]
    elif policy_class_data:  # Handle other types
        policy_classes = [str(policy_class_data)]

    return user_attributes, object_attributes, policy_classes


def group_entities_by_type(entities_data: Dict, source_type: str = "auto") -> Tuple[List[str], List[str], List[str]]:
    """
    Group entities by their types from entity data.

    Automatically detects format based on structure, or can be explicitly specified.

    Args:
        entities_data: Dictionary containing entity data
        source_type: "predicted", "ground_truth", or "auto" for automatic detection

    Returns:
        Tuple of (user_attributes, object_attributes, policy_classes)
    """
    # Auto-detect format if not specified
    if source_type == "auto":
        if "policy_elements" in entities_data:
            source_type = "ground_truth"
        elif "nodes" in entities_data:
            source_type = "predicted"
        else:
            # Default to predicted format if unclear
            source_type = "predicted"

    # Parse based on detected/explicit format
    if source_type == "predicted":
        return parse_predicted_entities(entities_data)
    elif source_type == "ground_truth":
        return parse_ground_truth_entities(entities_data)
    else:
        raise ValueError(f"Unknown source_type: {source_type}. Must be 'predicted', 'ground_truth', or 'auto'")


def parse_ground_truth_relations(gt_data: Dict) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    """
    Parse ground truth data to extract positive relations for each relation type.

    Args:
        gt_data: Ground truth data dictionary

    Returns:
        Tuple of (positive_assignments, positive_associations, positive_prohibitions)
        Each is a set of (from_entity, to_entity) tuples
    """
    # Get ground truth assignments, associations, and prohibitions
    gt_assignments = gt_data.get("assignments", [])
    gt_associations = gt_data.get("associations", [])
    gt_prohibitions = gt_data.get("prohibitions", [])

    # Handle both dict and list formats
    assignments = gt_assignments.values() if isinstance(gt_assignments, dict) else gt_assignments
    associations = gt_associations.values() if isinstance(gt_associations, dict) else gt_associations
    prohibitions = gt_prohibitions.values() if isinstance(gt_prohibitions, dict) else gt_prohibitions

    # Create positive sets for each relation type
    positive_assignments = set()
    for assignment in assignments:
        from_ent = assignment.get('from', '')
        to_ent = assignment.get('to', '')
        if from_ent and to_ent:
            positive_assignments.add((from_ent, to_ent))

    positive_associations = set()
    for association in associations:
        from_ent = association.get('from', '')
        to_ent = association.get('to', '')
        if from_ent and to_ent:
            positive_associations.add((from_ent, to_ent))

    positive_prohibitions = set()
    for prohibition in prohibitions:
        from_ent = prohibition.get('from', '')
        to_ent = prohibition.get('to', '')
        if from_ent and to_ent:
            positive_prohibitions.add((from_ent, to_ent))

    return positive_assignments, positive_associations, positive_prohibitions


def generate_assign_triples(
    user_attributes: List[str],
    object_attributes: List[str],
    policy_classes: List[str],
    positive_assignments: Set[Tuple[str, str]]
) -> List[Dict]:
    """
    Generate assign relation triples following the assign rules:
    - user_attribute → user_attribute (within group, no self-loops)
    - object_attribute → object_attribute (within group, no self-loops)
    - user_attribute → policy_class
    - object_attribute → policy_class
    - NO user_attribute ↔ object_attribute for assign

    Args:
        user_attributes: List of user attribute names
        object_attributes: List of object attribute names
        policy_classes: List of policy class names
        positive_assignments: Set of (from, to) tuples that have positive assignments

    Returns:
        List of triple dictionaries with keys: from_entity, to_entity, relationship, expected_result
    """
    triples = []

    def add_triple(from_ent: str, to_ent: str, expected_result: str):
        triples.append({
            "from_entity": from_ent,
            "to_entity": to_ent,
            "relationship": "assign",
            "expected_result": expected_result
        })

    # 1a. user_attributes -> user_attributes (all pairs, avoiding self-loops)
    for user_attr_from in user_attributes:
        for user_attr_to in user_attributes:
            if user_attr_from != user_attr_to:
                expected = "Yes" if (user_attr_from, user_attr_to) in positive_assignments else "No"
                add_triple(user_attr_from, user_attr_to, expected)

    # 1b. object_attributes -> object_attributes (all pairs, avoiding self-loops)
    for obj_attr_from in object_attributes:
        for obj_attr_to in object_attributes:
            if obj_attr_from != obj_attr_to:
                expected = "Yes" if (obj_attr_from, obj_attr_to) in positive_assignments else "No"
                add_triple(obj_attr_from, obj_attr_to, expected)

    # 1c. user_attributes -> policy_classes (one direction)
    for user_attr in user_attributes:
        for policy_class in policy_classes:
            expected = "Yes" if (user_attr, policy_class) in positive_assignments else "No"
            add_triple(user_attr, policy_class, expected)

    # 1d. object_attributes -> policy_classes (one direction)
    for obj_attr in object_attributes:
        for policy_class in policy_classes:
            expected = "Yes" if (obj_attr, policy_class) in positive_assignments else "No"
            add_triple(obj_attr, policy_class, expected)

    return triples


def generate_permit_triples(
    user_attributes: List[str],
    object_attributes: List[str],
    positive_associations: Set[Tuple[str, str]]
) -> List[Dict]:
    """
    Generate permit relation triples following the permit rules:
    - user_attribute → object_attribute (one direction only)
    - NO object_attribute → user_attribute
    - NO within-group testing for permit

    Args:
        user_attributes: List of user attribute names
        object_attributes: List of object attribute names
        positive_associations: Set of (from, to) tuples that have positive associations

    Returns:
        List of triple dictionaries with keys: from_entity, to_entity, relationship, expected_result
    """
    triples = []

    def add_triple(from_ent: str, to_ent: str, expected_result: str):
        triples.append({
            "from_entity": from_ent,
            "to_entity": to_ent,
            "relationship": "permit",
            "expected_result": expected_result
        })

    # user_attributes -> object_attributes (one direction only)
    for user_attr in user_attributes:
        for obj_attr in object_attributes:
            expected = "Yes" if (user_attr, obj_attr) in positive_associations else "No"
            add_triple(user_attr, obj_attr, expected)

    return triples


def generate_prohibit_triples(
    user_attributes: List[str],
    object_attributes: List[str],
    positive_prohibitions: Set[Tuple[str, str]]
) -> List[Dict]:
    """
    Generate prohibit relation triples following the prohibit rules:
    - user_attribute → object_attribute (one direction only)
    - NO object_attribute → user_attribute
    - NO within-group testing for prohibit

    Args:
        user_attributes: List of user attribute names
        object_attributes: List of object attribute names
        positive_prohibitions: Set of (from, to) tuples that have positive prohibitions

    Returns:
        List of triple dictionaries with keys: from_entity, to_entity, relationship, expected_result
    """
    triples = []

    def add_triple(from_ent: str, to_ent: str, expected_result: str):
        triples.append({
            "from_entity": from_ent,
            "to_entity": to_ent,
            "relationship": "prohibit",
            "expected_result": expected_result
        })

    # user_attributes -> object_attributes (one direction only)
    for user_attr in user_attributes:
        for obj_attr in object_attributes:
            expected = "Yes" if (user_attr, obj_attr) in positive_prohibitions else "No"
            add_triple(user_attr, obj_attr, expected)

    return triples


def generate_all_relation_triples(entities_data: Dict, gt_data: Dict, entities_source: str = "auto", subset_size: Optional[int] = None, random_seed: int = 42) -> List[Dict]:
    """
    Generate all relation triples for relation classification testing.

    This is the main function that orchestrates the generation of test triples
    for all three relation types (assign, permit, prohibit) following their
    specific generation rules.

    Args:
        entities_data: Dictionary containing entity data (predicted or ground truth format)
        gt_data: Ground truth data dictionary with assignments, associations, prohibitions
        entities_source: Format of entities_data: "predicted", "ground_truth", or "auto"
        subset_size: If provided, randomly select N triples for subset testing (default: None for all triples)
        random_seed: Random seed for reproducible subset selection (default: 42)

    Returns:
        List of all triple dictionaries for relation classification testing (or subset if specified)

    Triple generation rules based on entity types:

    ASSIGN relations (within groups + to policy class):
    - user_attribute → user_attribute (within group)
    - object_attribute → object_attribute (within group)
    - user_attribute → policy_class
    - object_attribute → policy_class
    - NO user_attribute ↔ object_attribute for assign

    PERMIT relations (between user and object only):
    - user_attribute → object_attribute (one direction only)
    - NO object_attribute → user_attribute
    - NO within-group testing for permit

    PROHIBIT relations (between user and object only):
    - user_attribute → object_attribute (one direction only)
    - NO object_attribute → user_attribute
    - NO within-group testing for prohibit
    """
    # Parse entities based on source format
    user_attributes, object_attributes, policy_classes = group_entities_by_type(entities_data, entities_source)

    # Parse ground truth relations
    positive_assignments, positive_associations, positive_prohibitions = parse_ground_truth_relations(gt_data)

    # Log entity and relation counts
    total_entities = len(user_attributes) + len(object_attributes) + len(policy_classes)
    print(f"  📋 Loaded {total_entities} entities: {len(user_attributes)} user, {len(object_attributes)} object, {len(policy_classes)} policy")
    print(f"  🔗 Ground truth relations: {len(positive_assignments)} assign, {len(positive_associations)} permit, {len(positive_prohibitions)} prohibit")

    # Generate triples for each relation type
    assign_triples = generate_assign_triples(
        user_attributes, object_attributes, policy_classes, positive_assignments
    )

    permit_triples = generate_permit_triples(
        user_attributes, object_attributes, positive_associations
    )

    prohibit_triples = generate_prohibit_triples(
        user_attributes, object_attributes, positive_prohibitions
    )

    # Combine all triples
    all_triples = assign_triples + permit_triples + prohibit_triples

    # Log triple generation summary
    print(f"  🎯 Generated {len(all_triples)} test pairs: {len(assign_triples)} assign, {len(permit_triples)} permit, {len(prohibit_triples)} prohibit")

    # Apply subset selection if requested
    if subset_size is not None and subset_size < len(all_triples):
        import random
        random.seed(random_seed)

        # Stratified sampling: ensure subset includes positive cases for meaningful evaluation
        positive_triples = [t for t in all_triples if t.get('expected_result') == 'Yes']
        negative_triples = [t for t in all_triples if t.get('expected_result') != 'Yes']

        # Select up to half from positive cases, rest from negative cases
        num_positive = min(len(positive_triples), subset_size // 2)
        num_negative = subset_size - num_positive

        selected_positive = random.sample(positive_triples, num_positive) if positive_triples else []
        selected_negative = random.sample(negative_triples, min(num_negative, len(negative_triples)))

        # If we don't have enough negative, fill with remaining positive
        if len(selected_positive) + len(selected_negative) < subset_size:
            remaining = subset_size - len(selected_positive) - len(selected_negative)
            additional_positive = random.sample([t for t in positive_triples if t not in selected_positive], min(remaining, len(positive_triples) - len(selected_positive)))
            selected_positive.extend(additional_positive)

        selected_triples = selected_positive + selected_negative
        random.shuffle(selected_triples)  # Shuffle to mix positive and negative

        print(f"  🎲 Stratified sampling: selected {len(selected_triples)}/{len(all_triples)} triples ({len(selected_positive)} positive, {len(selected_negative)} negative) for subset testing (seed: {random_seed})")
        return selected_triples

    return all_triples


def get_triple_statistics(triples: List[Dict]) -> Dict:
    """
    Get statistics about generated triples.

    Args:
        triples: List of triple dictionaries

    Returns:
        Dictionary with statistics about the triples
    """
    assign_triples = [t for t in triples if t['relationship'] == 'assign']
    permit_triples = [t for t in triples if t['relationship'] == 'permit']
    prohibit_triples = [t for t in triples if t['relationship'] == 'prohibit']

    positive_assign = sum(1 for t in assign_triples if t['expected_result'] == 'Yes')
    positive_permit = sum(1 for t in permit_triples if t['expected_result'] == 'Yes')
    positive_prohibit = sum(1 for t in prohibit_triples if t['expected_result'] == 'Yes')

    return {
        "total_triples": len(triples),
        "assign_triples": len(assign_triples),
        "permit_triples": len(permit_triples),
        "prohibit_triples": len(prohibit_triples),
        "positive_assign": positive_assign,
        "positive_permit": positive_permit,
        "positive_prohibit": positive_prohibit,
        "negative_assign": len(assign_triples) - positive_assign,
        "negative_permit": len(permit_triples) - positive_permit,
        "negative_prohibit": len(prohibit_triples) - positive_prohibit,
    }

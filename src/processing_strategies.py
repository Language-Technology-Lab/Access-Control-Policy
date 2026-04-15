"""Processing strategies (entity extraction, relation classification, path generation)."""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time

from .config import (
    Entity, RelationshipResult, KnowledgeGraph, ProcessingResult,
    ProcessingConfig, APIConfig, MAX_IMAGE_SIDE,
    SUPPORTED_VISION_MODELS, FALLBACK_VISION_MODELS
)
from .file_utils import encode_image_to_base64, parse_json_response, save_json
from .access_prompt import (
    get_entity_extraction_messages,
    get_relation_classification_messages,
    get_path_generation_messages
)
from .entity_pair_generator import generate_all_relation_triples, get_triple_statistics, create_entity_type_lookup


class ProcessingStrategy(ABC):
    """Abstract base class for processing strategies."""

    def __init__(self, api_config: APIConfig, processing_config: ProcessingConfig):
        self.api_config = api_config
        self.processing_config = processing_config
        self.client = None
        self._setup_client()

    def _setup_client(self):
        """Setup OpenAI client."""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_config.api_key)
        except ImportError:
            raise ImportError("OpenAI package is required for processing")

    def call_vision_model(self, messages: List[Dict], model_override: Optional[str] = None) -> Tuple[str, Dict]:
        """Call vision model with optional model override (e.g. for fallback).

        Returns:
            Tuple of (response_text, usage_info)
        """
        model = model_override or self.api_config.model
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=self.api_config.max_tokens,
            temperature=1,
        )

        choice = response.choices[0] if response.choices else None
        content = choice.message.content if choice and choice.message else None
        response_text = (content if content is not None else "") or ""
        response_text = str(response_text)

        usage = getattr(response, "usage", None)
        usage_info = {
            'prompt_tokens': usage.prompt_tokens if usage else 0,
            'completion_tokens': usage.completion_tokens if usage else 0,
            'total_tokens': usage.total_tokens if usage else 0
        }

        return response_text, usage_info

    @abstractmethod
    def process(self, image_path: str, **kwargs) -> ProcessingResult:
        """Process an image according to the strategy."""
        pass

    @abstractmethod
    def get_method_name(self) -> str:
        """Get the method name for this strategy."""
        pass


class EntityExtractionStrategy(ProcessingStrategy):
    """Strategy for entity extraction from images."""

    def get_method_name(self) -> str:
        return "extract_entities"

    def process(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        **kwargs
    ) -> ProcessingResult:
        """Extract entities from image."""
        try:
            image_base64 = encode_image_to_base64(image_path, max_side=MAX_IMAGE_SIDE)
            filename = Path(image_path).name

            messages = get_entity_extraction_messages(
                image_base64,
                filename,
                few_shot_examples=self.processing_config.few_shot_examples,
                image_detail=self.api_config.image_detail
            )

            response, usage_info = self.call_vision_model(messages)
            response_data = parse_json_response(response)

            # Extract entities
            entities = self._extract_entities_from_response(response_data)

            # Save result
            result_data = {
                "source_image": filename,
                "method": "entity_extraction",
                "total_entities": len(entities),
                "nodes": [entity.model_dump() for entity in entities]
            }

            if output_path:
                save_json(result_data, output_path)

            return ProcessingResult(
                success=True,
                output_path=output_path,
                entities_extracted=len(entities),
                metadata={"entities": entities, "usage": usage_info}
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                error_message=str(e)
            )

    def _extract_entities_from_response(self, response_data: Dict) -> List[Entity]:
        """Extract Entity objects from model response."""
        entities_data = []
        if isinstance(response_data, dict) and "nodes" in response_data:
            entities_data = response_data["nodes"]
        elif isinstance(response_data, list):
            entities_data = response_data

        entities = []
        for node_dict in entities_data:
            try:
                entity_dict = {
                    "label": node_dict.get("content", ""),
                    "type": node_dict.get("type", node_dict.get("task_type", "unknown"))
                }
                entity = Entity(**entity_dict)
                entities.append(entity)
            except Exception as e:
                print(f"Warning: Could not create Entity from {node_dict}: {e}")

        return entities


class RelationClassificationStrategy(ProcessingStrategy):
    """Strategy for binary relation classification."""
    
    # Class-level flag to track if first prompt has been printed
    _first_prompt_printed = False

    def get_method_name(self) -> str:
        return "relation_classification"

    def process(
        self,
        image_path: str,
        triple_data: Dict,
        output_path: Optional[str] = None,
        **kwargs
    ) -> ProcessingResult:
        """Classify relation between two entities."""
        # Handle single triple processing
        return self._process_single_triple(image_path, triple_data, output_path, **kwargs)

    def process_batch(
        self,
        image_path: str,
        triples_data: List[Dict],
        output_path: Optional[str] = None,
        entities_data: Optional[Dict] = None,
        **kwargs
    ) -> ProcessingResult:
        """Classify relations for multiple entity pairs using parallel processing with ThreadPoolExecutor."""
        try:
            total_triples = len(triples_data)
            
            # Create entity_type_lookup from entities_data
            entity_type_lookup = {}
            if entities_data:
                # Auto-detect format since entities_data may have been converted
                entity_type_lookup = create_entity_type_lookup(entities_data, "auto")
            
            # Print the first prompt ONLY for the first image (first time this method is called)
            if triples_data and not RelationClassificationStrategy._first_prompt_printed:
                first_triple = triples_data[0]
                image_base64 = encode_image_to_base64(image_path, max_side=MAX_IMAGE_SIDE)
                first_messages = get_relation_classification_messages(
                    image_base64,
                    first_triple,
                    few_shot_examples=self.processing_config.few_shot_examples
                )
                
                # Extract and print the prompt text
                print("\n" + "="*80)
                print("FIRST PROMPT (for first entity pair):")
                print("="*80)
                for msg in first_messages:
                    if msg.get("role") == "user":
                        content = msg.get("content", [])
                        if isinstance(content, list):
                            for item in content:
                                if item.get("type") == "text":
                                    print(item.get("text", ""))
                        elif isinstance(content, str):
                            print(content)
                    elif msg.get("role") == "system":
                        print(f"\n[System]: {msg.get('content', '')}")
                print("="*80 + "\n")
                
                # Mark that we've printed the first prompt
                RelationClassificationStrategy._first_prompt_printed = True
            
            # Log that all pairs have been generated
            print(f"  📋 All {total_triples} pairs generated. Starting parallel classification...")
            
            # Create a list to store results in order
            results = [None] * total_triples
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=16) as executor:
                # Submit all tasks
                future_to_index = {}
                for i, triple_data in enumerate(triples_data):
                    future = executor.submit(
                        self._process_single_triple,
                        image_path,
                        triple_data,
                        None,  # output_path for individual results
                        entity_type_lookup=entity_type_lookup,
                        **kwargs
                    )
                    future_to_index[future] = i
                
                # Process completed tasks as they finish
                completed_count = 0
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    triple_data = triples_data[index]
                    from_entity = triple_data.get('from_entity', 'unknown')
                    to_entity = triple_data.get('to_entity', 'unknown')
                    relation = triple_data.get('relationship', 'unknown')
                    
                    completed_count += 1
                    progress_msg = f"  🔍 [{completed_count}/{total_triples}] Testing: {from_entity} → {to_entity} ({relation})"
                    print(progress_msg, end="", flush=True)
                    
                    try:
                        result = future.result()
                        if result.success and result.metadata.get("result"):
                            results[index] = result.metadata["result"]
                            print(" ✓")  # Success indicator
                        else:
                            # Create a failed result entry
                            entity1_name = triple_data.get('from_entity', 'unknown')
                            entity2_name = triple_data.get('to_entity', 'unknown')
                            entity1_type = entity_type_lookup.get(entity1_name, "unknown")
                            entity2_type = entity_type_lookup.get(entity2_name, "unknown")
                            
                            failed_result = RelationshipResult(
                                exists="Unknown",
                                confidence="unknown",
                                explanation=f"Processing failed: {result.error_message}",
                                source_image=Path(image_path).name,
                                timestamp=datetime.now().isoformat(),
                                method="relation_classification",
                                entity1=entity1_name,
                                entity1_type=entity1_type,
                                entity2=entity2_name,
                                entity2_type=entity2_type,
                                relation=triple_data.get('relationship', 'unknown'),
                                subrelations=[],
                                groundtruth=None
                            )
                            results[index] = failed_result
                            print(" ✗")  # Failure indicator
                    except Exception as e:
                        # Handle exceptions from individual tasks
                        entity1_name = triple_data.get('from_entity', 'unknown')
                        entity2_name = triple_data.get('to_entity', 'unknown')
                        entity1_type = entity_type_lookup.get(entity1_name, "unknown")
                        entity2_type = entity_type_lookup.get(entity2_name, "unknown")
                        
                        failed_result = RelationshipResult(
                            exists="Unknown",
                            confidence="unknown",
                            explanation=f"Exception during processing: {str(e)}",
                            source_image=Path(image_path).name,
                            timestamp=datetime.now().isoformat(),
                            method="relation_classification",
                            entity1=entity1_name,
                            entity1_type=entity1_type,
                            entity2=entity2_name,
                            entity2_type=entity2_type,
                            relation=triple_data.get('relationship', 'unknown'),
                            subrelations=[],
                            groundtruth=None
                        )
                        results[index] = failed_result
                        print(" ✗")  # Failure indicator

            # Calculate final statistics
            successful_classifications = len([r for r in results if r and r.exists != "Unknown"])
            total_processed = len([r for r in results if r is not None])

            # Aggregate usage from all individual results
            total_usage = {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0
            }
            for result in results:
                if result and hasattr(result, 'metadata') and result.metadata.get('usage'):
                    usage = result.metadata['usage']
                    total_usage['prompt_tokens'] += usage.get('prompt_tokens', 0)
                    total_usage['completion_tokens'] += usage.get('completion_tokens', 0)
                    total_usage['total_tokens'] += usage.get('total_tokens', 0)

            print(f"  ✅ Completed: {successful_classifications}/{total_processed} relations classified successfully")

            # Save all results
            if output_path:
                results_data = [result.model_dump() for result in results if result is not None]
                save_json(results_data, output_path)

            return ProcessingResult(
                success=True,
                output_path=output_path,
                metadata={
                    "results": results,
                    "total_triples": len(triples_data),
                    "successful_classifications": successful_classifications,
                    "usage": total_usage
                }
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                error_message=str(e)
            )

    def _process_single_triple(
        self,
        image_path: str,
        triple_data: Dict,
        output_path: Optional[str] = None,
        entity_type_lookup: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> ProcessingResult:
        """Classify relation between two entities."""
        try:
            from_entity = triple_data['from_entity']
            to_entity = triple_data['to_entity']
            relation_type = triple_data.get('relationship', 'assign')
            expected_result = triple_data.get('expected_result', 'Unknown')

            # Look up entity types
            entity_type_lookup = entity_type_lookup or {}
            entity1_type = entity_type_lookup.get(from_entity, "unknown")
            entity2_type = entity_type_lookup.get(to_entity, "unknown")

            image_base64 = encode_image_to_base64(image_path, max_side=MAX_IMAGE_SIDE)
            messages = get_relation_classification_messages(
                image_base64,
                triple_data,
                few_shot_examples=self.processing_config.few_shot_examples,
                image_detail=self.api_config.image_detail
            )

            response, usage_info = self.call_vision_model(messages)
            try:
                classification_data = parse_json_response(response)
            except (ValueError, json.JSONDecodeError) as e:
                # Log parsing failure but continue with default values
                print(f"  ⚠️ Warning: Failed to parse JSON response for {from_entity} → {to_entity}: {str(e)[:100]}")
                classification_data = {}

            # Extract exists value from API response (model's prediction)
            exists_value = classification_data.get("exists", "Unknown")
            
            # Log "Unknown" predictions for debugging
            if exists_value == "Unknown":
                print(f"  ⚠️ Warning: Prediction returned 'Unknown' for {from_entity} → {to_entity} ({relation_type})")

            # Extract and validate subrelations from classification response
            subrelations = classification_data.get("subrelations", [])
            if not isinstance(subrelations, list):
                subrelations = []

            # Validate subrelations based on relation type and existence
            # - assign relations should never have subrelations
            # - permit/prohibit relations should only have subrelations if they exist
            if relation_type == "assign":
                subrelations = []  # assign relations don't have subrelations
            elif exists_value != "Yes":
                subrelations = []  # no relationship = no subrelations
            # For permit/prohibit with exists="Yes", keep the model's subrelations

            # Groundtruth: Rule-based extraction from ground truth JSON data
            # The expected_result field was already determined by checking if this triple
            # exists in the ground truth assignments/associations/prohibitions
            # (see entity_pair_generator.parse_ground_truth_relations and generate_*_triples)
            groundtruth = expected_result if expected_result in ["Yes", "No"] else None

            result = RelationshipResult(
                exists=exists_value,
                confidence=classification_data.get("confidence", "unknown"),
                explanation=classification_data.get("explanation"),
                source_image=Path(image_path).name,
                timestamp=datetime.now().isoformat(),
                method="relation_classification",
                entity1=from_entity,
                entity1_type=entity1_type,
                entity2=to_entity,
                entity2_type=entity2_type,
                relation=relation_type,
                subrelations=subrelations,
                groundtruth=groundtruth
            )

            if output_path:
                save_json(result.model_dump(), output_path)

            return ProcessingResult(
                success=True,
                output_path=output_path,
                metadata={"result": result, "usage": usage_info}
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                error_message=str(e)
            )


class PathEnumerationStrategy(ProcessingStrategy):
    """Strategy for complete path enumeration."""

    def get_method_name(self) -> str:
        return "enumerate_paths"

    def process(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        **kwargs
    ) -> ProcessingResult:
        """Extract complete knowledge graph with all paths."""
        try:
            image_base64 = encode_image_to_base64(image_path, max_side=MAX_IMAGE_SIDE)
            filename = Path(image_path).name

            messages = get_path_generation_messages(
                image_base64,
                filename,
                few_shot_examples=self.processing_config.few_shot_examples,
                image_detail=self.api_config.image_detail
            )

            response, usage_info = self.call_vision_model(messages)
            knowledge_graph_data = parse_json_response(response)

            # Build knowledge graph
            knowledge_graph = self._build_knowledge_graph(knowledge_graph_data, filename)

            # Save result
            if output_path:
                output_data = knowledge_graph.model_dump()
                # Convert GraphPath objects to dictionaries
                from .config import GraphPath
                output_data["paths"] = [path.__dict__ for path in knowledge_graph.paths]
                save_json(output_data, output_path)

            return ProcessingResult(
                success=True,
                output_path=output_path,
                metadata={
                    "knowledge_graph": knowledge_graph,
                    "nodes": len(knowledge_graph.nodes),
                    "edges": len(knowledge_graph.edges),
                    "paths": len(knowledge_graph.paths),
                    "usage": usage_info
                }
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                error_message=str(e)
            )

    def _build_knowledge_graph(self, data: Dict, filename: str) -> KnowledgeGraph:
        """Build KnowledgeGraph object from response data."""
        from .config import KnowledgeGraphNode, KnowledgeGraphEdge, GraphPath

        # Convert nodes
        nodes = []
        for node_data in data.get("nodes", []):
            try:
                node = KnowledgeGraphNode(
                    id=node_data["node_id"],
                    label=node_data["content"],
                    type=node_data["type"]
                )
                nodes.append(node)
            except Exception as e:
                print(f"Warning: Could not create KnowledgeGraphNode from {node_data}: {e}")

        # Convert edges
        edges = []
        for edge_data in data.get("edges", []):
            try:
                edge = KnowledgeGraphEdge(
                    source=edge_data["from_id"],
                    target=edge_data["to_id"],
                    source_name=edge_data["source_name"],
                    target_name=edge_data["target_name"],
                    relation_type=edge_data["relationship"]
                )
                edges.append(edge)
            except Exception as e:
                print(f"Warning: Could not create KnowledgeGraphEdge from {edge_data}: {e}")

        # Convert paths
        paths = []
        for path_data in data.get("paths", []):
            try:
                path = GraphPath(
                    nodes=path_data["nodes"],
                    edges=path_data.get("relationships", []),
                    description=path_data.get("description")
                )
                paths.append(path)
            except Exception as e:
                print(f"Warning: Could not create GraphPath from {path_data}: {e}")

        # Create metadata
        graph_metadata = data.get("graph_metadata", {})
        graph_metadata.update({
            "timestamp": datetime.now().isoformat(),
            "source_image": filename,
            "processing_method": "path_enumeration",
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "total_paths": len(paths)
        })

        return KnowledgeGraph(
            nodes=nodes,
            edges=edges,
            paths=paths,
            graph_metadata=graph_metadata
        )


class PathGenerationStrategy(ProcessingStrategy):
    """Strategy for end-to-end path generation (nodes + edges extraction)."""

    def get_method_name(self) -> str:
        return "path_generation"

    def process(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        ground_truth_path: Optional[str] = None,
        fuzzy_matching: bool = False,
        **kwargs
    ) -> ProcessingResult:
        """Extract complete knowledge graph with nodes and edges (simplified format)."""
        try:
            image_base64 = encode_image_to_base64(image_path, max_side=MAX_IMAGE_SIDE)
            filename = Path(image_path).name

            # Validate image data
            import base64
            try:
                base64.b64decode(image_base64)
            except Exception as e:
                raise Exception(f"Invalid base64 image data: {e}")

            messages = get_path_generation_messages(
                image_base64,
                filename,
                few_shot_examples=self.processing_config.few_shot_examples,
                image_detail=self.api_config.image_detail
            )

            response_safe = ""
            usage_info = {}
            primary_model = self.api_config.model
            max_attempts = 2
            for attempt in range(max_attempts):
                response, usage_info = self.call_vision_model(messages)
                response_safe = (response or "").strip()
                if response_safe:
                    break
                if attempt < max_attempts - 1:
                    time.sleep(2)  # brief delay before retry (handles transient empty responses)

            # Empty after retries: try fallback models (e.g. gpt-4o-mini for complex graphs)
            if not response_safe and FALLBACK_VISION_MODELS:
                for fallback_model in FALLBACK_VISION_MODELS:
                    if fallback_model == primary_model or fallback_model not in SUPPORTED_VISION_MODELS:
                        continue
                    try:
                        response, usage_info = self.call_vision_model(messages, model_override=fallback_model)
                        response_safe = (response or "").strip()
                        if response_safe:
                            break
                    except Exception:
                        continue

            # Empty response after retries and fallbacks
            if not response_safe:
                raise ValueError(
                    "Model returned empty response (after retry and fallback models). "
                    "Try: (1) --image_detail high for complex images, "
                    "(2) --model gpt-4o-mini explicitly, or "
                    "(3) check API status and rate limits."
                )

            # Try to parse JSON response
            try:
                knowledge_graph_data = parse_json_response(response_safe)
            except ValueError as e:
                # Check if this is a model refusal (handled by parse_json_response)
                if "Model refusal:" in str(e):
                    raise ValueError(
                        f"Model refused to process image: '{response_safe[:200]}...'. "
                        f"This often indicates the image is too complex, too large, or contains content the model cannot analyze properly. "
                        f"Try using --image_detail high for better processing of complex images."
                    ) from e
                # Provide helpful error messages for other common failure cases
                elif "unable to view the contents of the image" in response_safe.lower():
                    raise ValueError(
                        f"Model refused to process image: '{response_safe[:200]}...'. "
                        f"This often indicates the image is too large, too small, or contains content the model cannot analyze. "
                        f"Try using --image_detail high or reducing image resolution."
                    ) from e
                elif "safety instructions" in response_safe.lower() or "safety" in response_safe.lower():
                    raise ValueError(
                        f"Model refused to process image due to safety filters: '{response_safe[:200]}...'. "
                        f"The image may contain inappropriate content."
                    ) from e
                elif "Empty response" in str(e):
                    raise ValueError(
                        "Model returned empty response. "
                        "Try: (1) --image_detail high for complex images, "
                        "(2) a model with larger context (e.g. gpt-4o-mini), or "
                        "(3) check API status and rate limits."
                    ) from e
                else:
                    # Re-raise with additional context
                    raise ValueError(
                        f"Failed to parse model response as JSON. "
                        f"Response preview: '{response_safe[:200]}...'. "
                        f"Original error: {str(e)}"
                    ) from e

            # Build simplified result with nodes and edges only
            result_data = self._build_result(knowledge_graph_data, filename)

            # Evaluate against ground truth if provided
            evaluation_result = None
            if ground_truth_path and Path(ground_truth_path).exists():
                from .evaluation import evaluate_path_generation
                evaluation_result = evaluate_path_generation(
                    result_data, ground_truth_path, quiet=False, fuzzy_matching=fuzzy_matching
                )
                # Add evaluation to result
                result_data["evaluation"] = evaluation_result

            # Save result
            if output_path:
                save_json(result_data, output_path)

            return ProcessingResult(
                success=True,
                output_path=output_path,
                metadata={
                    "result": result_data,
                    "nodes": len(result_data.get("nodes", [])),
                    "edges": len(result_data.get("edges", [])),
                    "evaluation": evaluation_result,
                    "usage": usage_info
                }
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return ProcessingResult(
                success=False,
                error_message=str(e)
            )

    def _build_result(self, data: Dict, filename: str) -> Dict:
        """Build simplified result with nodes, edges, and paths."""
        from datetime import datetime
        from .evaluation import _generate_paths_from_prediction

        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        # Normalize edge keys to canonical format
        edges = self._normalize_edges(edges)

        # Generate paths from the predicted nodes and edges
        try:
            paths = _generate_paths_from_prediction({"nodes": nodes, "edges": edges})
        except Exception:
            paths = []

        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "source_image": filename,
                "processing_method": "path_generation",
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "total_paths": len(paths)
            }
        }

    def _normalize_edges(self, edges: List[Dict]) -> List[Dict]:
        """Normalize edge dictionaries to use canonical key format.

        Args:
            edges: List of edge dictionaries

        Returns:
            List of normalized edge dictionaries
        """
        if not edges:
            return edges

        normalized_edges = []
        for edge in edges:
            if not isinstance(edge, dict):
                normalized_edges.append(edge)
                continue

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

            # Ensure we have the required canonical keys
            if "source_name" not in normalized_edge and "from_id" in normalized_edge:
                # Try to map from_id to source_name using node lookup
                # This is a fallback for cases where edges reference node IDs
                pass  # For now, leave as-is since we don't have node lookup here

            normalized_edges.append(normalized_edge)

        return normalized_edges


class RelationExtractionStrategy(ProcessingStrategy):
    """Strategy for extracting relation triples from entities."""

    def get_method_name(self) -> str:
        return "extract_relation"

    def process(
        self,
        entities_json_path: str = None,
        ground_truth_json_path: str = None,
        output_path: Optional[str] = None,
        entities_data: Dict = None,
        **kwargs
    ) -> ProcessingResult:
        """Extract relation triples from entities and ground truth."""
        try:
            from .file_utils import load_json

            # Load entities - either from parameter or file
            if entities_data is None and entities_json_path:
                entities_data = load_json(entities_json_path)
            elif entities_data is None:
                return ProcessingResult(
                    success=False,
                    error_message="Either entities_data or entities_json_path must be provided"
                )

            # Load ground truth
            if ground_truth_json_path:
                gt_data = load_json(ground_truth_json_path)
            else:
                return ProcessingResult(
                    success=False,
                    error_message="ground_truth_json_path is required"
                )

            # Generate triples
            triples = self._generate_triples(entities_data, gt_data)

            # Get statistics
            stats = get_triple_statistics(triples)

            # Save triples
            if output_path:
                save_json(triples, output_path)

            return ProcessingResult(
                success=True,
                output_path=output_path,
                metadata={
                    "triples": triples,
                    **stats  # Include all statistics from the generator
                }
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                error_message=str(e)
            )

    def _generate_triples(self, entities_data: Dict, gt_data: Dict) -> List[Dict]:
        """
        Generate relation triples from entities and ground truth.

        This method delegates to the entity_pair_generator module for cleaner,
        more modular triple generation logic.

        Args:
            entities_data: Dictionary containing entity nodes with types and names (always in predicted format)
            gt_data: Ground truth data dictionary with assignments, associations, prohibitions

        Returns:
            List of triple dictionaries for relation classification testing

        See entity_pair_generator.generate_all_relation_triples() for detailed rules.
        """
        # entities_data is always in predicted format (converted in core_processor)
        return generate_all_relation_triples(
            entities_data,
            gt_data,
            entities_source="predicted",
            subset_size=self.processing_config.subset_size
        )


# ============================================================================
# STRATEGY FACTORY
# ============================================================================

class ProcessingStrategyFactory:
    """Factory for creating processing strategies."""

    @staticmethod
    def create_strategy(
        method: str,
        api_config: APIConfig,
        processing_config: ProcessingConfig
    ) -> ProcessingStrategy:
        """Create appropriate strategy based on method name."""
        strategies = {
            "extract_entities": EntityExtractionStrategy,
            "relation_classification": RelationClassificationStrategy,
            "enumerate_paths": PathEnumerationStrategy,
            "path_generation": PathGenerationStrategy,
            "extract_relation": RelationExtractionStrategy,
        }

        strategy_class = strategies.get(method)
        if not strategy_class:
            raise ValueError(f"Unknown processing method: {method}")

        return strategy_class(api_config, processing_config)

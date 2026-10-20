"""Import graph JSONL records into Neo4j."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _cypher_identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Graph record has an invalid {field}: {value!r}")
    return f"`{value.replace('`', '``')}`"


def _validate_properties(value: object, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise ValueError(f"Graph record has invalid {field}; expected an object.")
    return value


def _merge_node(tx: Any, node: object) -> None:
    if not isinstance(node, dict):
        raise ValueError("Graph record contains an invalid node.")
    node_id = node.get("id")
    if not isinstance(node_id, (str, int, float, bool)):
        raise ValueError(f"Graph node has an invalid id: {node_id!r}")
    label = _cypher_identifier(node.get("type"), "node type")
    properties = _validate_properties(node.get("properties"), "node properties")
    properties["id"] = node_id
    tx.run(
        f"MERGE (node:{label} {{id: $id}}) "
        "SET node += $properties",
        id=node_id,
        properties=properties,
    ).consume()


def _merge_relationship(tx: Any, relationship: object) -> None:
    if not isinstance(relationship, dict):
        raise ValueError("Graph record contains an invalid relationship.")
    source = relationship.get("source")
    target = relationship.get("target")
    if not isinstance(source, dict) or not isinstance(target, dict):
        raise ValueError("Relationship source and target must be node objects.")

    source_id = source.get("id")
    target_id = target.get("id")
    if not isinstance(source_id, (str, int, float, bool)):
        raise ValueError(f"Relationship has an invalid source id: {source_id!r}")
    if not isinstance(target_id, (str, int, float, bool)):
        raise ValueError(f"Relationship has an invalid target id: {target_id!r}")

    source_label = _cypher_identifier(source.get("type"), "source node type")
    target_label = _cypher_identifier(target.get("type"), "target node type")
    relationship_type = _cypher_identifier(
        relationship.get("type"), "relationship type"
    )
    properties = _validate_properties(
        relationship.get("properties"), "relationship properties"
    )
    tx.run(
        f"MATCH (source:{source_label} {{id: $source_id}}) "
        f"MATCH (target:{target_label} {{id: $target_id}}) "
        f"MERGE (source)-[relationship:{relationship_type}]->(target) "
        "SET relationship += $properties",
        source_id=source_id,
        target_id=target_id,
        properties=properties,
    ).consume()


def _import_record(tx: Any, record: object) -> tuple[int, int]:
    if not isinstance(record, dict):
        raise ValueError("Graph JSONL entries must be JSON objects.")
    nodes = record.get("nodes", [])
    relationships = record.get("relationships", [])
    if not isinstance(nodes, list) or not isinstance(relationships, list):
        raise ValueError("Graph record nodes and relationships must be arrays.")

    for node in nodes:
        _merge_node(tx, node)
    for relationship in relationships:
        if not isinstance(relationship, dict):
            raise ValueError("Graph record contains an invalid relationship.")
        _merge_node(tx, relationship.get("source"))
        _merge_node(tx, relationship.get("target"))
        _merge_relationship(tx, relationship)
    return len(nodes), len(relationships)


def import_graph_jsonl(input_path: Path) -> tuple[int, int, int]:
    """Merge nodes and relationships from a graph JSONL file into Neo4j."""
    from .. import db_connect

    records_imported = 0
    nodes_imported = 0
    relationships_imported = 0
    try:
        db_connect.verify_connection()
        with db_connect.driver.session(
            database=db_connect.NEO4J_DATABASE
        ) as session:
            with input_path.open(encoding="utf-8") as graph_file:
                for line_number, line in enumerate(graph_file, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON in {input_path} at line {line_number}."
                        ) from exc

                    node_count, relationship_count = session.execute_write(
                        _import_record, record
                    )
                    nodes_imported += node_count
                    relationships_imported += relationship_count
                    records_imported += 1
                    logger.info(
                        "Imported graph record %d from line %d",
                        records_imported,
                        line_number,
                    )
    finally:
        db_connect.close_connection()

    return records_imported, nodes_imported, relationships_imported

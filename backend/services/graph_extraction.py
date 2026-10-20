"""Extract graph nodes and relationships from saved JSONL chunks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time

from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai.chat_models import GoogleAPIError

from .llm_model import get_chat_model

logger = logging.getLogger(__name__)
RETRYABLE_API_CODES = {429, 500, 502, 503, 504}
RETRY_DELAY_SECONDS = 120
MAX_EXTRACTION_ATTEMPTS = 5
QUANTITATIVE_EXTRACTION_INSTRUCTIONS = (
    "Extract explicit quantitative facts as well as named entities and "
    "relationships. Attach each quantity to the most relevant node or "
    "relationship, preserving its exact value, unit, metric, year or period, "
    "and population or geography when stated. Keep qualifiers such as "
    "'around', 'more than', 'average', and 'median', and preserve denominators. "
    "For tables, keep each cell value associated with its row label and column "
    "year. Do not invent values or create standalone nodes for numbers that "
    "are only measurements."
)


def _chunk_to_document(chunk: object, input_path: Path, line_number: int) -> Document:
    if not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str):
        raise ValueError(
            f"Expected a JSON object with a text field at "
            f"{input_path}:{line_number}."
        )

    metadata = {
        key: chunk[key]
        for key in ("source", "chunk_index")
        if key in chunk and chunk[key] is not None
    }
    return Document(page_content=chunk["text"], metadata=metadata)


def _chunk_key(source: object, chunk_index: object) -> tuple[str | None, int]:
    if source is not None and not isinstance(source, str):
        raise ValueError(f"Chunk source must be a string or null, got {source!r}.")
    if not isinstance(chunk_index, int) or isinstance(chunk_index, bool):
        raise ValueError(f"Chunk index must be an integer, got {chunk_index!r}.")
    return source, chunk_index


def _load_completed_chunks(output_path: Path) -> set[tuple[str | None, int]]:
    completed: set[tuple[str | None, int]] = set()
    if not output_path.exists():
        return completed

    with output_path.open(encoding="utf-8") as graph_file:
        for line_number, line in enumerate(graph_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {output_path} at line {line_number}."
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object at {output_path}:{line_number}."
                )
            completed.add(
                _chunk_key(record.get("source"), record.get("chunk_index"))
            )
    return completed


def _process_response_with_retry(
    transformer: LLMGraphTransformer, document: Document, chunk_index: int
):
    for attempt in range(1, MAX_EXTRACTION_ATTEMPTS + 1):
        try:
            return transformer.process_response(document)
        except GoogleAPIError as exc:
            if exc.code not in RETRYABLE_API_CODES or attempt == MAX_EXTRACTION_ATTEMPTS:
                raise
            logger.warning(
                "Gemini request for chunk %d failed with HTTP %s; "
                "retrying in %d seconds (attempt %d/%d)",
                chunk_index,
                exc.code,
                RETRY_DELAY_SECONDS,
                attempt + 1,
                MAX_EXTRACTION_ATTEMPTS,
            )
            time.sleep(RETRY_DELAY_SECONDS)
    raise RuntimeError(f"Graph extraction retries exhausted for chunk {chunk_index}.")


def extract_graph_from_chunks(input_path: Path, output_path: Path) -> int:
    """Transform each chunk into nodes and relationships and save them as JSONL."""
    transformer = LLMGraphTransformer(
        llm=get_chat_model(),
        node_properties=True,
        relationship_properties=True,
        additional_instructions=QUANTITATIVE_EXTRACTION_INSTRUCTIONS,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    completed = _load_completed_chunks(output_path)
    if completed:
        logger.info("Resuming graph extraction; %d chunks already saved", len(completed))

    with input_path.open(encoding="utf-8") as chunks_file, output_path.open(
        "a", encoding="utf-8"
    ) as graph_file:
        for line_number, line in enumerate(chunks_file, start=1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {input_path} at line {line_number}."
                ) from exc

            document = _chunk_to_document(chunk, input_path, line_number)
            chunk_index = document.metadata.get("chunk_index", line_number - 1)
            source = document.metadata.get("source")
            key = _chunk_key(source, chunk_index)
            if key in completed:
                logger.info("Skipping already completed chunk %s", chunk_index)
                continue

            logger.info("Extracting graph from chunk %s", chunk_index)
            graph_document = _process_response_with_retry(
                transformer, document, chunk_index
            )
            record = {
                "source": source,
                "chunk_index": chunk_index,
                "nodes": [
                    {
                        "id": node.id,
                        "type": node.type,
                        "properties": node.properties,
                    }
                    for node in graph_document.nodes
                ],
                "relationships": [
                    {
                        "source": {
                            "id": relationship.source.id,
                            "type": relationship.source.type,
                        },
                        "target": {
                            "id": relationship.target.id,
                            "type": relationship.target.type,
                        },
                        "type": relationship.type,
                        "properties": relationship.properties,
                    }
                    for relationship in graph_document.relationships
                ],
            }
            graph_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            graph_file.flush()
            completed.add(key)
            processed += 1

    return processed

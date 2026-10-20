"""Command-line entry point for the knowledge graph pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from langchain_core.documents import Document

from app.services.data_process import chunk_file
from app.services.graph_extraction import extract_graph_from_chunks
from app.services.neo4j_import import import_graph_jsonl

logger = logging.getLogger(__name__)


def save_chunks(chunks: Sequence[Document], output_path: Path) -> None:
    """Write chunks as one JSON object per line for downstream processing."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for chunk in chunks:
            record = {
                "chunk_index": chunk.metadata["chunk_index"],
                "source": chunk.metadata.get("source"),
                "text": chunk.page_content,
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Create semantic chunks, extract a graph, or import a graph into Neo4j."
        )
    )
    parser.add_argument(
        "document",
        type=Path,
        nargs="?",
        help="PDF or text document to convert to semantic chunks",
    )
    input_modes = parser.add_mutually_exclusive_group()
    input_modes.add_argument(
        "--graph-from",
        type=Path,
        help="Extract nodes and relationships from a chunks JSONL file",
    )
    input_modes.add_argument(
        "--import-graph",
        type=Path,
        help="Import extracted graph JSONL nodes and relationships into Neo4j",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="JSONL output path (default depends on the selected mode)",
    )
    args = parser.parse_args()

    if sum(
        mode is not None
        for mode in (args.document, args.graph_from, args.import_graph)
    ) != 1:
        parser.error(
            "Provide exactly one of a document path, --graph-from, or --import-graph."
        )

    if args.import_graph is not None:
        records, nodes, relationships = import_graph_jsonl(args.import_graph)
        logger.info(
            "Imported %d records (%d nodes, %d relationships) into Neo4j",
            records,
            nodes,
            relationships,
        )
        return

    if args.graph_from is not None:
        output_path = args.output or Path("output") / (
            f"{args.graph_from.stem}.graph.jsonl"
        )
        processed = extract_graph_from_chunks(args.graph_from, output_path)
        logger.info(
            "Processed %d chunks; graph output saved to %s", processed, output_path
        )
        return

    chunks = chunk_file(args.document)
    output_path = args.output or Path("output") / (
        f"{args.document.stem}.chunks.jsonl"
    )
    save_chunks(chunks, output_path)
    logger.info("Saved %d chunks to %s", len(chunks), output_path)


if __name__ == "__main__":
    main()

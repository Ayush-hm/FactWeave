"""Extract documents and split them into semantic LLM chunks."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import pymupdf
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError

logger = logging.getLogger(__name__)

EMBEDDING_BATCH_SIZE = 100
EMBEDDING_REQUESTS_PER_MINUTE = 90
EMBEDDING_CACHE_PATH = Path.cwd() / "semantic_embeddings.sqlite3"
DOCLING_CACHE_VERSION = "docling-page-v1-dpi300"


def _pdf_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as pdf_file:
        for block in iter(lambda: pdf_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _get_cached_page(cache_key: str) -> str | None:
    with closing(sqlite3.connect(EMBEDDING_CACHE_PATH)) as connection:
        row = connection.execute(
            "SELECT markdown FROM docling_pages WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
    return row[0] if row is not None else None


def _cache_page(cache_key: str, markdown: str) -> None:
    with closing(sqlite3.connect(EMBEDDING_CACHE_PATH)) as connection:
        connection.execute(
            "INSERT OR REPLACE INTO docling_pages (cache_key, markdown) "
            "VALUES (?, ?)",
            (cache_key, markdown),
        )
        connection.commit()


class _RateLimitedCachedEmbeddings(Embeddings):
    """Rate-limit Gemini embedding requests and persist successful vectors."""

    def __init__(self, embeddings: GoogleGenerativeAIEmbeddings) -> None:
        self._embeddings = embeddings
        self._request_interval = 60 / EMBEDDING_REQUESTS_PER_MINUTE
        self._next_request_at = 0.0
        self._cache_path = EMBEDDING_CACHE_PATH
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self._cache_path)) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    vector TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _cache_key(text: str) -> str:
        value = f"gemini-embedding-001\0{text}".encode("utf-8")
        return hashlib.sha256(value).hexdigest()

    def _load_cached(self, keys: Sequence[str]) -> dict[str, list[float]]:
        cached: dict[str, list[float]] = {}
        with closing(sqlite3.connect(self._cache_path)) as connection:
            for start in range(0, len(keys), 500):
                batch = keys[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"SELECT cache_key, vector FROM embeddings "
                    f"WHERE cache_key IN ({placeholders})",
                    batch,
                )
                cached.update(
                    (key, json.loads(vector)) for key, vector in rows
                )
        return cached

    def _wait_for_request_slot(self) -> None:
        wait_seconds = self._next_request_at - time.monotonic()
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        self._next_request_at = time.monotonic() + self._request_interval

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [self._cache_key(text) for text in texts]
        vectors_by_key = self._load_cached(keys)
        missing_texts = list(
            dict.fromkeys(
                text for text, key in zip(texts, keys) if key not in vectors_by_key
            )
        )

        for start in range(0, len(missing_texts), EMBEDDING_BATCH_SIZE):
            batch = missing_texts[start : start + EMBEDDING_BATCH_SIZE]
            batch_keys = [self._cache_key(text) for text in batch]
            for attempt in range(5):
                self._wait_for_request_slot()
                try:
                    vectors = self._embeddings.embed_documents(
                        batch, batch_size=EMBEDDING_BATCH_SIZE
                    )
                    break
                except GoogleGenerativeAIError as exc:
                    if "RESOURCE_EXHAUSTED" not in str(exc) or attempt == 4:
                        raise
                    retry_delay = re.search(r"retry in ([\d.]+)s", str(exc), re.I)
                    wait_seconds = (
                        float(retry_delay.group(1))
                        if retry_delay
                        else 60.0
                    )
                    logger.warning(
                        "Gemini embedding quota reached; retrying batch in %.1f seconds",
                        wait_seconds,
                    )
                    time.sleep(wait_seconds)

            vectors_by_key.update(zip(batch_keys, vectors))
            with closing(sqlite3.connect(self._cache_path)) as connection:
                connection.executemany(
                    "INSERT OR REPLACE INTO embeddings (cache_key, vector) "
                    "VALUES (?, ?)",
                    [
                        (key, json.dumps(vector))
                        for key, vector in zip(batch_keys, vectors)
                    ],
                )
                connection.commit()
            logger.info(
                "Cached embedding batch (%d/%d texts)",
                min(start + len(batch), len(missing_texts)),
                len(missing_texts),
            )

        return [vectors_by_key[key] for key in keys]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _create_semantic_chunker():
    """Create a SemanticChunker using Gemini embeddings."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY in the .env file.")
    os.environ.setdefault("GOOGLE_API_KEY", api_key)

    embeddings = _RateLimitedCachedEmbeddings(
        GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    )
    return SemanticChunker(embeddings)


def chunk_documents(documents: Sequence[Document]) -> list[Document]:
    """Split LangChain documents at semantic boundaries using Gemini embeddings."""
    source_documents = [
        document for document in documents if document.page_content.strip()
    ]
    if not source_documents:
        return []

    chunks = _create_semantic_chunker().split_documents(source_documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
    return chunks


def chunk_text(text: str, source: str | None = None) -> list[Document]:
    """Wrap text in a LangChain document and return its semantic chunks."""
    metadata = {"source": source} if source is not None else {}
    return chunk_documents([Document(page_content=text, metadata=metadata)])


def extract_pdf(path: str | Path) -> str:
    """Render PDF pages and cache each page's Docling Markdown output."""
    converter: DocumentConverter | None = None
    page_markdown: list[str] = []
    EMBEDDING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(EMBEDDING_CACHE_PATH)) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS docling_pages (
                cache_key TEXT PRIMARY KEY,
                markdown TEXT NOT NULL
            )
            """
        )
        connection.commit()

    pdf_path = Path(path)
    pdf_fingerprint = _pdf_fingerprint(pdf_path)
    with pymupdf.open(str(path)) as pdf, TemporaryDirectory(
        prefix="docling-pages-"
    ) as image_dir:
        total_pages = pdf.page_count
        logger.info("Processing PDF %s (%d pages)", path, total_pages)
        for page_index in range(pdf.page_count):
            page_number = page_index + 1
            cache_key = (
                f"{DOCLING_CACHE_VERSION}:{pdf_fingerprint}:{page_number}"
            )
            cached_markdown = _get_cached_page(cache_key)
            if cached_markdown is not None:
                logger.info(
                    "Using cached Docling output for page %d of %d",
                    page_number,
                    total_pages,
                )
                if cached_markdown.strip():
                    page_markdown.append(cached_markdown)
                else:
                    logger.warning(
                        "No text extracted from page %d of %d",
                        page_number,
                        total_pages,
                    )
                continue

            page = pdf.load_page(page_index)
            logger.info("Rendering page %d of %d", page_number, total_pages)
            image_path = Path(image_dir) / f"page-{page_number:05}.png"
            page.get_pixmap(dpi=300, alpha=False).save(str(image_path))
            logger.info("Converting page %d of %d with Docling", page_number, total_pages)
            if converter is None:
                converter = DocumentConverter()
            result = converter.convert(str(image_path))
            markdown = result.document.export_to_markdown()
            _cache_page(cache_key, markdown)
            if markdown.strip():
                page_markdown.append(markdown)
            else:
                logger.warning("No text extracted from page %d of %d", page_number, total_pages)

    logger.info("Finished processing PDF %s", path)
    return "\n\n".join(page_markdown)


def chunk_file(path: str | Path) -> list[Document]:
    """Load a PDF or text file as a LangChain document and semantically split it."""
    file_path = Path(path)
    if file_path.suffix.lower() == ".pdf":
        text = extract_pdf(file_path)
    else:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    return chunk_text(text, str(file_path))

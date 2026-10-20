# FactWeave

FactWeave turns PDF or text documents into a Neo4j knowledge graph. It creates semantically meaningful chunks with Gemini embeddings, extracts entities and relationships with Gemini, and stores the resulting graph in Neo4j.

## Pipeline

1. Convert a PDF (via Docling) or read a text document.
2. Split the content into semantic chunks and save them as JSONL.
3. Extract graph nodes, relationships, and explicit quantitative facts into graph JSONL.
4. Merge the graph records into Neo4j.

## Requirements

- Python 3
- A Google Gemini API key
- A Neo4j database only when importing a graph

## Setup

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_ai_api_key

# Required only for the Neo4j import step
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

`GEMINI_API_KEY` may be used instead of `GOOGLE_API_KEY`. Optionally set `GEMINI_MODEL` to override the graph-extraction model.

## Usage

Run the three stages in order:

```powershell
# 1. Create semantic chunks
python main.py path\to\document.pdf

# 2. Extract a graph from the chunks file
python main.py --graph-from output\document.chunks.jsonl

# 3. Import the graph into Neo4j
python main.py --import-graph output\document.chunks.graph.jsonl
```

Use `-o` or `--output` to choose a different JSONL output path:

```powershell
python main.py path\to\document.pdf --output output\my-document.chunks.jsonl
```

## Output

- `output/<document>.chunks.jsonl` contains one semantic chunk per line.
- `output/<document>.chunks.graph.jsonl` contains the nodes and relationships extracted from each chunk.
- `semantic_embeddings.sqlite3` is a local cache for Docling output and Gemini embeddings; it can be regenerated and is ignored by Git.

## Project layout

```text
main.py                     Command-line pipeline entry point
backend/services/           Document processing, Gemini, and Neo4j services
requirements.txt            Python dependencies
```

## Current package-path note

The entry point currently imports `app.services`, while the checked-in service package is named `backend`. Align those names (for example, update the imports in `main.py` to `backend.services`) before running the CLI.


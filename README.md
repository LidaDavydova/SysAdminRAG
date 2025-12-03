# RAG System for a SysAdmin Assistant

A smart RAG system for creating a system administrator assistant using:

* **RAGatouille + ColBERT** for efficient semantic search
* **Ollama gemma:2b** for generating answers in Russian
* **Smart chunking** that respects document structure

## Features

* 🧠 Semantic search using ColBERT
* 📚 Automatic splitting of documents into optimal chunks
* 💬 Interactive chat interface
* 🔍 Search through Ubuntu documentation
* 📖 References to information sources
* 🎯 **LangChain document compressors** for improving context quality (optional)

Demo:

## Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and run Ollama

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Start the Ollama server
ollama serve

# In another terminal, download the gemma:2b model
ollama pull gemma:2b
```

### 3. Prepare the data

Make sure you have a `parsed.jsonl` file with data in the format:

```json
{"id": "...", "source_url": "...", "title": "...", "text": "...", "meta": {...}}
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Ensure Ollama is running

```bash
ollama serve
# In another terminal:
ollama pull gemma:2b
```

### 3. Build the index if it does not exist in `.ragatouille/colbert/indexes`

```bash
python main.py --mode build --data parsed.jsonl
```

### 4. Launch the chat interface

Frontend:

```bash
streamlit run rag_chat_app/frontend/app.py
```

Backend:

```bash
uvicorn rag_chat_app.backend.app:app --reload --port 8000
```

## Usage

### Building the index

First, build the ColBERT index from your data:

```bash
python main.py --mode build --data parsed.jsonl
```

To rebuild the index:

```bash
python main.py --mode build --data parsed.jsonl --rebuild
```

**Note:** Building the index may take some time (10–30 minutes depending on data size and machine performance).

### Interactive chat

After building the index, start the interactive mode:

```bash
python main.py --mode chat
```

Example query:

```
Question: How do I set up Active Directory on Ubuntu?
```

### Test mode

For quick testing:

```bash
python main.py --mode test --query "How do I backup the system?"
```

## Project Structure

```
.
├── benchmarks # 2 benchmarks
│   ├── benchmark_rag.csv
│   ├── benchmark_rag.py
│   ├── build_all_indexes.sh
│   └── build_indexes.log
├── code_data # parse/load/split prepare data
│   ├── add_dataset.py
│   ├── add_url.py
│   ├── parse_ubiuntu.py
│   ├── scraper.py
│   └── split_parsed.py
├── data
│   ├── dataset1.jsonl
│   ├── dataset2.jsonl
│   ├── dataset_upload # datasets can be uploaded (commands.json makes too small chunks)
│   │   ├── commands.json
│   │   └── dataset1.parquet
│   ├── parsed.jsonl
│   ├── parsed_part1.jsonl
│   ├── parsed_part2.jsonl
│   ├── parsed_part3.jsonl
│   ├── parsed_part4.jsonl
│   ├── parsed_part5.jsonl
│   └── urls.txt
├── RAG
│   ├── chunking.py
│   ├── document_compressor.py
│   ├── main.py
│   └── rag_system.py
├── rag_chat_app # app
│   ├── backend
│   │   ├── app.py
│   │   ├── models
│   │   │   └── rag_model.py
│   │   ├── requirements.txt
│   │   ├── routes
│   │   │   ├── chat.py
│   │   └── services
│   │       └── retrieval.py
│   ├── frontend
│   │   ├── app.py
│   │   └── requirements.txt
│   └── README.md
├── README.md
└── requirements.txt
```

## Configuration

### Chunking parameters

In `rag_system.py` you can adjust chunking parameters:

```python
rag = SysAdminRAG(
    chunk_size=512,      # Chunk size (in tokens)
    chunk_overlap=50     # Overlap between chunks
)
```

### Ollama parameters

```bash
python main.py --ollama-url http://localhost:11434 --ollama-model gemma:2b
```

### Using document compressors

LangChain document compressors help filter and compress retrieved documents before sending them to the LLM, improving answer quality:

```bash
# With compression (recommended for better quality)
python main.py --mode chat --use-compression

# With similarity threshold adjustment
python main.py --mode chat --use-compression --compression-threshold 0.8
```

**How it works:**

* After searching documents via ColBERT, the compressor uses `EmbeddingsFilter` for additional filtering
* Documents with low semantic similarity to the query are filtered out
* This reduces noise in the context and improves answer generation quality

**Parameters:**

* `--use-compression`: Enable document compression
* `--compression-threshold`: Similarity threshold (0.0–1.0, default 0.76)

  * Higher = stricter filtering (fewer documents)
  * Lower = looser filtering (more documents)

## Architecture

1. **Chunking** (`chunking.py`):

   * Splits documents into semantically related parts
   * Respects sentence and paragraph boundaries
   * Preserves metadata for each chunk

2. **RAG System** (`rag_system.py`):

   * Uses RAGatouille with ColBERT for indexing
   * Performs semantic search on queries
   * Integrates with Ollama for answer generation

3. **Main** (`main.py`):

   * CLI interface for interacting with the system
   * Modes: build, chat, test

## Example questions

* "How do I set up Active Directory on Ubuntu?"
* "How do I backup the system?"
* "How do I install and configure Bacula?"
* "How do I configure DNS on Ubuntu Server?"
* "How do I use etckeeper to version configuration files?"

## Troubleshooting

### Ollama connection error

Make sure Ollama is running:

```bash
ollama serve
```

Check availability:

```bash
curl http://localhost:11434/api/tags
```

### Index building error

* Make sure you have enough RAM (ColBERT requires ~4–8GB)
* Check the format of `parsed.jsonl`
* Ensure all dependencies are installed

### Slow search

* Reduce the number of returned results (`k` parameter)
* Use smaller chunk sizes
* Make sure the index is correctly built

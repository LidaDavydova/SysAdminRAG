"""
Utility for adding new URLs to existing RAG indexes.
Can add to existing index or create a new shard.
"""

import argparse
import json
import os
import hashlib
import time
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

try:
    from readability import Document
except Exception:
    Document = None

try:
    import trafilatura
except Exception:
    trafilatura = None

from rag_system import SysAdminRAG
from chunking import SmartChunker


def extract_text(html, url=None):
    """Extract text from HTML"""
    if Document is not None:
        doc = Document(html)
        summary = doc.summary()
        soup = BeautifulSoup(summary, "html.parser")
        text = soup.get_text("\n", strip=True)
        title = doc.title() or ""
        return title, text
    if trafilatura is not None:
        text = trafilatura.extract(html, url=url) or ""
        title = ""  # keep it simple
        return title, text
    # fallback naive
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string if soup.title else ""
    for s in soup(["script", "style", "nav", "footer", "header", "aside"]):
        s.extract()
    text = soup.get_text("\n", strip=True)
    return title, text


def make_id(url):
    """Generate ID from URL"""
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return h


def fetch_url(url):
    """Fetch and parse URL"""
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "sysadmin-scraper/1.0"})
        r.raise_for_status()
        html = r.text
        title, text = extract_text(html, url=url)
        
        if not text or len(text.split()) < 5:
            raise ValueError(f"Extracted too small text for {url}")
        
        doc = {
            "id": make_id(url),
            "source_url": url,
            "title": title,
            "text": text,
            "meta": {
                "site": urlparse(url).netloc,
                "crawl_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "original_len_words": len(text.split())
            }
        }
        return doc
    except Exception as e:
        raise Exception(f"Error fetching {url}: {e}")


def add_to_existing_index(url, index_path, rag_system):
    """
    Add new URL to existing index.
    This rebuilds the index with new document added.
    Note: This is expensive as it rebuilds the entire index.
    """
    print(f"Adding URL to existing index: {index_path}")
    
    # Fetch and parse URL
    print(f"Fetching URL: {url}")
    new_doc = fetch_url(url)
    print(f"Extracted {len(new_doc['text'].split())} words from {url}")
    
    # Load existing chunks metadata to get all existing documents
    abs_index_path = os.path.abspath(index_path)
    rel_path = os.path.relpath(abs_index_path, os.getcwd())
    metadata_path = os.path.join(
        ".ragatouille", "colbert", "indexes", rel_path, "chunks_metadata.json"
    )
    
    existing_docs = []
    if os.path.exists(metadata_path):
        print("Loading existing documents from metadata...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            existing_metadata = json.load(f)
            # Extract unique source_ids
            seen_ids = set()
            for chunk_data in existing_metadata.values():
                source_id = chunk_data.get('source_id')
                if source_id and source_id not in seen_ids:
                    seen_ids.add(source_id)
                    # Reconstruct document from metadata
                    existing_docs.append({
                        "id": source_id,
                        "source_url": chunk_data.get('source_url', ''),
                        "title": chunk_data.get('title', ''),
                        "text": chunk_data.get('text', ''),
                        "meta": chunk_data.get('metadata', {})
                    })
        print(f"Found {len(existing_docs)} existing documents")
    
    # Add new document
    all_docs = existing_docs + [new_doc]
    
    # Save to temporary JSONL
    import tempfile
    temp_dir = tempfile.gettempdir()
    temp_jsonl = os.path.join(temp_dir, f"add_url_{int(time.time())}.jsonl")
    with open(temp_jsonl, 'w', encoding='utf-8') as f:
        for doc in all_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    print(f"Rebuilding index with {len(all_docs)} documents (including new one)...")
    print("Warning: This will rebuild the entire index, which may take time.")
    
    # Create temporary RAG instance for this index
    temp_rag = SysAdminRAG(
        index_path=index_path,
        ollama_url=rag_system.ollama_url,
        ollama_model=rag_system.ollama_model,
        chunk_size=rag_system.chunker.chunk_size,
        chunk_overlap=rag_system.chunker.chunk_overlap
    )
    temp_rag.build_index(temp_jsonl, force_rebuild=True)
    
    # Cleanup
    os.remove(temp_jsonl)
    print(f"Successfully added URL to index: {index_path}")


def find_pending_shard(base_index_path):
    """
    Find or create a pending shard for accumulating documents.
    Returns path to pending shard JSONL file.
    """
    pending_dir = os.path.join(base_index_path, "_pending")
    os.makedirs(pending_dir, exist_ok=True)
    pending_file = os.path.join(pending_dir, "pending_docs.jsonl")
    return pending_file


def get_pending_docs_count(pending_file):
    """Count documents in pending file"""
    if not os.path.exists(pending_file):
        return 0
    count = 0
    with open(pending_file, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count


def create_shard_from_pending(pending_file, base_index_path, rag_system, shard_num):
    """Create a new shard from accumulated pending documents"""
    shard_path = os.path.join(base_index_path, f"part{shard_num}")
    print(f"Creating new shard from pending documents: {shard_path}")
    
    # Create RAG system instance for this shard
    shard_rag = SysAdminRAG(
        index_path=shard_path,
        ollama_url=rag_system.ollama_url,
        ollama_model=rag_system.ollama_model,
        chunk_size=rag_system.chunker.chunk_size,
        chunk_overlap=rag_system.chunker.chunk_overlap
    )
    
    print(f"Building index shard with {get_pending_docs_count(pending_file)} documents...")
    shard_rag.build_index(pending_file, force_rebuild=False)
    
    # Clear pending file
    os.remove(pending_file)
    print(f"Successfully created shard: {shard_path}")
    return shard_path


def add_url_to_pending(url, pending_file):
    """Add URL to pending documents file"""
    print(f"Fetching URL: {url}")
    new_doc = fetch_url(url)
    print(f"Extracted {len(new_doc['text'].split())} words from {url}")
    
    # Append to pending file
    with open(pending_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(new_doc, ensure_ascii=False) + "\n")
    
    return new_doc


def create_new_shard(url, base_index_path, rag_system, max_docs_per_shard=25):
    """
    Add URL to pending documents or create new shard if pending is full.
    
    Args:
        url: URL to add
        base_index_path: Base path for indexes
        rag_system: RAG system instance
        max_docs_per_shard: Maximum documents per shard before creating index
    """
    pending_file = find_pending_shard(base_index_path)
    pending_count = get_pending_docs_count(pending_file)
    
    # Add URL to pending
    new_doc = add_url_to_pending(url, pending_file)
    pending_count += 1
    
    print(f"Pending documents: {pending_count}/{max_docs_per_shard}")
    
    # If pending is full or close to full, create shard
    if pending_count >= max_docs_per_shard:
        # Find next shard number
        shard_num = 1
        base_dir = os.path.join(
            ".ragatouille", "colbert", "indexes", base_index_path
        )
        if os.path.exists(base_dir):
            existing_shards = [d for d in os.listdir(base_dir) 
                              if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('part')]
            if existing_shards:
                numbers = []
                for shard in existing_shards:
                    try:
                        num = int(shard.replace('part', ''))
                        numbers.append(num)
                    except:
                        pass
                if numbers:
                    shard_num = max(numbers) + 1
        
        return create_shard_from_pending(pending_file, base_index_path, rag_system, shard_num)
    else:
        print(f"Document added to pending. Will create shard when {max_docs_per_shard} documents accumulated.")
        return None


def main():
    parser = argparse.ArgumentParser(description='Add new URL to RAG system')
    parser.add_argument(
        '--url',
        type=str,
        required=True,
        help='URL to add'
    )
    parser.add_argument(
        '--index',
        type=str,
        default='colbert_index',
        help='Base index path'
    )
    parser.add_argument(
        '--add-to-existing',
        type=str,
        help='Add to existing shard (e.g., colbert_index/part1). If not specified, creates new shard.'
    )
    parser.add_argument(
        '--max-docs-per-shard',
        type=int,
        default=20,
        help='Maximum documents per shard before creating index (default: 20). Documents accumulate in _pending/ until this limit.'
    )
    parser.add_argument(
        '--force-create-shard',
        action='store_true',
        help='Force create shard immediately, even if pending is not full'
    )
    parser.add_argument(
        '--ollama-url',
        type=str,
        default='http://localhost:11434',
        help='Ollama server URL'
    )
    parser.add_argument(
        '--ollama-model',
        type=str,
        default='gemma:2b',
        help='Ollama model name'
    )
    
    args = parser.parse_args()
    
    # Initialize RAG system
    rag = SysAdminRAG(
        index_path=args.index,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model
    )
    
    if args.add_to_existing:
        # Add to specific existing index (rebuilds entire index)
        print("Warning: Adding to existing index will rebuild it completely.")
        print("This may take time. Consider creating a new shard instead.")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
        add_to_existing_index(args.url, args.add_to_existing, rag)
    else:
        # Add to pending or create new shard
        if args.force_create_shard:
            # Force create shard immediately
            pending_file = find_pending_shard(args.index)
            if os.path.exists(pending_file) and get_pending_docs_count(pending_file) > 0:
                # Create shard from pending + new URL
                add_url_to_pending(args.url, pending_file)
                # Find next shard number
                base_dir = os.path.join(".ragatouille", "colbert", "indexes", args.index)
                shard_num = 1
                if os.path.exists(base_dir):
                    existing_shards = [d for d in os.listdir(base_dir) 
                                      if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('part')]
                    if existing_shards:
                        numbers = [int(s.replace('part', '')) for s in existing_shards 
                                  if s.replace('part', '').isdigit()]
                        if numbers:
                            shard_num = max(numbers) + 1
                new_shard_path = create_shard_from_pending(pending_file, args.index, rag, shard_num)
            else:
                # Create shard with just this URL
                new_shard_path = create_new_shard(args.url, args.index, rag, max_docs_per_shard=1)
        else:
            # Add to pending (will create shard when limit reached)
            new_shard_path = create_new_shard(args.url, args.index, rag, args.max_docs_per_shard)
        
        if new_shard_path:
            print(f"\n✓ New shard created: {new_shard_path}")
            print(f"\nThe new shard will be automatically discovered with --auto-discover")
            print(f"Or use it explicitly:")
            print(f"  python main.py --mode chat --indexes '{args.index}/part1,{new_shard_path}'")
        else:
            print(f"\n✓ URL added to pending documents")
            print(f"Shard will be created automatically when {args.max_docs_per_shard} documents are accumulated.")
            print(f"To force create shard now, use --force-create-shard")


if __name__ == "__main__":
    main()


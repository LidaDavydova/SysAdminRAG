"""
RAG system using RAGatouille and ColBERT for intelligent search.
Integrated with Ollama for answer generation.
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import requests

from ragatouille import RAGPretrainedModel
try:
    from chunking import SmartChunker
except:
    from .chunking import SmartChunker


# Import compressors (optional, if langchain is installed)
try:
    from document_compressor import DocumentCompressor
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    DocumentCompressor = None



class SysAdminRAG:
    """RAG system for sysadmin assistant"""
    
    def __init__(
        self,
        index_path: str = "",
        index_paths: Optional[List[str]] = None,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5:0.5b",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        use_document_compression: bool = False,
        compression_similarity_threshold: float = 0.76
    ):
        """
        Args:
            index_path: Path for saving ColBERT index (default)
            index_paths: List of index paths (for sharding across multiple indexes)
            ollama_url: Ollama server URL
            ollama_model: Ollama model name
            chunk_size: Chunk size
            chunk_overlap: Overlap between chunks
            use_document_compression: Whether to use LangChain compressors for filtering
            compression_similarity_threshold: Similarity threshold for compression (0.0-1.0)
        """

        # Support for multiple indexes (shards)
        if index_paths is not None and len(index_paths) > 0:
            self.index_paths = index_paths
            # Use first path as default for build_index and metadata
            self.index_path = index_paths[0]
        else:
            self.index_paths = [index_path]
            self.index_path = index_path
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.chunker = SmartChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        # ColBERT model will be loaded lazily for each index
        self.rag_model = None
        self.index_loaded = False
        self.shard_models = []
        
        # Initialize document compressor (optional)
        self.use_compression = use_document_compression and HAS_LANGCHAIN
        if self.use_compression:
            try:
                self.compressor = DocumentCompressor(
                    use_compression=True,
                    similarity_threshold=compression_similarity_threshold
                )
                print("Document compressor initialized.")
            except Exception as e:
                print(f"Warning: failed to initialize compressor: {e}")
                self.use_compression = False
                self.compressor = None
        else:
            self.compressor = None
        
    
    def _check_ollama_connection(self) -> bool:
        """Checks Ollama server availability"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def build_index(self, jsonl_path: str, force_rebuild: bool = False):
        """
        Builds ColBERT index from JSONL file
        
        Args:
            jsonl_path: Path to JSONL file with documents
            force_rebuild: Rebuild index if it already exists
        """
        if os.path.exists(self.index_path) and not force_rebuild:
            print(f"Index already exists in {self.index_path}. Use force_rebuild=True to rebuild.")
            return
        
        print("Splitting documents into chunks...")
        chunks = self.chunker.chunk_jsonl(jsonl_path)
        print(f"Created {len(chunks)} chunks from documents")
        
        # Prepare data for indexing
        # RAGatouille expects a list of dictionaries with 'doc_id' and 'text' fields
        documents = []
        for chunk in chunks:
            documents.append({
                'doc_id': chunk.chunk_id,
                'text': chunk.text,
                'title': chunk.title,
                'source_url': chunk.source_url,
                'metadata': chunk.metadata
            })
        
        print("Initializing RAGatouille model...")
        # Use pretrained ColBERT model
        self.rag_model = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        
        print("Building ColBERT index...")
        # Prepare data in correct format
        texts = [doc['text'] for doc in documents]
        
        # Create index in standard .ragatouille directory
        # Index name is tied to directory name (for shards)
        # RAGatouille saves indexes in .ragatouille/colbert/indexes/
        # Use relative path from current directory as index name
        abs_index_path = os.path.abspath(self.index_path)
        index_name = os.path.relpath(abs_index_path, os.getcwd())
        print(f"ColBERT index name: {index_name}")
        
        try:
            self.rag_model.index(
                collection=texts,
                index_name=index_name,
                max_document_length=512,
            )
        except TypeError:
            # For old/alternative API
            self.rag_model.index(
                collection=texts,
                index_name=index_name,
                max_document_length=512,
            )
        
        # Save chunk metadata next to index in .ragatouille/colbert/indexes/
        # (needed for displaying sources - title, source_url, etc.)
        index_path_in_ragatouille = os.path.join(
            ".ragatouille", "colbert", "indexes", index_name
        )
        os.makedirs(index_path_in_ragatouille, exist_ok=True)
        
        # Create mapping of chunk index to its metadata
        metadata_path = os.path.join(index_path_in_ragatouille, "chunks_metadata.json")
        metadata = {}
        for idx, chunk in enumerate(chunks):
            metadata[str(idx)] = {
                'chunk_id': chunk.chunk_id,
                'source_id': chunk.source_id,
                'source_url': chunk.source_url,
                'title': chunk.title,
                'chunk_index': chunk.chunk_index,
                'metadata': chunk.metadata,
                'text': chunk.text
            }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"Metadata saved to: {metadata_path}")
        
        self.index_loaded = True
        print(f"Index successfully built and saved to {self.index_path}")
    
    def _get_index_path(self, base_index_path: str) -> Optional[str]:
        # Absolute path
        abs_base_path = os.path.abspath(base_index_path)
        
        # Only use the last part as index name
        index_name = os.path.basename(abs_base_path)
        # Full path in .ragatouille/colbert/indexes/
        index_path = os.path.join(".ragatouille", "colbert", "indexes", index_name)
        
        if os.path.exists(index_path):
            return index_path
        # Fallback
        if os.path.exists(abs_base_path):
            return abs_base_path
        
        return None

    
    def _discover_indexes(self, base_dir: str) -> List[str]:
        """
        Automatically finds all indexes in the specified directory.
        Searches for subdirectories with index_path.txt, chunks_metadata.json or indexes in .ragatouille.
        """
        discovered = []

        base_dir = os.path.join(
            ".ragatouille", "colbert", "indexes", base_dir
        )
        
        if not os.path.exists(base_dir):
            print(f"Warning: directory {base_dir} not found")
            return discovered
        
        # Search all subdirectories in base directory
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            print(f"Checking directory: {item_path}")
            if os.path.isdir(item_path):
                # Check for index indicators
                has_index_path = os.path.exists(os.path.join(item_path, "index_path.txt"))
                has_metadata = os.path.exists(os.path.join(item_path, "chunks_metadata.json"))
                has_index_name = os.path.exists(os.path.join(item_path, "index_name.txt"))
                
                # Also check for index in .ragatouille
                abs_item_path = os.path.abspath(item_path)
                index_name = os.path.relpath(abs_item_path, os.getcwd())
                has_ragatouille_index = os.path.exists(index_name)
                
                if has_index_path or has_metadata or has_index_name or has_ragatouille_index:
                    discovered.append(item_path)
        
        return discovered
    
    def load_index(self, auto_discover: bool = False):
        """
        Lazy check for at least one index.
        Indexes will be loaded one by one during search to save memory.
        
        Args:
            auto_discover: If True, automatically finds all indexes in base directory
        """
        # If auto-discovery is enabled, find all indexes
        if auto_discover:
            discovered_indexes = []
            for base_path in self.index_paths:
                discovered = self._discover_indexes(base_path)
                discovered_indexes.extend(discovered)
            
            if discovered_indexes:
                self.index_paths = discovered_indexes
                print(f"Auto-discovered {len(discovered_indexes)} indexes:")
                for idx_path in discovered_indexes:
                    print(f"  - {idx_path}")
            else:
                print("Warning: auto-discovery found no indexes")
        
        existing = []
        for base in self.index_paths:
            path = self._get_index_path(base)
            if path:
                existing.append((base, path))
                # models from indexes are loaded
                print(f'Loading model from index {os.path.abspath(path)}')
                model = RAGPretrainedModel.from_index(index_path=os.path.abspath(path))
                try:
                    model.search(" ", k=1)
                except Exception as e:
                    print(f"Warning: failed to preload index {path}: {e}")
        
                self.shard_models.append((path, model))
        
        print(f"RAG models from indexes are loaded, n={len(self.shard_models)}")
        
        if not existing:
            raise FileNotFoundError(
                "No indexes found. Make sure you ran build mode "
                "for the required parts (shards)."
            )
        
        self.index_loaded = True
        print("Found indexes for search:")
        for base, path in existing:
            print(f"Base: {base} -> index path: {path}")

    
    def _search_single_index(self, index_path: str, model: RAGPretrainedModel, query: str, k: int) -> List[Dict[str, Any]]:
        """
        Performs search on a single index and returns formatted results.
        """
        
        results = model.search(query, k=k)
        
        # Load metadata from index directory in .ragatouille
        metadata_path = os.path.join(index_path, "chunks_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                chunks_metadata = json.load(f)
        else:
            chunks_metadata = {}
        
        formatted_results = []
        for idx, result in enumerate(results):
            if isinstance(result, str):
                text = result
                score = 1.0
                result_idx = str(idx)
            else:
                text = result.get('text', result.get('content', ''))
                score = result.get('score', result.get('rank', 1.0))
                result_idx = str(result.get('rank', idx))
            
            metadata = chunks_metadata.get(result_idx, {})
            
            formatted_results.append({
                'chunk_id': metadata.get('chunk_id', result_idx),
                'text': text if text else metadata.get('text', ''),
                'score': float(score),
                'title': metadata.get('title', ''),
                'source_url': metadata.get('source_url', ''),
                'metadata': metadata.get('metadata', {}),
                'index_path': index_path,
            })
        
        return formatted_results

    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs search on the index
        
        Args:
            query: Search query
            k: Number of results
        
        Returns:
            List of relevant chunks with metadata
        """
        if not self.index_loaded:
            raise RuntimeError("Index not loaded. Call load_index() or build_index()")
        
        # If multiple indexes (shards) - search each and combine results
        all_results: List[Dict[str, Any]] = []
        for index_path, index_model in self.shard_models:
            shard_results = self._search_single_index(index_path, index_model, query, k=2)
            all_results.extend(shard_results)
        
        if not all_results:
            return []
        
        # Sort by descending score and return top-k
        all_results.sort(key=lambda r: r.get('score', 0.0), reverse=True)
        return all_results[:k]
    
    def _generate_with_ollama(self, prompt: str, context: str) -> str:
        """Generates answer using Ollama"""
        if not self._check_ollama_connection():
            raise ConnectionError(f"Cannot connect to Ollama at {self.ollama_url}")
        
        # Form prompt for sysadmin assistant
        full_prompt = f"""You are an experienced Linux system administrator specializing in Ubuntu Server.

Use ONLY information from the provided documentation below. Do not make up anything.

DOCUMENTATION:
{context}

USER QUESTION: {prompt}

INSTRUCTIONS:
1. Answer clearly and structured in English
2. Use only facts from the documentation above
3. If information is insufficient, honestly say so
4. Provide specific commands and examples from the documentation
5. Avoid general phrases, be specific

ANSWER:"""
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "top_p": 0.9,
                        "num_predict": 512
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result.get('response', 'Sorry, failed to generate answer.')
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def ask(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Main method for getting answer to a question
        
        Args:
            question: User question
            k: Number of relevant chunks to use
        
        Returns:
            Dictionary with answer and metadata
        """
        # Search for relevant chunks
        search_results = self.search(question, k=k)
        
        if not search_results:
            return {
                'answer': 'Sorry, no relevant information found to answer your question.',
                'sources': [],
                'search_results': []
            }
        
        # Apply document compression if enabled
        if self.use_compression and self.compressor:
            try:
                original_count = len(search_results)
                search_results = self.compressor.compress_documents(search_results, question)
                compressed_count = len(search_results)
                if compressed_count < original_count:
                    print(f"Compression: {original_count} -> {compressed_count} documents")
                # If compression filtered all documents, use originals
                if compressed_count == 0 and original_count > 0:
                    print("Warning: compressor filtered all documents. Using originals.")
                    search_results = self.search(question, k=k)  # Re-search without compression
            except Exception as e:
                print(f"Warning: error during document compression: {e}")
                print("Using original documents.")
        
        # Combine context from found chunks
        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"[Document {i} - {result.get('title', 'No title')}]\n{result['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        answer = self._generate_with_ollama(question, context)
        
        # Form sources
        sources = [
            {
                'title': r.get('title', ''),
                'url': r.get('source_url', ''),
                'relevance_score': r.get('score', 0.0)
            }
            for r in search_results
        ]
        
        return {
            'answer': answer,
            'sources': sources,
            'search_results': search_results
        }


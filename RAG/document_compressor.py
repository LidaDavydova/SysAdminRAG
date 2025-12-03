"""
Module for document compression and filtering using LangChain compressors.
Improves context quality before passing to LLM.
"""

from typing import List, Dict, Any, Optional

# LangChain imports with handling of different versions
try:
    from langchain.retrievers.document_compressors import (
        DocumentCompressorPipeline,
        EmbeddingsFilter
    )
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.schema import Document
except ImportError:
    try:
        # Alternative imports for other versions
        from langchain.retrievers.document_compressors import (
            DocumentCompressorPipeline,
            EmbeddingsFilter
        )
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain_core.documents import Document
    except ImportError:
        # Last attempt
        from langchain.retrievers.document_compressors import (
            DocumentCompressorPipeline,
            EmbeddingsFilter
        )
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.docstore.document import Document


class DocumentCompressor:
    """Document compressor for improving RAG quality"""
    
    def __init__(
        self,
        use_compression: bool = True,
        similarity_threshold: float = 0.76,
        embeddings_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        """
        Args:
            use_compression: Whether to use document compression
            similarity_threshold: Similarity threshold for filtering (0.0-1.0)
            embeddings_model: Embeddings model
        """
        self.use_compression = use_compression
        self.similarity_threshold = similarity_threshold
        self.compressor = None
        
        if use_compression:
            try:
                # Initialize embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name=embeddings_model,
                    model_kwargs={'device': 'cpu'}
                )
                
                # Create filter based on embeddings
                embeddings_filter = EmbeddingsFilter(
                    embeddings=embeddings,
                    similarity_threshold=similarity_threshold
                )
                
                # Create compressor pipeline
                self.compressor = DocumentCompressorPipeline(
                    transformers=[embeddings_filter]
                )
                
            except Exception as e:
                print(f"Warning: failed to initialize compressor: {e}")
                print("Compression will be disabled.")
                self.use_compression = False
    
    def compress_documents(
        self,
        documents: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Compresses and filters documents based on query
        
        Args:
            documents: List of documents with fields text, title, source_url, score
            query: Search query
        
        Returns:
            Filtered and compressed list of documents
        """
        if not self.use_compression or not self.compressor:
            return documents
        
        try:
            # Convert to LangChain Document format
            langchain_docs = []
            for doc in documents:
                langchain_docs.append(
                    Document(
                        page_content=doc.get('text', ''),
                        metadata={
                            'title': doc.get('title', ''),
                            'source_url': doc.get('source_url', ''),
                            'score': doc.get('score', 0.0),
                            'chunk_id': doc.get('chunk_id', ''),
                            **doc.get('metadata', {})
                        }
                    )
                )
            
            # Apply compression
            compressed_docs = self.compressor.compress_documents(
                langchain_docs,
                query
            )
            
            # Convert back to our format
            compressed_results = []
            for doc in compressed_docs:
                compressed_results.append({
                    'text': doc.page_content,
                    'title': doc.metadata.get('title', ''),
                    'source_url': doc.metadata.get('source_url', ''),
                    'score': doc.metadata.get('score', 0.0),
                    'chunk_id': doc.metadata.get('chunk_id', ''),
                    'metadata': {k: v for k, v in doc.metadata.items() 
                                if k not in ['title', 'source_url', 'score', 'chunk_id']}
                })
            
            return compressed_results
            
        except Exception as e:
            print(f"Error during document compression: {e}")
            print("Returning original documents without compression.")
            return documents
    
    def filter_by_relevance(
        self,
        documents: List[Dict[str, Any]],
        query: str,
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Simple relevance filtering (by score)
        
        Args:
            documents: List of documents
            query: Search query (not used, but for compatibility)
            min_score: Minimum score for filtering
        
        Returns:
            Filtered list of documents
        """
        if min_score is None:
            min_score = self.similarity_threshold
        
        filtered = [
            doc for doc in documents 
            if doc.get('score', 0.0) >= min_score
        ]
        
        return filtered


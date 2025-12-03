"""
Module for intelligent text chunking for RAG system.
Uses semantic splitting with respect to document structure.
"""

import re
import json
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Chunk:
    """Class for representing a text chunk"""
    text: str
    chunk_id: str
    source_id: str
    source_url: str
    title: str
    chunk_index: int
    metadata: Dict[str, Any]


class SmartChunker:
    """Smart chunker for splitting documents into semantically related parts"""
    
    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 50,
        min_chunk_size: int = 150,
        respect_sentences: bool = True,
        respect_paragraphs: bool = True
    ):
        """
        Args:
            chunk_size: Maximum chunk size in tokens (approximately)
            chunk_overlap: Overlap size between chunks
            min_chunk_size: Minimum chunk size
            respect_sentences: Split only at sentence boundaries
            respect_paragraphs: Split only at paragraph boundaries
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.respect_sentences = respect_sentences
        self.respect_paragraphs = respect_paragraphs
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Splits text into paragraphs"""
        # Split by double line breaks
        paragraphs = re.split(r'\n\s*\n', text)
        # Remove empty paragraphs and extra whitespace
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Splits text into sentences"""
        # Simple splitting by periods, exclamation and question marks
        # Taking abbreviations into account
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _estimate_tokens(self, text: str) -> int:
        """Approximate token count estimation (1 token ≈ 4 characters)"""
        return len(text) // 4
    
    def _split_paragraph(self, paragraph: str) -> List[str]:
        """Splits paragraph into chunks if it's too large"""
        if self._estimate_tokens(paragraph) <= self.chunk_size:
            return [paragraph]
        
        chunks = []
        if self.respect_sentences:
            sentences = self._split_into_sentences(paragraph)
            current_chunk = []
            current_size = 0
            
            for sentence in sentences:
                sentence_size = self._estimate_tokens(sentence)
                
                if current_size + sentence_size > self.chunk_size and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    # Add overlap
                    if self.chunk_overlap > 0:
                        overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                        current_chunk = overlap_sentences + [sentence]
                        current_size = sum(self._estimate_tokens(s) for s in current_chunk)
                    else:
                        current_chunk = [sentence]
                        current_size = sentence_size
                else:
                    current_chunk.append(sentence)
                    current_size += sentence_size
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
        else:
            # Simple character-based splitting
            words = paragraph.split()
            current_chunk = []
            current_size = 0
            
            for word in words:
                word_size = self._estimate_tokens(word + ' ')
                if current_size + word_size > self.chunk_size and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    # Overlap
                    if self.chunk_overlap > 0:
                        overlap_words = current_chunk[-self.chunk_overlap:] if len(current_chunk) >= self.chunk_overlap else current_chunk
                        current_chunk = overlap_words + [word]
                        current_size = sum(self._estimate_tokens(w + ' ') for w in current_chunk)
                    else:
                        current_chunk = [word]
                        current_size = word_size
                else:
                    current_chunk.append(word)
                    current_size += word_size
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def chunk_document(self, doc: Dict[str, Any]) -> List[Chunk]:
        """
        Splits document into chunks
        
        Args:
            doc: Dictionary with fields id, source_url, title, text, meta
        
        Returns:
            List of Chunk objects
        """
        text = doc.get('text', '')
        if not text:
            return []
        
        chunks = []
        
        if self.respect_paragraphs:
            paragraphs = self._split_into_paragraphs(text)
            
            for para_idx, paragraph in enumerate(paragraphs):
                para_chunks = self._split_paragraph(paragraph)
                
                for chunk_idx, chunk_text in enumerate(para_chunks):
                    # Skip too small chunks
                    if self._estimate_tokens(chunk_text) < self.min_chunk_size:
                        continue
                    
                    chunk_id = f"{doc['id']}_chunk_{para_idx}_{chunk_idx}"
                    
                    chunk = Chunk(
                        text=chunk_text,
                        chunk_id=chunk_id,
                        source_id=doc['id'],
                        source_url=doc.get('source_url', ''),
                        title=doc.get('title', ''),
                        chunk_index=len(chunks),
                        metadata={
                            **doc.get('meta', {}),
                            'paragraph_index': para_idx,
                            'chunk_in_paragraph': chunk_idx,
                            'total_chunks': len(para_chunks)
                        }
                    )
                    chunks.append(chunk)
        else:
            # Simple splitting without paragraph consideration
            all_chunks = self._split_paragraph(text)
            for chunk_idx, chunk_text in enumerate(all_chunks):
                if self._estimate_tokens(chunk_text) < self.min_chunk_size:
                    continue
                
                chunk_id = f"{doc['id']}_chunk_{chunk_idx}"
                
                chunk = Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    source_id=doc['id'],
                    source_url=doc.get('source_url', ''),
                    title=doc.get('title', ''),
                    chunk_index=chunk_idx,
                    metadata=doc.get('meta', {})
                )
                chunks.append(chunk)
        
        return chunks
    
    def chunk_jsonl(self, jsonl_path: str) -> List[Chunk]:
        """
        Reads JSONL file and splits all documents into chunks
        
        Args:
            jsonl_path: Path to JSONL file
        
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                doc = json.loads(line)
                chunks = self.chunk_document(doc)
                all_chunks.extend(chunks)
        
        return all_chunks


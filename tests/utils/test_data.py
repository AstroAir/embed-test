"""Test data generators and utilities."""

import random
import string
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

from pdf_vector_system.vector_db.models import DocumentChunk, SearchResult, SearchQuery
from pdf_vector_system.pdf.text_processor import TextChunk


class TestDataGenerator:
    """Generator for various types of test data."""
    
    def __init__(self, seed: int = 42):
        """Initialize with a random seed for reproducible results."""
        random.seed(seed)
        self.seed = seed
    
    def generate_text(
        self,
        min_words: int = 10,
        max_words: int = 100,
        include_special_chars: bool = False,
        include_unicode: bool = False
    ) -> str:
        """Generate random text for testing."""
        word_count = random.randint(min_words, max_words)
        
        # Base word list
        words = [
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "machine", "learning", "artificial", "intelligence", "data", "science",
            "vector", "embedding", "database", "search", "retrieval", "processing",
            "document", "text", "analysis", "natural", "language", "model",
            "algorithm", "neural", "network", "deep", "learning", "transformer"
        ]
        
        # Generate text
        generated_words = [random.choice(words) for _ in range(word_count)]
        text = " ".join(generated_words)
        
        # Add special characters if requested
        if include_special_chars:
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            for _ in range(random.randint(1, 5)):
                pos = random.randint(0, len(text))
                char = random.choice(special_chars)
                text = text[:pos] + char + text[pos:]
        
        # Add unicode characters if requested
        if include_unicode:
            unicode_chars = "café naïve résumé Zürich München"
            unicode_words = unicode_chars.split()
            for _ in range(random.randint(1, 3)):
                pos = random.randint(0, len(generated_words))
                generated_words.insert(pos, random.choice(unicode_words))
            text = " ".join(generated_words)
        
        return text
    
    def generate_chunks(
        self,
        count: int = 5,
        min_words: int = 20,
        max_words: int = 80
    ) -> List[str]:
        """Generate a list of text chunks."""
        return [
            self.generate_text(min_words, max_words)
            for _ in range(count)
        ]
    
    def generate_embeddings(
        self,
        count: int = 5,
        dimension: int = 384,
        normalize: bool = True
    ) -> List[List[float]]:
        """Generate random embeddings."""
        embeddings = []
        
        for _ in range(count):
            # Generate random values
            embedding = [random.gauss(0, 1) for _ in range(dimension)]
            
            # Normalize if requested
            if normalize:
                magnitude = sum(x * x for x in embedding) ** 0.5
                if magnitude > 0:
                    embedding = [x / magnitude for x in embedding]
            
            embeddings.append(embedding)
        
        return embeddings
    
    def generate_pdf_content(
        self,
        page_count: int = 3,
        paragraphs_per_page: int = 3,
        sentences_per_paragraph: int = 4
    ) -> Dict[int, str]:
        """Generate mock PDF content by page."""
        content = {}
        
        for page_num in range(1, page_count + 1):
            paragraphs = []
            
            for _ in range(paragraphs_per_page):
                sentences = []
                for _ in range(sentences_per_paragraph):
                    sentence = self.generate_text(5, 15) + "."
                    sentences.append(sentence.capitalize())
                
                paragraph = " ".join(sentences)
                paragraphs.append(paragraph)
            
            page_content = "\n\n".join(paragraphs)
            content[page_num] = page_content
        
        return content
    
    def generate_metadata(
        self,
        document_id: Optional[str] = None,
        include_optional: bool = True
    ) -> Dict[str, Any]:
        """Generate document metadata."""
        metadata = {
            "document_id": document_id or f"doc_{random.randint(1000, 9999)}",
            "created_at": time.time(),
            "content_length": random.randint(100, 5000),
            "chunk_index": random.randint(0, 50)
        }
        
        if include_optional:
            optional_fields = {
                "page_number": random.randint(1, 10),
                "start_char": random.randint(0, 1000),
                "end_char": random.randint(1001, 2000),
                "title": f"Test Document {random.randint(1, 100)}",
                "author": random.choice(["Alice", "Bob", "Charlie", "Diana"]),
                "subject": random.choice(["Science", "Technology", "Literature", "History"])
            }
            
            # Randomly include some optional fields
            for key, value in optional_fields.items():
                if random.random() > 0.3:  # 70% chance to include
                    metadata[key] = value
        
        return metadata
    
    def generate_document_chunks(
        self,
        count: int = 5,
        document_id: str = "test_doc",
        embedding_dim: int = 384
    ) -> List[DocumentChunk]:
        """Generate DocumentChunk objects."""
        chunks = []
        texts = self.generate_chunks(count)
        embeddings = self.generate_embeddings(count, embedding_dim)
        
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            chunk = DocumentChunk.create_chunk(
                document_id=document_id,
                chunk_index=i,
                content=text,
                embedding=embedding,
                page_number=random.randint(1, 3),
                start_char=i * 100,
                end_char=(i + 1) * 100
            )
            chunks.append(chunk)
        
        return chunks
    
    def generate_search_results(
        self,
        count: int = 5,
        document_id: str = "test_doc"
    ) -> List[SearchResult]:
        """Generate SearchResult objects."""
        results = []
        
        for i in range(count):
            result = SearchResult(
                id=f"{document_id}_chunk_{i}",
                content=self.generate_text(10, 50),
                score=random.uniform(0.5, 1.0),
                metadata=self.generate_metadata(document_id)
            )
            results.append(result)
        
        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def generate_text_chunks(
        self,
        count: int = 5,
        base_text: Optional[str] = None
    ) -> List[TextChunk]:
        """Generate TextChunk objects."""
        if base_text is None:
            base_text = self.generate_text(200, 500)
        
        # Simple chunking by splitting text
        words = base_text.split()
        chunk_size = len(words) // count
        
        chunks = []
        for i in range(count):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < count - 1 else len(words)
            
            chunk_words = words[start_idx:end_idx]
            chunk_content = " ".join(chunk_words)
            
            chunk = TextChunk(
                content=chunk_content,
                chunk_index=i,
                start_char=start_idx * 5,  # Approximate character position
                end_char=end_idx * 5
            )
            chunks.append(chunk)
        
        return chunks


# Convenience functions for common test data
def create_sample_text(
    word_count: int = 50,
    include_special: bool = False,
    include_unicode: bool = False
) -> str:
    """Create sample text for testing."""
    generator = TestDataGenerator()
    return generator.generate_text(
        word_count, word_count,
        include_special, include_unicode
    )


def create_sample_chunks(count: int = 5, min_words: int = 20, max_words: int = 80) -> List[str]:
    """Create sample text chunks for testing."""
    generator = TestDataGenerator()
    return generator.generate_chunks(count, min_words, max_words)


def create_sample_embeddings(count: int = 5, dimension: int = 384) -> List[List[float]]:
    """Create sample embeddings for testing."""
    generator = TestDataGenerator()
    return generator.generate_embeddings(count, dimension)


def create_sample_pdf_content(page_count: int = 3) -> Dict[int, str]:
    """Create sample PDF content for testing."""
    generator = TestDataGenerator()
    return generator.generate_pdf_content(page_count)


def create_sample_metadata(document_id: str = "test_doc") -> Dict[str, Any]:
    """Create sample metadata for testing."""
    generator = TestDataGenerator()
    return generator.generate_metadata(document_id)


def create_realistic_document_content() -> Dict[int, str]:
    """Create realistic document content for comprehensive testing."""
    return {
        1: """
        Introduction to Machine Learning
        
        Machine learning is a subset of artificial intelligence that focuses on the development
        of algorithms and statistical models that enable computer systems to improve their
        performance on a specific task through experience. Unlike traditional programming,
        where explicit instructions are provided, machine learning systems learn patterns
        from data and make predictions or decisions based on that learning.
        
        The field has gained tremendous popularity in recent years due to the availability
        of large datasets, increased computational power, and advances in algorithmic
        techniques. Applications range from image recognition and natural language processing
        to recommendation systems and autonomous vehicles.
        """.strip(),
        
        2: """
        Types of Machine Learning
        
        There are three main categories of machine learning: supervised learning, unsupervised
        learning, and reinforcement learning. Supervised learning involves training models
        on labeled data, where the desired output is known. Common examples include
        classification and regression tasks.
        
        Unsupervised learning, on the other hand, deals with unlabeled data and aims to
        discover hidden patterns or structures. Clustering and dimensionality reduction
        are typical unsupervised learning tasks. Reinforcement learning involves an agent
        learning to make decisions through interaction with an environment, receiving
        rewards or penalties for its actions.
        """.strip(),
        
        3: """
        Deep Learning and Neural Networks
        
        Deep learning is a specialized subset of machine learning that uses artificial
        neural networks with multiple layers to model and understand complex patterns
        in data. These networks are inspired by the structure and function of the human
        brain, consisting of interconnected nodes (neurons) that process information.
        
        The "deep" in deep learning refers to the number of layers in the network.
        Deep neural networks have proven particularly effective in tasks such as image
        recognition, speech processing, and natural language understanding. Popular
        architectures include convolutional neural networks (CNNs) for image processing
        and recurrent neural networks (RNNs) for sequential data.
        """.strip()
    }


def save_test_data_to_file(data: Any, filepath: Path) -> None:
    """Save test data to a JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_test_data_from_file(filepath: Path) -> Any:
    """Load test data from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

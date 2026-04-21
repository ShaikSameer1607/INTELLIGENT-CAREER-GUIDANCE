"""
===============================================================================
BERT EMBEDDING MODULE - Text Feature Extraction
Uses pretrained BERT to convert resume text into dense embeddings
===============================================================================
"""

import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm


class BERTEmbeddingExtractor:
    """
    Extracts 768-dimensional embeddings from text using pretrained BERT.
    
    Why BERT instead of TF-IDF?
    ===========================
    1. SEMANTIC UNDERSTANDING: BERT understands context
       - TF-IDF: "Python" and "programming" are unrelated
       - BERT: Knows they're related concepts
    
    2. CONTEXTUAL EMBEDDINGS: Same word, different meaning
       - "Java" (programming) vs "Java" (coffee) - BERT knows the difference
    
    3. PRETRAINED KNOWLEDGE: BERT already learned from billions of documents
       - No need to train from scratch
       - Just extract features (fast and efficient)
    
    4. DENSE REPRESENTATIONS: 768 dimensions capture rich semantics
       - TF-IDF: Sparse, high-dimensional (thousands of features)
       - BERT: Dense, fixed-dimensional (768 features)
    
    Why NOT train BERT?
    ===================
    - Training requires massive GPU resources
    - We only have 5000 samples (too small for fine-tuning)
    - Feature extraction is fast and works well
    """
    
    def __init__(self, model_name='bert-base-uncased', device=None):
        """
        Initialize BERT tokenizer and model.
        
        Args:
            model_name: Pretrained BERT model name
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        print("="*80)
        print("INITIALIZING BERT EMBEDDING EXTRACTOR")
        print("="*80)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        print("\n[1/2] Loading BERT tokenizer...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        print(f"✓ Tokenizer loaded: {model_name}")
        
        # Load pretrained model (in evaluation mode)
        print("\n[2/2] Loading pretrained BERT model...")
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode (no training)
        print(f"✓ BERT model loaded (frozen for feature extraction)")
        
        # Embedding dimension
        self.embedding_dim = 768
        print(f"\n✓ Embedding dimension: {self.embedding_dim}")
        print("="*80)
    
    def extract_single_embedding(self, text, max_length=512):
        """
        Extract BERT embedding for a single text.
        
        How it works:
        1. Tokenize text into subwords
        2. Convert tokens to IDs
        3. Pass through BERT
        4. Extract [CLS] token embedding (represents entire text)
        
        Args:
            text: Input text string
            max_length: Maximum sequence length (BERT limit: 512)
        
        Returns:
            768-dimensional numpy array
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        # Extract embeddings (no gradient computation for speed)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get [CLS] token embedding (first token)
        # Why [CLS]? It aggregates information from entire sequence
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return cls_embedding[0]  # Remove batch dimension
    
    def extract_embeddings_batch(self, texts, batch_size=16, max_length=512):
        """
        Extract BERT embeddings for multiple texts efficiently.
        
        Why batching?
        - Faster than processing one-by-one
        - Better GPU utilization
        - Progress tracking with tqdm
        
        Args:
            texts: List of text strings
            batch_size: Number of texts per batch
            max_length: Maximum sequence length
        
        Returns:
            Numpy array of shape (num_texts, 768)
        """
        print(f"\nExtracting BERT embeddings for {len(texts)} texts...")
        print(f"Batch size: {batch_size}, Max length: {max_length}")
        
        all_embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="BERT Extraction"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            # Extract embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get [CLS] embeddings
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)
        
        # Combine all batches
        embeddings = np.vstack(all_embeddings)
        
        print(f"✓ Extracted embeddings shape: {embeddings.shape}")
        return embeddings
    
    def extract_embeddings_from_dataframe(self, df, text_column, batch_size=16):
        """
        Extract embeddings from a DataFrame column.
        
        Args:
            df: Pandas DataFrame
            text_column: Column name containing text
            batch_size: Batch size for processing
        
        Returns:
            Numpy array of embeddings
        """
        texts = df[text_column].fillna('').tolist()
        return self.extract_embeddings_batch(texts, batch_size)


# Utility function for easy usage
def get_bert_embeddings(texts, batch_size=16, device=None):
    """
    Convenience function to extract BERT embeddings.
    
    Args:
        texts: List of text strings
        batch_size: Batch size for processing
        device: 'cuda' or 'cpu'
    
    Returns:
        Tuple of (extractor, embeddings)
    """
    extractor = BERTEmbeddingExtractor(device=device)
    embeddings = extractor.extract_embeddings_batch(texts, batch_size)
    return extractor, embeddings


# Example usage
if __name__ == "__main__":
    # Test with sample texts
    sample_texts = [
        "Experienced Python developer with skills in machine learning and deep learning",
        "Frontend web developer proficient in React, JavaScript, and CSS",
        "Data scientist with expertise in statistical analysis and visualization"
    ]
    
    print("Testing BERT Embedding Extractor...")
    extractor = BERTEmbeddingExtractor()
    embeddings = extractor.extract_embeddings_batch(sample_texts, batch_size=2)
    
    print(f"\nSample embeddings shape: {embeddings.shape}")
    print(f"Expected: (3, 768)")
    print(f"Match: {embeddings.shape == (3, 768)}")

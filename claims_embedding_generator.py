import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    BertModel, RobertaModel, DistilBertModel
)
from sentence_transformers import SentenceTransformer
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import the preprocessor from the previous component
from claims_preprocessor import ClaimsPreprocessor, PreprocessingConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration class for embedding generation parameters"""
    # Model configurations
    clinical_model_name: str = "emilyalsentzer/Bio_ClinicalBERT"  # Clinical domain model
    general_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # General purpose model
    fallback_model_name: str = "distilbert-base-uncased"  # Fallback model
    
    # Embedding dimensions
    embedding_dim: int = 768
    max_sequence_length: int = 512
    
    # Feature engineering
    use_positional_embeddings: bool = True
    use_provider_embeddings: bool = True
    use_temporal_embeddings: bool = True
    use_composite_embeddings: bool = True
    
    # Model parameters
    pooling_strategy: str = "mean"  # "mean", "cls", "max"
    normalize_embeddings: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Caching
    cache_embeddings: bool = True
    cache_path: str = "./embedding_cache"

class PositionalEmbedding(nn.Module):
    """Positional embedding layer for claim sequences"""
    
    def __init__(self, max_position: int, embedding_dim: int):
        super(PositionalEmbedding, self).__init__()
        self.max_position = max_position
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(max_position, embedding_dim)
        
    def forward(self, positions):
        return self.embedding(positions)

class ProviderEmbedding(nn.Module):
    """Provider-aware embedding layer"""
    
    def __init__(self, num_providers: int, embedding_dim: int):
        super(ProviderEmbedding, self).__init__()
        self.num_providers = num_providers
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_providers, embedding_dim)
        
    def forward(self, provider_ids):
        return self.embedding(provider_ids)

class TemporalEmbedding(nn.Module):
    """Temporal embedding layer for time-aware representations"""
    
    def __init__(self, embedding_dim: int):
        super(TemporalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.day_embedding = nn.Embedding(32, embedding_dim // 4)  # Days 1-31
        self.month_embedding = nn.Embedding(13, embedding_dim // 4)  # Months 1-12
        self.year_embedding = nn.Embedding(100, embedding_dim // 4)  # Relative years
        self.weekday_embedding = nn.Embedding(7, embedding_dim // 4)  # Weekdays 0-6
        
    def forward(self, day, month, year, weekday):
        day_emb = self.day_embedding(day)
        month_emb = self.month_embedding(month)
        year_emb = self.year_embedding(year)
        weekday_emb = self.weekday_embedding(weekday)
        return torch.cat([day_emb, month_emb, year_emb, weekday_emb], dim=-1)

class ClaimEmbeddingGenerator:
    """
    Claim Embedding Generator Component for Healthcare Claims Processing
    
    This component converts claims into high-dimensional vectors using domain-adapted
    transformer models with positional and provider-aware embeddings.
    """
    
    def __init__(self, config: EmbeddingConfig = None, preprocessor: ClaimsPreprocessor = None):
        """
        Initialize the Claim Embedding Generator
        
        Args:
            config (EmbeddingConfig): Configuration parameters
            preprocessor (ClaimsPreprocessor): Preprocessor instance
        """
        self.config = config or EmbeddingConfig()
        self.preprocessor = preprocessor
        self.device = torch.device(self.config.device)
        
        # Model components
        self.clinical_model = None
        self.clinical_tokenizer = None
        self.sentence_transformer = None
        self.fallback_model = None
        self.fallback_tokenizer = None
        
        # Custom embedding layers
        self.positional_embedding = None
        self.provider_embedding = None
        self.temporal_embedding = None
        
        # Mappings and statistics
        self.provider_mapping = {}
        self.embedding_stats = {}
        self.embedding_cache = {}
        
        # Initialize models
        self._initialize_models()
        
        logger.info("Claim Embedding Generator initialized successfully")
    
    def _initialize_models(self):
        """Initialize transformer models and custom embedding layers"""
        try:
            # Initialize clinical domain model
            self._load_clinical_model()
            
            # Initialize sentence transformer
            self._load_sentence_transformer()
            
            # Initialize fallback model
            self._load_fallback_model()
            
            # Initialize custom embedding layers (will be set up during fitting)
            self._initialize_custom_embeddings()
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def _load_clinical_model(self):
        """Load clinical domain-adapted model"""
        try:
            self.clinical_tokenizer = AutoTokenizer.from_pretrained(
                self.config.clinical_model_name,
                max_length=self.config.max_sequence_length
            )
            self.clinical_model = AutoModel.from_pretrained(
                self.config.clinical_model_name
            ).to(self.device)
            self.clinical_model.eval()
            logger.info(f"Loaded clinical model: {self.config.clinical_model_name}")
        except Exception as e:
            logger.warning(f"Failed to load clinical model: {e}")
            self.clinical_model = None
            self.clinical_tokenizer = None
    
    def _load_sentence_transformer(self):
        """Load sentence transformer model"""
        try:
            self.sentence_transformer = SentenceTransformer(
                self.config.general_model_name,
                device=self.device
            )
            logger.info(f"Loaded sentence transformer: {self.config.general_model_name}")
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}")
            self.sentence_transformer = None
    
    def _load_fallback_model(self):
        """Load fallback model"""
        try:
            self.fallback_tokenizer = AutoTokenizer.from_pretrained(
                self.config.fallback_model_name,
                max_length=self.config.max_sequence_length
            )
            self.fallback_model = AutoModel.from_pretrained(
                self.config.fallback_model_name
            ).to(self.device)
            self.fallback_model.eval()
            logger.info(f"Loaded fallback model: {self.config.fallback_model_name}")
        except Exception as e:
            logger.warning(f"Failed to load fallback model: {e}")
            self.fallback_model = None
            self.fallback_tokenizer = None
    
    def _initialize_custom_embeddings(self):
        """Initialize custom embedding layers"""
        # These will be properly initialized when we know the vocabulary sizes
        if self.config.use_positional_embeddings:
            self.positional_embedding = PositionalEmbedding(
                max_position=1000,  # Will be adjusted based on data
                embedding_dim=self.config.embedding_dim // 4
            ).to(self.device)
        
        if self.config.use_temporal_embeddings:
            self.temporal_embedding = TemporalEmbedding(
                embedding_dim=self.config.embedding_dim // 4
            ).to(self.device)
    
    def _get_text_embeddings(self, texts: List[str], model_type: str = "clinical") -> torch.Tensor:
        """
        Generate embeddings for text using specified model
        
        Args:
            texts (List[str]): List of text strings
            model_type (str): Model to use ("clinical", "sentence", "fallback")
            
        Returns:
            torch.Tensor: Text embeddings
        """
        if not texts:
            return torch.zeros((0, self.config.embedding_dim))
        
        # Clean empty texts
        cleaned_texts = [text if text and str(text).strip() else "unknown" for text in texts]
        
        if model_type == "clinical" and self.clinical_model is not None:
            return self._get_clinical_embeddings(cleaned_texts)
        elif model_type == "sentence" and self.sentence_transformer is not None:
            return self._get_sentence_embeddings(cleaned_texts)
        elif self.fallback_model is not None:
            return self._get_fallback_embeddings(cleaned_texts)
        else:
            logger.warning("No available models for text embedding")
            return torch.zeros((len(texts), self.config.embedding_dim))
    
    def _get_clinical_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Generate embeddings using clinical model"""
        embeddings = []
        batch_size = 8  # Process in batches to manage memory
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                inputs = self.clinical_tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_sequence_length
                ).to(self.device)
                
                # Get model outputs
                outputs = self.clinical_model(**inputs)
                
                # Apply pooling strategy
                if self.config.pooling_strategy == "cls":
                    batch_embeddings = outputs.last_hidden_state[:, 0, :]
                elif self.config.pooling_strategy == "max":
                    batch_embeddings = torch.max(outputs.last_hidden_state, dim=1)[0]
                else:  # mean pooling
                    attention_mask = inputs['attention_mask']
                    batch_embeddings = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                
                embeddings.append(batch_embeddings.cpu())
        
        return torch.cat(embeddings, dim=0)
    
    def _get_sentence_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Generate embeddings using sentence transformer"""
        embeddings = self.sentence_transformer.encode(
            texts,
            convert_to_tensor=True,
            device=self.device
        )
        return embeddings.cpu()
    
    def _get_fallback_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Generate embeddings using fallback model"""
        embeddings = []
        batch_size = 8
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                inputs = self.fallback_tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_sequence_length
                ).to(self.device)
                
                outputs = self.fallback_model(**inputs)
                
                if self.config.pooling_strategy == "cls":
                    batch_embeddings = outputs.last_hidden_state[:, 0, :]
                else:
                    attention_mask = inputs['attention_mask']
                    batch_embeddings = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                
                embeddings.append(batch_embeddings.cpu())
        
        return torch.cat(embeddings, dim=0)
    
    def _create_provider_embeddings(self, provider_ids: List[str], fit: bool = True) -> torch.Tensor:
        """Create provider-aware embeddings"""
        if not self.config.use_provider_embeddings:
            return torch.zeros((len(provider_ids), self.config.embedding_dim // 4))
        
        if fit:
            # Create provider mapping
            unique_providers = list(set(provider_ids))
            self.provider_mapping = {provider: idx for idx, provider in enumerate(unique_providers)}
            
            # Initialize provider embedding layer
            self.provider_embedding = ProviderEmbedding(
                num_providers=len(unique_providers),
                embedding_dim=self.config.embedding_dim // 4
            ).to(self.device)
        
        # Convert provider IDs to indices
        provider_indices = torch.tensor([
            self.provider_mapping.get(provider, 0) for provider in provider_ids
        ], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            embeddings = self.provider_embedding(provider_indices)
        
        return embeddings.cpu()
    
    def _create_positional_embeddings(self, positions: List[int]) -> torch.Tensor:
        """Create positional embeddings for claim sequences"""
        if not self.config.use_positional_embeddings:
            return torch.zeros((len(positions), self.config.embedding_dim // 4))
        
        position_tensor = torch.tensor(positions, dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            embeddings = self.positional_embedding(position_tensor)
        
        return embeddings.cpu()
    
    def _create_temporal_embeddings(self, df: pd.DataFrame) -> torch.Tensor:
        """Create temporal embeddings from date features"""
        if not self.config.use_temporal_embeddings:
            return torch.zeros((len(df), self.config.embedding_dim // 4))
        
        # Extract temporal features
        day = torch.tensor(df['service_date_day'].fillna(1).astype(int).values, dtype=torch.long).to(self.device)
        month = torch.tensor(df['service_date_month'].fillna(1).astype(int).values, dtype=torch.long).to(self.device)
        year = torch.tensor((df['service_date_year'].fillna(2023).astype(int) - 2020).values, dtype=torch.long).to(self.device)  # Relative to 2020
        weekday = torch.tensor(df['service_date_dayofweek'].fillna(0).astype(int).values, dtype=torch.long).to(self.device)
        
        # Ensure values are within bounds
        day = torch.clamp(day, 1, 31)
        month = torch.clamp(month, 1, 12)
        year = torch.clamp(year, 0, 99)
        weekday = torch.clamp(weekday, 0, 6)
        
        with torch.no_grad():
            embeddings = self.temporal_embedding(day, month, year, weekday)
        
        return embeddings.cpu()
    
    def _create_structured_embeddings(self, df: pd.DataFrame) -> torch.Tensor:
        """Create embeddings from structured categorical and numerical features"""
        # Get encoded categorical features
        categorical_cols = [col for col in df.columns if col.endswith('_encoded')]
        numerical_cols = [col for col in df.columns if col.endswith('_scaled')]
        
        features = []
        
        # Add categorical features
        for col in categorical_cols:
            if col in df.columns:
                features.append(df[col].values.reshape(-1, 1))
        
        # Add numerical features
        for col in numerical_cols:
            if col in df.columns:
                features.append(df[col].values.reshape(-1, 1))
        
        if features:
            structured_features = np.concatenate(features, axis=1)
            return torch.tensor(structured_features, dtype=torch.float32)
        else:
            return torch.zeros((len(df), 1))
    
    def generate_claim_embeddings(self, processed_data: Dict[str, Any], fit: bool = True) -> Dict[str, torch.Tensor]:
        """
        Generate comprehensive embeddings for claims
        
        Args:
            processed_data (Dict): Output from ClaimsPreprocessor
            fit (bool): Whether to fit embedding layers
            
        Returns:
            Dict[str, torch.Tensor]: Various types of embeddings
        """
        logger.info("Starting claim embedding generation")
        
        df = processed_data['processed_dataframe']
        embeddings = {}
        
        # 1. Text embeddings from procedure descriptions
        if 'procedure_description' in df.columns:
            procedure_texts = df['procedure_description'].fillna('unknown').tolist()
            
            # Try clinical model first, fallback to others
            text_embeddings = self._get_text_embeddings(procedure_texts, "clinical")
            if text_embeddings.shape[1] == 0:  # If clinical model failed
                text_embeddings = self._get_text_embeddings(procedure_texts, "sentence")
            if text_embeddings.shape[1] == 0:  # If sentence model failed
                text_embeddings = self._get_text_embeddings(procedure_texts, "fallback")
            
            embeddings['text_embeddings'] = text_embeddings
        
        # 2. Provider embeddings
        if 'provider_npi' in df.columns:
            provider_ids = df['provider_npi'].fillna('unknown').astype(str).tolist()
            provider_embeddings = self._create_provider_embeddings(provider_ids, fit=fit)
            embeddings['provider_embeddings'] = provider_embeddings
        
        # 3. Positional embeddings
        if self.config.use_positional_embeddings:
            positions = list(range(len(df)))
            positional_embeddings = self._create_positional_embeddings(positions)
            embeddings['positional_embeddings'] = positional_embeddings
        
        # 4. Temporal embeddings
        if self.config.use_temporal_embeddings and any(col.startswith('service_date_') for col in df.columns):
            temporal_embeddings = self._create_temporal_embeddings(df)
            embeddings['temporal_embeddings'] = temporal_embeddings
        
        # 5. Structured feature embeddings
        structured_embeddings = self._create_structured_embeddings(df)
        embeddings['structured_embeddings'] = structured_embeddings
        
        # 6. Composite embeddings
        if self.config.use_composite_embeddings:
            composite_embeddings = self._create_composite_embeddings(embeddings)
            embeddings['composite_embeddings'] = composite_embeddings
        
        # Normalize embeddings if requested
        if self.config.normalize_embeddings:
            embeddings = self._normalize_embeddings(embeddings)
        
        # Store embedding statistics
        self.embedding_stats = {
            'num_claims': len(df),
            'embedding_dimensions': {k: v.shape for k, v in embeddings.items()},
            'provider_vocabulary_size': len(self.provider_mapping) if hasattr(self, 'provider_mapping') else 0
        }
        
        logger.info(f"Generated embeddings for {len(df)} claims")
        return embeddings
    
    def _create_composite_embeddings(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Create composite embeddings by combining different embedding types"""
        composite_parts = []
        
        # Collect all available embeddings
        for key, emb in embeddings.items():
            if emb.numel() > 0:  # Check if tensor is not empty
                composite_parts.append(emb)
        
        if composite_parts:
            # Ensure all tensors have the same first dimension
            min_length = min(emb.shape[0] for emb in composite_parts)
            composite_parts = [emb[:min_length] for emb in composite_parts]
            
            # Concatenate along feature dimension
            composite_embedding = torch.cat(composite_parts, dim=1)
            return composite_embedding
        else:
            # Return zero tensor if no embeddings available
            return torch.zeros((1, self.config.embedding_dim))
    
    def _normalize_embeddings(self, embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Normalize embeddings using L2 normalization"""
        normalized_embeddings = {}
        
        for key, embedding in embeddings.items():
            if embedding.numel() > 0:
                normalized_embeddings[key] = torch.nn.functional.normalize(embedding, p=2, dim=1)
            else:
                normalized_embeddings[key] = embedding
        
        return normalized_embeddings
    
    def get_embedding_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor, 
                                metric: str = "cosine") -> float:
        """
        Calculate similarity between two embeddings
        
        Args:
            embedding1 (torch.Tensor): First embedding
            embedding2 (torch.Tensor): Second embedding
            metric (str): Similarity metric ("cosine", "euclidean", "dot")
            
        Returns:
            float: Similarity score
        """
        if metric == "cosine":
            similarity = torch.nn.functional.cosine_similarity(
                embedding1.unsqueeze(0), embedding2.unsqueeze(0)
            ).item()
        elif metric == "euclidean":
            similarity = -torch.norm(embedding1 - embedding2).item()  # Negative for similarity
        elif metric == "dot":
            similarity = torch.dot(embedding1.flatten(), embedding2.flatten()).item()
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
        
        return similarity
    
    def find_similar_claims(self, query_embedding: torch.Tensor, 
                           all_embeddings: torch.Tensor, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find most similar claims to a query embedding
        
        Args:
            query_embedding (torch.Tensor): Query embedding
            all_embeddings (torch.Tensor): All claim embeddings
            top_k (int): Number of similar claims to return
            
        Returns:
            List[Tuple[int, float]]: List of (index, similarity_score) tuples
        """
        # Calculate cosine similarities
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0), all_embeddings
        )
        
        # Get top-k similar claims
        top_similarities, top_indices = torch.topk(similarities, k=min(top_k, len(similarities)))
        
        results = [(idx.item(), sim.item()) for idx, sim in zip(top_indices, top_similarities)]
        return results
    
    def save_embeddings(self, embeddings: Dict[str, torch.Tensor], filepath: str):
        """
        Save embeddings to disk
        
        Args:
            embeddings (Dict[str, torch.Tensor]): Embeddings to save
            filepath (str): Path to save embeddings
        """
        embedding_data = {
            'embeddings': embeddings,
            'config': self.config,
            'provider_mapping': getattr(self, 'provider_mapping', {}),
            'embedding_stats': self.embedding_stats
        }
        
        torch.save(embedding_data, filepath)
        logger.info(f"Embeddings saved to {filepath}")
    
    def load_embeddings(self, filepath: str) -> Dict[str, torch.Tensor]:
        """
        Load embeddings from disk
        
        Args:
            filepath (str): Path to load embeddings from
            
        Returns:
            Dict[str, torch.Tensor]: Loaded embeddings
        """
        embedding_data = torch.load(filepath, map_location=self.device)
        
        self.config = embedding_data['config']
        self.provider_mapping = embedding_data.get('provider_mapping', {})
        self.embedding_stats = embedding_data.get('embedding_stats', {})
        
        logger.info(f"Embeddings loaded from {filepath}")
        return embedding_data['embeddings']

# Example usage and testing
if __name__ == "__main__":
    from sqlalchemy import create_engine
    
    # Load data and preprocessor
    db_path = "claims_data.db"
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    df = pd.read_sql("SELECT * FROM claims", engine)
    
    # Initialize preprocessor and process data
    preprocessing_config = PreprocessingConfig(
        max_sequence_length=256,
        tfidf_max_features=500
    )
    preprocessor = ClaimsPreprocessor(preprocessing_config)
    processed_data = preprocessor.preprocess_claims_data(df, fit=True)
    
    # Initialize embedding generator
    embedding_config = EmbeddingConfig(
        embedding_dim=768,
        use_positional_embeddings=True,
        use_provider_embeddings=True,
        use_temporal_embeddings=True,
        device="cpu"  # Use CPU for demo
    )
    
    embedding_generator = ClaimEmbeddingGenerator(embedding_config, preprocessor)
    
    # Generate embeddings
    embeddings = embedding_generator.generate_claim_embeddings(processed_data, fit=True)
    
    print("\n=== CLAIM EMBEDDING GENERATION COMPLETED ===")
    print(f"Generated embeddings for {embedding_generator.embedding_stats['num_claims']} claims")
    print("\nEmbedding dimensions:")
    for key, shape in embedding_generator.embedding_stats['embedding_dimensions'].items():
        print(f"  {key}: {shape}")
    
    # Test similarity
    if 'composite_embeddings' in embeddings and embeddings['composite_embeddings'].shape[0] > 1:
        emb1 = embeddings['composite_embeddings'][0]
        emb2 = embeddings['composite_embeddings'][1]
        similarity = embedding_generator.get_embedding_similarity(emb1, emb2)
        print(f"\nSimilarity between first two claims: {similarity:.4f}")
    
    # Save embeddings
    embedding_generator.save_embeddings(embeddings, 'claim_embeddings.pt')
    print("\nEmbeddings saved successfully!")

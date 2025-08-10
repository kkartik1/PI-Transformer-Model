import pandas as pd
import numpy as np
import spacy
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
import pickle
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuration class for preprocessing parameters"""
    max_sequence_length: int = 512
    embedding_dim: int = 768
    tokenizer_model: str = "distilbert-base-uncased"
    spacy_model: str = "en_core_web_sm"
    tfidf_max_features: int = 1000
    numerical_scaling_method: str = "minmax"  # "standard" or "minmax"
    date_feature_engineering: bool = True
    text_preprocessing: bool = True

class ClaimsPreprocessor:
    """
    Claims Preprocessor Component for Healthcare Claims Processing
    
    This component handles cleaning, normalization, tokenization, and embedding preparation
    for healthcare claims data using spaCy, scikit-learn, HuggingFace Tokenizers, and NumPy.
    """
    
    def __init__(self, config: PreprocessingConfig = None):
        """
        Initialize the Claims Preprocessor
        
        Args:
            config (PreprocessingConfig): Configuration parameters
        """
        self.config = config or PreprocessingConfig()
        
        # Initialize components
        self.nlp = None
        self.tokenizer = None
        self.label_encoders = {}
        self.scalers = {}
        self.tfidf_vectorizers = {}
        self.vocabulary_mappings = {}
        
        # Processing statistics
        self.processing_stats = {}
        
        # Initialize NLP components
        self._initialize_nlp_components()
        
        logger.info("Claims Preprocessor initialized successfully")
    
    def _initialize_nlp_components(self):
        """Initialize spaCy and HuggingFace components"""
        try:
            # Load spaCy model
            try:
                self.nlp = spacy.load(self.config.spacy_model)
                logger.info(f"Loaded spaCy model: {self.config.spacy_model}")
            except OSError:
                logger.warning(f"spaCy model {self.config.spacy_model} not found. Using blank model.")
                self.nlp = spacy.blank("en")
            
            # Load HuggingFace tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_model,
                max_length=self.config.max_sequence_length,
                truncation=True,
                padding=True
            )
            logger.info(f"Loaded HuggingFace tokenizer: {self.config.tokenizer_model}")
            
        except Exception as e:
            logger.error(f"Error initializing NLP components: {e}")
            # Fallback to basic tokenization
            self.tokenizer = None
            self.nlp = None
    
    def clean_text_fields(self, text: str) -> str:
        """
        Clean and normalize text fields
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or text == '' or str(text).lower() == 'nan':
            return ''
        
        text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Normalize case
        text = text.lower()
        
        # Remove special characters but keep medical coding characters
        text = re.sub(r'[^\w\s\.,\-]', '', text)
        
        # Handle common medical abbreviations
        medical_abbreviations = {
            'w/': 'with',
            'w/o': 'without',
            'pt': 'patient',
            'dx': 'diagnosis',
            'tx': 'treatment',
            'hx': 'history'
        }
        
        for abbrev, full in medical_abbreviations.items():
            text = text.replace(abbrev, full)
        
        return text
    
    def normalize_identifiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize identifier fields (patient_id, provider_npi, etc.)
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with normalized identifiers
        """
        df_normalized = df.copy()
        
        identifier_fields = ['patient_id', 'provider_npi', 'rendering_provider_npi', 
                           'policy_number', 'group_number', 'patient_control_number']
        
        for field in identifier_fields:
            if field in df_normalized.columns:
                # Remove leading/trailing whitespace
                df_normalized[field] = df_normalized[field].astype(str).str.strip()
                
                # Handle missing values
                df_normalized[field] = df_normalized[field].replace(['nan', 'NaN', 'None', ''], 'UNKNOWN')
                
                # Normalize format (remove special characters except digits)
                df_normalized[field] = df_normalized[field].apply(
                    lambda x: re.sub(r'[^\d]', '', str(x)) if str(x) != 'UNKNOWN' else x
                )
        
        logger.info("Identifier normalization completed")
        return df_normalized
    
    def extract_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract meaningful features from date fields
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with extracted date features
        """
        if not self.config.date_feature_engineering:
            return df
        
        df_features = df.copy()
        date_fields = ['submission_date', 'service_date', 'patient_dob']
        
        for field in date_fields:
            if field in df_features.columns:
                # Convert to datetime if not already
                df_features[field] = pd.to_datetime(df_features[field], errors='coerce')
                
                # Extract features
                df_features[f'{field}_year'] = df_features[field].dt.year
                df_features[f'{field}_month'] = df_features[field].dt.month
                df_features[f'{field}_day'] = df_features[field].dt.day
                df_features[f'{field}_dayofweek'] = df_features[field].dt.dayofweek
                df_features[f'{field}_quarter'] = df_features[field].dt.quarter
                
                # Calculate age for patient_dob
                if field == 'patient_dob':
                    current_date = datetime.now()
                    df_features['patient_age'] = (
                        current_date - df_features[field]
                    ).dt.days / 365.25
                
                # Calculate time differences
                if field == 'service_date' and 'submission_date' in df_features.columns:
                    df_features['days_to_submission'] = (
                        df_features['submission_date'] - df_features['service_date']
                    ).dt.days
        
        logger.info("Date feature extraction completed")
        return df_features
    
    def encode_categorical_fields(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical fields using label encoding
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit (bool): Whether to fit encoders or use existing ones
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical fields
        """
        df_encoded = df.copy()
        
        categorical_fields = [
            'patient_gender', 'patient_state', 'insurance_name', 'provider_specialty',
            'place_of_service', 'procedure_code', 'diagnosis_1', 'diagnosis_2',
            'diagnosis_3', 'diagnosis_4', 'modifier'
        ]
        
        for field in categorical_fields:
            if field in df_encoded.columns:
                # Handle missing values
                df_encoded[field] = df_encoded[field].fillna('UNKNOWN').astype(str)
                
                if fit:
                    if field not in self.label_encoders:
                        self.label_encoders[field] = LabelEncoder()
                    
                    # Fit and transform
                    df_encoded[f'{field}_encoded'] = self.label_encoders[field].fit_transform(
                        df_encoded[field]
                    )
                    
                    # Store vocabulary mapping
                    self.vocabulary_mappings[field] = dict(
                        zip(self.label_encoders[field].classes_, 
                            self.label_encoders[field].transform(self.label_encoders[field].classes_))
                    )
                else:
                    # Transform using existing encoder
                    if field in self.label_encoders:
                        # Handle unseen categories
                        known_categories = set(self.label_encoders[field].classes_)
                        df_encoded[field] = df_encoded[field].apply(
                            lambda x: x if x in known_categories else 'UNKNOWN'
                        )
                        df_encoded[f'{field}_encoded'] = self.label_encoders[field].transform(
                            df_encoded[field]
                        )
        
        logger.info("Categorical encoding completed")
        return df_encoded
    
    def scale_numerical_fields(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical fields
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit (bool): Whether to fit scalers or use existing ones
            
        Returns:
            pd.DataFrame: Dataframe with scaled numerical fields
        """
        df_scaled = df.copy()
        
        numerical_fields = ['charges', 'units', 'patient_age', 'days_to_submission']
        
        # Add date-derived numerical fields
        date_fields = ['submission_date', 'service_date', 'patient_dob']
        for field in date_fields:
            numerical_fields.extend([
                f'{field}_year', f'{field}_month', f'{field}_day',
                f'{field}_dayofweek', f'{field}_quarter'
            ])
        
        for field in numerical_fields:
            if field in df_scaled.columns:
                # Handle missing values
                df_scaled[field] = pd.to_numeric(df_scaled[field], errors='coerce').fillna(0)
                
                if fit:
                    # Initialize scaler
                    if self.config.numerical_scaling_method == "standard":
                        scaler = StandardScaler()
                    else:
                        scaler = MinMaxScaler()
                    
                    self.scalers[field] = scaler
                    
                    # Fit and transform
                    df_scaled[f'{field}_scaled'] = scaler.fit_transform(
                        df_scaled[[field]]
                    ).flatten()
                else:
                    # Transform using existing scaler
                    if field in self.scalers:
                        df_scaled[f'{field}_scaled'] = self.scalers[field].transform(
                            df_scaled[[field]]
                        ).flatten()
        
        logger.info("Numerical scaling completed")
        return df_scaled
    
    def create_text_embeddings(self, texts: List[str], field_name: str, fit: bool = True) -> np.ndarray:
        """
        Create embeddings for text fields using TF-IDF
        
        Args:
            texts (List[str]): List of text strings
            field_name (str): Name of the field for storing vectorizer
            fit (bool): Whether to fit vectorizer or use existing one
            
        Returns:
            np.ndarray: Text embeddings
        """
        # Clean texts
        cleaned_texts = [self.clean_text_fields(text) for text in texts]
        
        if fit:
            # Initialize TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=self.config.tfidf_max_features,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            
            self.tfidf_vectorizers[field_name] = vectorizer
            embeddings = vectorizer.fit_transform(cleaned_texts).toarray()
        else:
            # Transform using existing vectorizer
            if field_name in self.tfidf_vectorizers:
                embeddings = self.tfidf_vectorizers[field_name].transform(cleaned_texts).toarray()
            else:
                logger.warning(f"No fitted vectorizer found for {field_name}")
                embeddings = np.zeros((len(texts), self.config.tfidf_max_features))
        
        logger.info(f"Created embeddings for {field_name}: {embeddings.shape}")
        return embeddings
    
    def tokenize_for_transformer(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize texts for transformer models
        
        Args:
            texts (List[str]): List of text strings
            
        Returns:
            Dict[str, torch.Tensor]: Tokenized inputs
        """
        if self.tokenizer is None:
            logger.warning("HuggingFace tokenizer not available")
            return {}
        
        # Clean texts
        cleaned_texts = [self.clean_text_fields(text) for text in texts]
        
        # Tokenize
        tokenized = self.tokenizer(
            cleaned_texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_sequence_length,
            return_tensors='pt'
        )
        
        logger.info(f"Tokenized {len(texts)} texts for transformer input")
        return tokenized
    
    def extract_named_entities(self, texts: List[str]) -> List[Dict]:
        """
        Extract named entities from text using spaCy
        
        Args:
            texts (List[str]): List of text strings
            
        Returns:
            List[Dict]: Named entities for each text
        """
        if self.nlp is None:
            logger.warning("spaCy model not available")
            return [{}] * len(texts)
        
        entities_list = []
        
        for text in texts:
            cleaned_text = self.clean_text_fields(text)
            doc = self.nlp(cleaned_text)
            
            entities = {
                'persons': [ent.text for ent in doc.ents if ent.label_ == 'PERSON'],
                'organizations': [ent.text for ent in doc.ents if ent.label_ == 'ORG'],
                'dates': [ent.text for ent in doc.ents if ent.label_ == 'DATE'],
                'money': [ent.text for ent in doc.ents if ent.label_ == 'MONEY'],
                'quantities': [ent.text for ent in doc.ents if ent.label_ in ['QUANTITY', 'CARDINAL']]
            }
            
            entities_list.append(entities)
        
        logger.info(f"Extracted named entities from {len(texts)} texts")
        return entities_list
    
    def create_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite features from multiple fields
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with composite features
        """
        df_composite = df.copy()
        
        # Patient-Provider combination
        if 'patient_id' in df_composite.columns and 'provider_npi' in df_composite.columns:
            df_composite['patient_provider_combo'] = (
                df_composite['patient_id'].astype(str) + '_' + 
                df_composite['provider_npi'].astype(str)
            )
        
        # Procedure-Diagnosis combination
        if 'procedure_code' in df_composite.columns and 'diagnosis_1' in df_composite.columns:
            df_composite['procedure_diagnosis_combo'] = (
                df_composite['procedure_code'].astype(str) + '_' + 
                df_composite['diagnosis_1'].astype(str)
            )
        
        # Charges per unit
        if 'charges' in df_composite.columns and 'units' in df_composite.columns:
            df_composite['charge_per_unit'] = np.where(
                df_composite['units'] > 0,
                df_composite['charges'] / df_composite['units'],
                0
            )
        
        logger.info("Composite features created")
        return df_composite
    
    def preprocess_claims_data(self, df: pd.DataFrame, fit: bool = True) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline for claims data
        
        Args:
            df (pd.DataFrame): Input claims dataframe
            fit (bool): Whether to fit preprocessors or use existing ones
            
        Returns:
            Dict[str, Any]: Preprocessed data and metadata
        """
        logger.info("Starting complete claims preprocessing pipeline")
        
        # Start with a copy of the data
        processed_df = df.copy()
        
        # Step 1: Normalize identifiers
        processed_df = self.normalize_identifiers(processed_df)
        
        # Step 2: Extract date features
        processed_df = self.extract_date_features(processed_df)
        
        # Step 3: Create composite features
        processed_df = self.create_composite_features(processed_df)
        
        # Step 4: Encode categorical fields
        processed_df = self.encode_categorical_fields(processed_df, fit=fit)
        
        # Step 5: Scale numerical fields
        processed_df = self.scale_numerical_fields(processed_df, fit=fit)
        
        # Step 6: Create text embeddings for procedure descriptions
        if 'procedure_description' in processed_df.columns:
            procedure_embeddings = self.create_text_embeddings(
                processed_df['procedure_description'].tolist(),
                'procedure_description',
                fit=fit
            )
        else:
            procedure_embeddings = np.array([])
        
        # Step 7: Tokenize text for transformer models
        if 'procedure_description' in processed_df.columns:
            transformer_tokens = self.tokenize_for_transformer(
                processed_df['procedure_description'].tolist()
            )
        else:
            transformer_tokens = {}
        
        # Step 8: Extract named entities
        if 'procedure_description' in processed_df.columns:
            named_entities = self.extract_named_entities(
                processed_df['procedure_description'].tolist()
            )
        else:
            named_entities = []
        
        # Collect processing statistics
        self.processing_stats = {
            'total_records': len(processed_df),
            'categorical_fields_encoded': len(self.label_encoders),
            'numerical_fields_scaled': len(self.scalers),
            'text_fields_vectorized': len(self.tfidf_vectorizers),
            'embedding_dimensions': {
                'procedure_description': procedure_embeddings.shape[1] if procedure_embeddings.size > 0 else 0
            }
        }
        
        result = {
            'processed_dataframe': processed_df,
            'text_embeddings': {
                'procedure_description': procedure_embeddings
            },
            'transformer_tokens': transformer_tokens,
            'named_entities': named_entities,
            'preprocessing_stats': self.processing_stats,
            'label_encoders': self.label_encoders,
            'scalers': self.scalers,
            'tfidf_vectorizers': self.tfidf_vectorizers,
            'vocabulary_mappings': self.vocabulary_mappings
        }
        
        logger.info("Claims preprocessing pipeline completed successfully")
        return result
    
    def save_preprocessors(self, filepath: str):
        """
        Save fitted preprocessors to disk
        
        Args:
            filepath (str): Path to save the preprocessors
        """
        preprocessor_data = {
            'label_encoders': self.label_encoders,
            'scalers': self.scalers,
            'tfidf_vectorizers': self.tfidf_vectorizers,
            'vocabulary_mappings': self.vocabulary_mappings,
            'config': self.config,
            'processing_stats': self.processing_stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        
        logger.info(f"Preprocessors saved to {filepath}")
    
    def load_preprocessors(self, filepath: str):
        """
        Load fitted preprocessors from disk
        
        Args:
            filepath (str): Path to load the preprocessors from
        """
        with open(filepath, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        self.label_encoders = preprocessor_data['label_encoders']
        self.scalers = preprocessor_data['scalers']
        self.tfidf_vectorizers = preprocessor_data['tfidf_vectorizers']
        self.vocabulary_mappings = preprocessor_data['vocabulary_mappings']
        self.config = preprocessor_data['config']
        self.processing_stats = preprocessor_data['processing_stats']
        
        logger.info(f"Preprocessors loaded from {filepath}")

# Example usage and testing
if __name__ == "__main__":
    from sqlalchemy import create_engine

    # Path to the SQLite database created by ClaimsDataLoader
    db_path = "claims_data.db"  # Update if your path differs
    engine = create_engine(f"sqlite:///{db_path}", echo=False)

    # Load claims data from SQLite
    query = "SELECT * FROM claims"
    df = pd.read_sql(query, engine)
    
    # Initialize preprocessor
    config = PreprocessingConfig(
        max_sequence_length=256,
        tfidf_max_features=500
    )
    
    preprocessor = ClaimsPreprocessor(config)
    
    # Run preprocessing pipeline
    result = preprocessor.preprocess_claims_data(df, fit=True)
    
    print("\n=== CLAIMS PREPROCESSING COMPLETED ===")
    print(f"Processed {result['preprocessing_stats']['total_records']} records")
    print(f"Categorical fields encoded: {result['preprocessing_stats']['categorical_fields_encoded']}")
    print(f"Numerical fields scaled: {result['preprocessing_stats']['numerical_fields_scaled']}")
    print(f"Text embeddings shape: {result['text_embeddings']['procedure_description'].shape}")
    
    print("\n=== SAMPLE PROCESSED DATA ===")
    processed_cols = [col for col in result['processed_dataframe'].columns if 'encoded' in col or 'scaled' in col]
    print(result['processed_dataframe'][processed_cols[:5]].head())
    
    # Save preprocessors
    preprocessor.save_preprocessors('claims_preprocessors.pkl')
    print("\nPreprocessors saved successfully!")

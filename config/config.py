import os
from dataclasses import dataclass

@dataclass
class Config:
    GROQ_API_KEY = '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'
    MODEL_NAME = "llama3-70b-8192"
    
    # LLM parameters
    TEMPERATURE = 0.5
    MAX_TOKENS = 1024
    TOP_P = 1
    
    # Text processing
    CHUNK_SIZE = 2000
    CHUNK_OVERLAP = 100
    
    # Embeddings model
    EMBEDDINGS_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # Logging
    LOG_FILE = "document_processor.log"
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
from pydantic_settings import BaseSettings # type: ignore
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    API_HOST: str = "localhost"
    API_PORT: int = 8000
    DEBUG: bool = True
    
    # Model configurations
    GLINER_MODEL: str = "urchade/gliner_base"
    ZERO_SHOT_MODEL: str = "facebook/bart-large-mnli"
    
    # Security
    SECRET_KEY: str
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
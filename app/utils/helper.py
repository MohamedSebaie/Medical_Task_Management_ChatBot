from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import re
import json
import logging

logger = logging.getLogger(__name__)

class TextPreprocessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize input text"""
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('\n', ' ')
        return text

    @staticmethod
    def extract_dates(text: str) -> List[str]:
        """Extract date patterns from text"""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            dates.extend(matches)
        return dates

class MedicalDataValidator:
    @staticmethod
    def validate_dosage(dosage: str) -> bool:
        """Validate medication dosage format"""
        dosage_pattern = r'^\d+(?:\.\d+)?(?:mg|g|ml|mcg)$'
        return bool(re.match(dosage_pattern, dosage.lower()))

    @staticmethod
    def validate_frequency(frequency: str) -> bool:
        """Validate medication frequency format"""
        valid_frequencies = [
            r'^\d+\s+times?\s+(?:per|a)\s+day$',
            r'^every\s+\d+\s+hours?$',
            r'^daily$',
            r'^weekly$',
            r'^monthly$'
        ]
        
        return any(re.match(pattern, frequency.lower()) for pattern in valid_frequencies)

class ContextManager:
    def __init__(self):
        self.context = {}
        self.context_expiry = {}
        self.DEFAULT_EXPIRY = timedelta(minutes=30)

    def add_to_context(
        self,
        key: str,
        value: Any,
        expiry: Optional[timedelta] = None
    ):
        """Add information to context with expiry"""
        self.context[key] = value
        self.context_expiry[key] = datetime.now() + (
            expiry or self.DEFAULT_EXPIRY
        )

    def get_from_context(self, key: str) -> Optional[Any]:
        """Get information from context if not expired"""
        if key not in self.context:
            return None
            
        if datetime.now() > self.context_expiry[key]:
            del self.context[key]
            del self.context_expiry[key]
            return None
            
        return self.context[key]

    def clear_expired(self):
        """Clear expired context entries"""
        current_time = datetime.now()
        expired_keys = [
            key for key, expiry in self.context_expiry.items()
            if current_time > expiry
        ]
        
        for key in expired_keys:
            del self.context[key]
            del self.context_expiry[key]

class DataCache:
    def __init__(self, cache_file: str = "data/cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """Load cache from file"""
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_cache(self):
        """Save cache to file"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        return self.cache.get(key)

    def set(self, key: str, value: Any):
        """Set value in cache"""
        self.cache[key] = value
        self._save_cache()

class Logger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._setup_logger()

    def _setup_logger(self):
        """Setup logger configuration"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_request(self, request_data: Dict):
        """Log API request"""
        self.logger.info(f"Incoming request: {request_data}")

    def log_response(self, response_data: Dict):
        """Log API response"""
        self.logger.info(f"Outgoing response: {response_data}")

    def log_error(self, error: Exception, context: Optional[Dict] = None):
        """Log error with context"""
        error_msg = f"Error: {str(error)}"
        if context:
            error_msg += f" Context: {context}"
        self.logger.error(error_msg, exc_info=True)
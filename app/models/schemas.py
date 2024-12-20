from pydantic import BaseModel, Field # type: ignore
from typing import Dict, List, Optional, Any
from datetime import datetime

class EntityBase(BaseModel):
    text: str
    type: str
    confidence: float = Field(ge=0.0, le=1.0)

class IntentResponse(BaseModel):
    primary_intent: str
    confidence: float

class TemporalInfo(BaseModel):
    dates: List[str] = []
    times: List[str] = []
    patterns: List[str] = []

class ProcessedResponse(BaseModel):
    intent: IntentResponse
    entities: Dict[str, List[EntityBase]]
    temporal_info: TemporalInfo
    processed_at: datetime

class CommandResponse(BaseModel):
    success: bool
    result: Optional[ProcessedResponse] = None
    error: Optional[str] = None
from pydantic import BaseModel, Field # type: ignore
from typing import Dict, List, Optional, Any
from datetime import datetime

class Entity(BaseModel):
    text: str
    type: str
    span: Optional[tuple] = None
    confidence: float = Field(ge=0.0, le=1.0)

class Intent(BaseModel):
    primary_intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    all_intents: List[Dict[str, Any]]

class TemporalInfo(BaseModel):
    dates: List[str] = []
    times: List[str] = []
    durations: List[str] = []
    frequencies: List[str] = []
    patterns: List[str] = []

class ProcessedResult(BaseModel):
    intent: Intent
    entities: Dict[str, List[Entity]]
    temporal_info: TemporalInfo
    raw_text: str
    processed_at: datetime

class ConversationContext(BaseModel):
    current_patient: Optional[Dict[str, Any]]
    current_medical_info: Optional[List[Dict[str, Any]]]
    last_mentioned_date: Optional[str]

class CommandRequest(BaseModel):
    text: str
    conversation_history: Optional[List[str]] = []

class CommandResponse(BaseModel):
    success: bool
    result: Dict[str, Any]
    error: Optional[str] = None
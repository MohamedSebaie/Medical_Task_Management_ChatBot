from asyncio.log import logger
from fastapi import FastAPI, HTTPException, Depends # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from .services.nlp_pipeline import MedicalNLPPipeline
from .config import Settings, get_settings
from typing import Dict, Any, List, Optional
from pydantic import BaseModel # type: ignore
import logging

app = FastAPI(
    title="Medical NLP Bot",
    description="API for processing medical text and commands",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize NLP pipeline
nlp_pipeline = MedicalNLPPipeline()

class CommandRequest(BaseModel):
    text: str
    conversation_history: Optional[List[str]] = []

class CommandResponse(BaseModel):
    success: bool
    result: Dict[str, Any]
    error: Optional[str] = None

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/api/process")
async def process_command(request: CommandRequest) -> Dict[str, Any]:
    try:
        result = nlp_pipeline.process_text(request.text)
        
        # Debug print
        print("Processing result:", result)
        
        response = {
            "success": True,
            "result": {
                "intent": {
                    "primary_intent": result["intent"]["primary_intent"],
                    "confidence": result["intent"]["confidence"]
                },
                "entities": {
                    k: [{"text": e["text"], "type": e["type"], "confidence": e["confidence"]} 
                       for e in v] 
                    for k, v in result["entities"].items() if v
                },
                "temporal_info": result["temporal_info"],
                "processed_at": result["processed_at"]
            }
        }
        
        # Add follow-up question if present
        if "follow_up_question" in result:
            response["result"]["follow_up_question"] = result["follow_up_question"]
            
        # Add medication validation if present
        if "medication_validation" in result:
            response["result"]["medication_validation"] = result["medication_validation"]
            
        return response
    except Exception as e:
        logger.error(f"Error processing command: {str(e)}")
        return {"success": False, "error": str(e)}

@app.get("/api/entities")
async def get_supported_entities():
    return {
        "entities": nlp_pipeline.medical_entities,
        "intents": nlp_pipeline.intent_labels
    }
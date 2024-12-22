from asyncio.log import logger
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from .services.nlp_pipeline import MedicalNLPPipeline
from .services.llm_pipeline import LLMMedicalPipeline
from .config import Settings, get_settings
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
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

# Initialize both pipelines
transformer_pipeline = MedicalNLPPipeline()
llm_pipeline = LLMMedicalPipeline()

class CommandRequest(BaseModel):
    text: str
    conversation_history: Optional[List[str]] = []
    pipeline_type: str = "transformer"

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
        # Select pipeline based on request
        pipeline = llm_pipeline if request.pipeline_type.lower() == "llm" else transformer_pipeline
        
        # Process text with selected pipeline
        result = pipeline.process_text(request.text)
        
        # Debug print
        print("Processing result:", result)
        
        # Extract temporal information from entities if it exists there
        temporal_info = []
        if "entities" in result and "temporal_info" in result["entities"]:
            temporal_info = result["entities"]["temporal_info"]
        
        response = {
            "success": True,
            "result": {
                "intent": {
                    "primary_intent": result["intent"]["primary_intent"],
                    "confidence": result["intent"]["confidence"]
                },
                "entities": {
                    k: [{"text": e["text"], "type": e["type"], "confidence": e.get("confidence", 1.0)} 
                       for e in v] 
                    for k, v in result["entities"].items() if v and k != "temporal_info"
                },
                "temporal_info": temporal_info,
                "simplified_format": result.get("simplified_format", {
                    "intent": result["intent"]["primary_intent"],
                    "entities": {
                        "patient": next((e["text"] for e in result["entities"].get("patient_info", []) 
                                      if e["type"] == "patient"), None),
                        "gender": next((e["text"] for e in result["entities"].get("patient_info", []) 
                                     if e["type"] == "gender"), None),
                        "age": next((e["text"] for e in result["entities"].get("temporal_info", []) 
                                  if e["type"] == "age"), None),
                        "condition": next((e["text"] for e in result["entities"].get("medical_info", []) 
                                       if e["type"] == "condition"), None)
                    }
                }),
                "processed_at": result["processed_at"],
                "pipeline_type": request.pipeline_type
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
        return {
            "success": False,
            "error": str(e),
            "result": {
                "intent": {"primary_intent": "unknown", "confidence": 0.0},
                "entities": {},
                "temporal_info": [],
                "simplified_format": {
                    "intent": "unknown",
                    "entities": {
                        "patient": None,
                        "gender": None,
                        "age": None,
                        "condition": None
                    }
                },
                "processed_at": result.get("processed_at", ""),
                "pipeline_type": request.pipeline_type
            }
        }

@app.get("/api/entities")
async def get_supported_entities():
    return {
        "transformer_entities": transformer_pipeline.medical_entities,
        "transformer_intents": transformer_pipeline.intent_labels,
        "llm_capabilities": {
            "entities": [
                "patient_info", "medical_info", "temporal_info", "location_info"
            ],
            "intents": [
                "add_patient", "assign_medication", "schedule_followup",
                "update_record", "query_info", "check_vitals", 
                "order_test", "review_results"
            ]
        }
    }
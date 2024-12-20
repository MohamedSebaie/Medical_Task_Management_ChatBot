from fastapi import FastAPI, HTTPException, Depends # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from .services.nlp_pipeline import MedicalNLPPipeline
from .config import Settings, get_settings
from typing import Dict, Any, List, Optional
from pydantic import BaseModel # type: ignore

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

@app.post("/api/process", response_model=CommandResponse)
async def process_command(
    request: CommandRequest,
    settings: Settings = Depends(get_settings)
):
    try:
        result = nlp_pipeline.process_text(request.text)
        
        # Process conversation history if provided
        if request.conversation_history:
            context = nlp_pipeline.process_conversation(
                request.conversation_history + [request.text]
            )
            result["conversation_context"] = context
        
        return CommandResponse(
            success=True,
            result=result
        )
    except Exception as e:
        return CommandResponse(
            success=False,
            result={},
            error=str(e)
        )

@app.get("/api/entities")
async def get_supported_entities():
    return {
        "entities": nlp_pipeline.medical_entities,
        "intents": nlp_pipeline.intent_labels
    }
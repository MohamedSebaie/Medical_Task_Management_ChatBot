from typing import Union
from app.services.nlp_pipeline import MedicalNLPPipeline
from app.services.llm_pipeline import LLMMedicalPipeline

class PipelineFactory:
    @staticmethod
    def create_pipeline(pipeline_type: str) -> Union[MedicalNLPPipeline, LLMMedicalPipeline]:
        if pipeline_type.lower() == "transformer":
            return MedicalNLPPipeline()
        elif pipeline_type.lower() == "llm":
            return LLMMedicalPipeline()
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
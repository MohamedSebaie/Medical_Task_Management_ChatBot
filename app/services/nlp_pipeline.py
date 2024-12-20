from gliner import GLiNER # type: ignore
from transformers import pipeline # type: ignore
from typing import Dict, List, Any, Optional
import spacy # type: ignore
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MedicalNLPPipeline:
    def __init__(self):
        try:
            # Initialize GLiNER model
            self.gliner = GLiNER.from_pretrained("urchade/gliner_base")
            
            # Initialize zero-shot classifier
            self.zero_shot = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1  # CPU
            )
            
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            
            # Initialize entity types and intents
            self._initialize_labels()
            
        except Exception as e:
            logger.error(f"Error initializing NLP pipeline: {str(e)}")
            raise

    def _initialize_labels(self):
        """Initialize entity types and intent labels"""
        self.medical_entities = [
            "patient", "doctor", "medication", "dosage",
            "frequency", "condition", "symptom", "procedure",
            "test", "date", "time", "duration", "facility",
            "department", "vital_sign", "lab_result"
        ]
        
        self.intent_labels = [
            "add_patient", "assign_medication", 
            "schedule_followup", "update_record",
            "query_info", "check_vitals", 
            "order_test", "review_results"
        ]
        
        self.temporal_patterns = [
            "daily", "twice", "weekly", "monthly",
            "every", "times a day", "hours"
        ]

    def process_text(self, text: str) -> Dict[str, Any]:
        """Process medical text through both GLiNER and zero-shot classification"""
        try:
            # Get intent using zero-shot classification
            intent_result = self._classify_intent(text)
            
            # Get entities using GLiNER
            entities = self.gliner.predict_entities(
                text,
                self.medical_entities
            )
            
            # Process with spaCy for additional linguistic features
            doc = self.nlp(text)
            
            # Extract temporal information
            temporal_info = self._extract_temporal_info(doc)
            
            # Structure the results
            structured_entities = self._structure_entities(entities)
            
            return {
                "intent": intent_result,
                "entities": structured_entities,
                "temporal_info": temporal_info,
                "raw_text": text,
                "processed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            raise

    def _classify_intent(self, text: str) -> Dict[str, Any]:
        """Classify intent using zero-shot classification"""
        hypothesis_template = "This is a request to {}."
        
        result = self.zero_shot(
            text,
            self.intent_labels,
            hypothesis_template=hypothesis_template,
            multi_label=True
        )
        
        return {
            "primary_intent": result["labels"][0],
            "confidence": result["scores"][0],
            "all_intents": [
                {"intent": label, "score": score}
                for label, score in zip(result["labels"], result["scores"])
            ]
        }

    def _structure_entities(self, entities: List[Dict]) -> Dict[str, List[Dict]]:
        """Structure extracted entities by category"""
        structured = {
            "patient_info": [],
            "medical_info": [],
            "temporal_info": [],
            "location_info": [],
            "other": []
        }
        
        category_mapping = {
            "patient": "patient_info",
            "doctor": "patient_info",
            "medication": "medical_info",
            "dosage": "medical_info",
            "frequency": "medical_info",
            "condition": "medical_info",
            "symptom": "medical_info",
            "procedure": "medical_info",
            "test": "medical_info",
            "date": "temporal_info",
            "time": "temporal_info",
            "duration": "temporal_info",
            "facility": "location_info",
            "department": "location_info",
        }
        
        for entity in entities:
            category = category_mapping.get(entity["label"], "other")
            structured[category].append({
                "text": entity["text"],
                "type": entity["label"],
                "span": entity.get("span", None),
                "confidence": entity.get("score", 1.0)
            })
        
        return structured

    def _extract_temporal_info(self, doc) -> Dict[str, Any]:
        """Extract temporal information from text"""
        temporal_info = {
            "dates": [],
            "times": [],
            "durations": [],
            "frequencies": [],
            "patterns": []
        }
        
        # Use spaCy for date recognition
        for ent in doc.ents:
            if ent.label_ in ["DATE", "TIME"]:
                temporal_info["dates"].append(ent.text)
        
        # Extract patterns
        text_lower = doc.text.lower()
        for pattern in self.temporal_patterns:
            if pattern in text_lower:
                temporal_info["patterns"].append(pattern)
        
        return temporal_info

    def process_conversation(
        self,
        conversation: List[str]
    ) -> List[Dict[str, Any]]:
        """Process a conversation history"""
        results = []
        context = {}
        
        for utterance in conversation:
            current_result = self.process_text(utterance)
            self._update_context(context, current_result)
            current_result["context"] = context.copy()
            results.append(current_result)
        
        return results

    def _update_context(
        self,
        context: Dict[str, Any],
        current_result: Dict[str, Any]
    ):
        """Update conversation context with new information"""
        entities = current_result["entities"]
        
        if entities["patient_info"]:
            context["current_patient"] = entities["patient_info"][0]
        
        if entities["medical_info"]:
            context["current_medical_info"] = entities["medical_info"]
        
        if current_result["temporal_info"]["dates"]:
            context["last_mentioned_date"] = (
                current_result["temporal_info"]["dates"][0]
            )
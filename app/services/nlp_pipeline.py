import logging
from typing import Dict, List, Any, Optional
from transformers import pipeline # type: ignore
from gliner import GLiNER # type: ignore
import spacy # type: ignore
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class MedicalNLPPipeline:
    def __init__(self):
        try:
            # Initialize GLiNER
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
            
            logger.info("Successfully initialized NLP pipeline")
            
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

    def _extract_entities_with_gliner(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using GLiNER"""
        try:
            # Get entities using GLiNER's predict_entities method
            entities = self.gliner.predict_entities(
                text,
                self.medical_entities
            )
            return entities
        except Exception as e:
            logger.error(f"Error in GLiNER extraction: {str(e)}")
            return []

    def _structure_entities(self, entities: List[Dict]) -> Dict[str, List[Dict]]:
        """Structure entities by category"""
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
            "age": "patient_info",  # Make sure age goes to patient_info
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
        
        # Process GLiNER entities
        for entity in entities:
            category = category_mapping.get(entity["label"], "other")
            structured[category].append({
                "text": entity["text"],
                "type": entity["label"],
                "span": entity.get("span", None),
                "confidence": entity.get("score", 1.0)
            })
        
        return structured

    def _extract_temporal_info(self, doc) -> Dict[str, List[str]]:
        """Extract temporal information from text"""
        temporal_info = {
            "dates": [],
            "times": [],
            "patterns": [],
            "age": []  # Added age category
        }

        # Extract pure dates (excluding age)
        date_pattern = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'
        matches = re.finditer(date_pattern, doc.text, re.IGNORECASE)
        temporal_info["dates"].extend(match.group() for match in matches)

        # Extract age separately
        age_pattern = r'\b(\d+)\s*(?:years?\s*old|y/?o)\b'
        age_matches = re.finditer(age_pattern, doc.text, re.IGNORECASE)
        temporal_info["age"].extend(match.group() for match in age_matches)

        # Extract times
        time_pattern = r'\b(?:1[0-2]|0?[1-9])(?::[0-5][0-9])?\s*(?:AM|PM)\b|\b(?:[01]?[0-9]|2[0-3]):[0-5][0-9]\b'
        matches = re.finditer(time_pattern, doc.text, re.IGNORECASE)
        temporal_info["times"].extend(match.group() for match in matches)

        # Extract patterns
        text_lower = doc.text.lower()
        for pattern in self.temporal_patterns:
            if pattern in text_lower:
                temporal_info["patterns"].append(pattern)

        return temporal_info

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
            "confidence": result["scores"][0]
        }

    def process_text(self, text: str) -> Dict[str, Any]:
        """Process medical text through GLiNER and intent classification"""
        try:
            # Get intent using zero-shot classification
            intent_result = self._classify_intent(text)
            
            # Get entities using GLiNER
            entities = self._extract_entities_with_gliner(text)
            
            # Process with spaCy for temporal features
            doc = self.nlp(text)
            
            # Extract temporal information
            temporal_info = self._extract_temporal_info(doc)
            
            # Structure the entities
            structured_entities = self._structure_entities(entities)
            
            # Add age to patient_info if found
            if temporal_info.get("age"):
                structured_entities["patient_info"].append({
                    "text": temporal_info["age"][0],
                    "type": "age",
                    "confidence": 0.99
                })
                # Remove age from temporal_info
                temporal_info.pop("age")
            
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
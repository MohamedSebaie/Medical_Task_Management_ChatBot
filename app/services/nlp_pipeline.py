import logging
from typing import Dict, List, Any, Optional
from app.services.medication_validator import MedicationValidator
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
            "department", "vital_sign", "lab_result", "gender",  # Added gender
            "demographics", "patient_gender"  # Additional gender-related entities
        ]
        
        self.gender_patterns = {
            "male": ["male", "m", "man", "boy", "gentleman"],
            "female": ["female", "f", "woman", "girl", "lady"],
            "other": ["other", "non-binary", "transgender", "prefer not to say"]
        }

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

    def _extract_gender_with_pattern(self, text: str) -> Dict[str, Any]:
        """Extract gender using pattern matching"""
        text_lower = text.lower()
        
        for gender, patterns in self.gender_patterns.items():
            for pattern in patterns:
                if f" {pattern} " in f" {text_lower} ":
                    return {
                        "label": "gender",
                        "text": gender,
                        "score": 0.95,
                        "method": "pattern"
                    }
        return None

    def _extract_entities_with_gliner(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using GLiNER"""
        try:
            # Get entities using GLiNER's predict_entities method
            entities = self.gliner.predict_entities(
                text,
                self.medical_entities
            )
            
            # Extract gender using pattern matching
            pattern_gender = self._extract_gender_with_pattern(text)
            
            # Check if GLiNER found any gender entities
            gliner_gender = next(
                (e for e in entities if e["label"] in ["gender", "patient_gender", "demographics"]), 
                None
            )
            
            # Combine gender information
            if pattern_gender and gliner_gender:
                # Both methods found gender - use the one with higher confidence
                if pattern_gender["score"] > gliner_gender.get("score", 0):
                    entities = [e for e in entities if e["label"] not in ["gender", "patient_gender", "demographics"]]
                    entities.append(pattern_gender)
            elif pattern_gender:
                # Only pattern matching found gender
                entities.append(pattern_gender)
            
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
            "age": "patient_info",
            "gender": "patient_info",  # Added gender mapping
            "patient_gender": "patient_info",  # Additional gender mapping
            "demographics": "patient_info",  # Additional demographics mapping
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
        
        # Process entities
        for entity in entities:
            category = category_mapping.get(entity["label"], "other")
            structured[category].append({
                "text": entity["text"],
                "type": entity["label"],
                "span": entity.get("span", None),
                "confidence": entity.get("score", 1.0),
                "method": entity.get("method", "gliner")  # Track detection method
            })
        
        return structured

    def _extract_temporal_info(self, doc) -> Dict[str, List[str]]:
        """Extract temporal information from text"""
        temporal_info = {
            "dates": [],
            "times": [],
            "patterns": [],
            "age": []
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

    def generate_follow_up_question(self, result: Dict[str, Any]) -> Optional[str]:
        """Generate follow-up questions based on intent and entities"""
        intent = result["intent"]["primary_intent"]
        entities = result["entities"]
        
        # Follow-up questions based on intent
        if intent == "assign_medication":
            # Check if we have all needed information for medication
            med_info = next((e for e in entities.get("medical_info", []) 
                            if e["type"] == "medication"), None)
            dosage_info = next((e for e in entities.get("medical_info", []) 
                            if e["type"] == "dosage"), None)
            frequency_info = next((e for e in entities.get("medical_info", []) 
                                if e["type"] == "frequency"), None)
            
            if not med_info:
                return "What medication would you like to prescribe?"
            if not dosage_info:
                return f"What is the dosage for {med_info['text']}?"
            if not frequency_info:
                return f"How often should the patient take {med_info['text']}?"
                
        elif intent == "add_patient":
            # Check if we have complete patient information
            patient_entities = entities.get("patient_info", [])
            has_name = any(e["type"] == "patient" for e in patient_entities)
            has_age = any(e["type"] == "age" for e in patient_entities)
            has_gender = any(e["type"] == "gender" for e in patient_entities)
            
            if not has_name:
                return "What is the patient's name?"
            if not has_age:
                return "What is the patient's age?"
            if not has_gender:
                return "What is the patient's gender?"
                
        elif intent == "schedule_followup":
            temporal_info = result["temporal_info"]
            if not temporal_info.get("dates"):
                return "On which date would you like to schedule the follow-up?"
            if not temporal_info.get("times"):
                return "At what time should the follow-up be scheduled?"
                
        return None

    def process_text(self, text: str) -> Dict[str, Any]:
        """Process medical text through GLiNER and intent classification"""
        try:
            # Get intent using zero-shot classification
            intent_result = self._classify_intent(text)
            
            # Get entities using GLiNER and pattern matching
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
                temporal_info.pop("age")
            
            result = {
            "intent": intent_result,
            "entities": structured_entities,
            "temporal_info": temporal_info,
            "raw_text": text,
            "processed_at": datetime.now().isoformat()
            }
            
            # Add medication validation and follow-up questions
            if intent_result["primary_intent"] == "assign_medication":
                med_entities = structured_entities.get("medical_info", [])
                medication = next((e["text"] for e in med_entities if e["type"] == "medication"), None)
                dosage = next((e["text"] for e in med_entities if e["type"] == "dosage"), None)
                frequency = next((e["text"] for e in med_entities if e["type"] == "frequency"), None)
                
                # Validate medication if present
                if medication:
                    validator = MedicationValidator()
                    is_valid, message = validator.validate_medication(medication, dosage, frequency)
                    result["medication_validation"] = {
                        "is_valid": is_valid,
                        "message": message
                    }
                
                # Generate appropriate follow-up question
                if not dosage:
                    result["follow_up_question"] = f"What is the dosage for {medication}?"
                elif not frequency:
                    result["follow_up_question"] = f"How often should {medication} be taken?"

            elif intent_result["primary_intent"] == "add_patient":
                patient_entities = structured_entities.get("patient_info", [])
                if not any(e["type"] == "age" for e in patient_entities):
                    result["follow_up_question"] = "What is the patient's age?"
                elif not any(e["type"] == "gender" for e in patient_entities):
                    result["follow_up_question"] = "What is the patient's gender?"

            return result
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            raise
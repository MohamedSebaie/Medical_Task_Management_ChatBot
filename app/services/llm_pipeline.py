from groq import Groq # type: ignore
from config.config import Config # type: ignore
from typing import Dict, Any, List
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)

class LLMMedicalPipeline:
    def __init__(self):
        self.client = Groq(api_key=Config.GROQ_API_KEY)
        self.entity_extraction_prompt = """
        Extract medical entities from the following text. Categorize them into:
        - patient_info (patient name, age, gender)
        - medical_info (medication, dosage, frequency, condition, symptom)
        - temporal_info (dates, times, duration)
        - location_info (facility, department)

        Text: {text}

        Return the results in the following JSON format:
        {
            "entities": {
                "patient_info": [{"text": "extracted text", "type": "entity type", "confidence": 0.95}],
                "medical_info": [{"text": "extracted text", "type": "entity type", "confidence": 0.95}],
                "temporal_info": [{"text": "extracted text", "type": "entity type", "confidence": 0.95}],
                "location_info": [{"text": "extracted text", "type": "entity type", "confidence": 0.95}]
            }
        }
        """

        self.intent_classification_prompt = """
        Classify the intent of the following medical text into one of these categories:
        - add_patient
        - assign_medication
        - schedule_followup
        - update_record
        - query_info
        - check_vitals
        - order_test
        - review_results

        Text: {text}

        Return the result in the following JSON format:
        {
            "primary_intent": "intent_category",
            "confidence": 0.95
        }
        """

        self.medication_validation_prompt = """
        Validate the following medication details against standard medical guidelines:
        
        Medication: {medication}
        Dosage: {dosage}
        Frequency: {frequency}

        Check for:
        1. Valid medication name
        2. Appropriate dosage range
        3. Standard frequency patterns

        Return the result in the following JSON format:
        {
            "is_valid": true/false,
            "message": "validation message",
            "follow_up_question": "if needed",
            "validation_step": "medication_name/dosage/frequency/complete"
        }
        """

    def _call_llm(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=Config.MODEL_NAME,
                temperature=0.2,  # Lower temperature for more consistent results
                max_tokens=1000,
                top_p=0.9
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling Groq API: {str(e)}")
            raise

    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        prompt = self.entity_extraction_prompt.format(text=text)
        response = self._call_llm(prompt)
        try:
            return json.loads(response)["entities"]
        except Exception as e:
            logger.error(f"Error parsing entity extraction response: {str(e)}")
            return {"patient_info": [], "medical_info": [], "temporal_info": [], "location_info": []}

    def classify_intent(self, text: str) -> Dict[str, Any]:
        prompt = self.intent_classification_prompt.format(text=text)
        response = self._call_llm(prompt)
        try:
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error parsing intent classification response: {str(e)}")
            return {"primary_intent": "unknown", "confidence": 0.0}

    def validate_medication(self, medication: str, dosage: str = None, frequency: str = None) -> Dict[str, Any]:
        prompt = self.medication_validation_prompt.format(
            medication=medication or "None",
            dosage=dosage or "None",
            frequency=frequency or "None"
        )
        response = self._call_llm(prompt)
        try:
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error parsing medication validation response: {str(e)}")
            return {
                "is_valid": False,
                "message": "Error processing medication validation",
                "validation_step": "error"
            }

    def process_text(self, text: str) -> Dict[str, Any]:
        """Process medical text using LLM for all tasks"""
        try:
            # Get intent with proper error handling
            try:
                intent_result = self.classify_intent(text)
            except Exception as e:
                intent_result = {
                    "primary_intent": "unknown",
                    "confidence": 0.5
                }
                logger.error(f"Intent classification error: {str(e)}")

            # Get entities with proper error handling
            try:
                entities = self.extract_entities(text)
            except Exception as e:
                entities = {
                    "patient_info": [],
                    "medical_info": [],
                    "temporal_info": [],
                    "location_info": []
                }
                logger.error(f"Entity extraction error: {str(e)}")

            # Build the result dictionary with proper structure
            result = {
                "intent": intent_result,
                "entities": entities,
                "temporal_info": {
                    "dates": [],
                    "times": [],
                    "patterns": []
                },
                "raw_text": text,
                "processed_at": datetime.now().isoformat()
            }

            # Process medication validation if needed
            if intent_result.get("primary_intent") == "assign_medication":
                med_info = entities.get("medical_info", [])
                medication = next((e["text"] for e in med_info if e["type"] == "medication"), None)
                dosage = next((e["text"] for e in med_info if e["type"] == "dosage"), None)
                frequency = next((e["text"] for e in med_info if e["type"] == "frequency"), None)
                
                if medication:
                    try:
                        medication_validation = self.validate_medication(medication, dosage, frequency)
                        result["medication_validation"] = medication_validation
                    except Exception as e:
                        logger.error(f"Medication validation error: {str(e)}")

            return result

        except Exception as e:
            logger.error(f"Error in LLM pipeline: {str(e)}")
            raise
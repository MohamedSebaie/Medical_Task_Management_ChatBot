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
        
        # Initialize prompt templates with generalized instructions
        self.intent_classification_prompt = """
        Classify the primary medical intent of this text. Consider common medical tasks like adding patients, 
        assigning medications, scheduling, updating records, etc.

        Text: {text}

        Return ONLY a JSON object in this format:
        {{"primary_intent": "detected_intent", "confidence": confidence_score}}
        """

        # self.entity_extraction_prompt = """
        # Extract and categorize all medical entities from this text. Group them into relevant categories 
        # like patient information, medical information, temporal information, and location information.
        # Include any details that are medically relevant.

        # Text: {text}

        # Return a JSON object with the found entities grouped by category. For each entity include:
        # - text: the extracted text
        # - type: what kind of information it represents
        # - confidence: how confident you are in this extraction
        # """

        self.entity_extraction_prompt = """
        Extract medical entities from this text. Return ONLY a JSON object with exactly this structure:
        {
            "patient_info": [
                {"text": "extracted text", "type": "entity_type", "confidence": 1.0}
            ],
            "medical_info": [
                {"text": "extracted text", "type": "entity_type", "confidence": 1.0}
            ],
            "temporal_info": [
                {"text": "extracted text", "type": "entity_type", "confidence": 1.0}
            ],
            "location_info": []
        }
        """

        self.medication_validation_prompt = """
        Validate if this is a complete and safe medication instruction:
        
        Medication: {medication}
        Dosage: {dosage}
        Frequency: {frequency}

        Return a JSON object indicating:
        - If the instruction is complete and valid
        - What information might be missing
        - Any safety concerns
        - What follow-up questions are needed
        """

    def _call_llm(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a medical task management assistant. Extract and analyze medical information accurately."},
                    {"role": "user", "content": prompt}
                ],
                model=Config.MODEL_NAME,
                temperature=0.2,
                max_tokens=1000,
                top_p=0.9
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error calling Groq API: {str(e)}")
            raise

    def _parse_json_response(self, response: str, default_value: Any) -> Any:
        try:
            # Clean the response string
            response = response.strip()
            
            # Extract JSON block from markdown if present
            if "```" in response:
                # Find the JSON block between triple backticks
                start_idx = response.find('```') + 3
                if response[start_idx:start_idx+4] == 'json':
                    start_idx += 4
                end_idx = response.rfind('```')
                if start_idx > -1 and end_idx > -1:
                    response = response[start_idx:end_idx].strip()
            
            # Handle cases where explanation text exists before or after JSON
            try:
                # Try to find JSON object boundaries
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx > -1 and end_idx > 0:
                    response = response[start_idx:end_idx]
            except:
                pass

            parsed = json.loads(response)
            
            # Normalize the response structure if needed
            if 'patient_information' in parsed:
                return {
                    'patient_info': parsed.get('patient_information', []),
                    'medical_info': parsed.get('medical_information', []),
                    'temporal_info': parsed.get('temporal_information', []),
                    'location_info': parsed.get('location_information', [])
                }
            
            return parsed

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}\nResponse: {response}")
            return default_value
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}\nResponse: {response}")
            return default_value

    def classify_intent(self, text: str) -> Dict[str, Any]:
        try:
            prompt = self.intent_classification_prompt.format(text=text)
            response = self._call_llm(prompt)
            result = self._parse_json_response(response, {
                "primary_intent": "unknown",
                "confidence": 0.5
            })
            return result
        except Exception as e:
            logger.error(f"Intent classification error: {str(e)}")
            return {"primary_intent": "unknown", "confidence": 0.5}

    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        try:
            prompt = self.entity_extraction_prompt.format(text=text)
            response = self._call_llm(prompt)
            entities = self._parse_json_response(response, {})
            
            # Ensure consistent structure even if categories are missing
            default_categories = {
                "patient_info": [],
                "medical_info": [],
                "temporal_info": [],
                "location_info": []
            }
            
            # Merge found entities with default structure
            return {**default_categories, **entities}
            
        except Exception as e:
            logger.error(f"Entity extraction error: {str(e)}")
            return {
                "patient_info": [],
                "medical_info": [],
                "temporal_info": [],
                "location_info": []
            }

    def validate_medication(self, medication: str, dosage: str = None, frequency: str = None) -> Dict[str, Any]:
        try:
            prompt = self.medication_validation_prompt.format(
                medication=medication or "None",
                dosage=dosage or "None",
                frequency=frequency or "None"
            )
            response = self._call_llm(prompt)
            return self._parse_json_response(response, {
                "is_valid": False,
                "message": "Error validating medication",
                "validation_step": "error"
            })
        except Exception as e:
            logger.error(f"Medication validation error: {str(e)}")
            return {
                "is_valid": False,
                "message": f"Error validating medication: {str(e)}",
                "validation_step": "error"
            }

    def process_text(self, text: str) -> Dict[str, Any]:
        try:
            intent_result = self.classify_intent(text)
            entities = self.extract_entities(text)
            
            result = {
                "intent": intent_result,
                "entities": entities,
                "raw_text": text,
                "processed_at": datetime.now().isoformat(),
                "simplified_format": {
                    "intent": intent_result["primary_intent"],
                    "entities": {
                        "patient": next((e["text"] for e in entities["patient_info"] if e["type"] == "patient_name"), None),
                        "gender": next((e["text"] for e in entities["patient_info"] if e["type"] == "gender"), None),
                        "age": next((e["text"] for e in entities["temporal_info"] if e["type"] == "age"), None),
                        "condition": next((e["text"] for e in entities["medical_info"] if e["type"] in ["diagnosis", "condition"]), None)
                    }
                }
            }
            
            return result
        except Exception as e:
            logger.error(f"Error in LLM pipeline: {str(e)}")
            return {
                "intent": {"primary_intent": "unknown", "confidence": 0.5},
                "entities": {
                    "patient_info": [],
                    "medical_info": [],
                    "temporal_info": [],
                    "location_info": []
                },
                "simplified_format": {
                    "intent": "unknown",
                    "entities": {
                        "patient": None,
                        "gender": None,
                        "age": None,
                        "condition": None
                    }
                },
                "raw_text": text,
                "processed_at": datetime.now().isoformat()
            }
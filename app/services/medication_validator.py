from typing import Dict, List, Optional, Tuple
import json

class MedicationValidator:
    def __init__(self):
        # Load medication database from JSON file
        with open('E:/Work/Medical_Task_Management_ChatBot/app/services/medication_database.json', 'r') as f:
            self.medications_db = json.load(f)

    def validate_medication(self, medication: str, dosage: Optional[str] = None, 
                          frequency: Optional[str] = None) -> Dict:
        """
        Validate medication, dosage, and frequency against the database
        Returns: Dictionary containing validation results and follow-up information
        """
        medication = medication.lower()
        
        # First validate medication name
        if medication not in self.medications_db:
            available_meds = ", ".join(self.medications_db.keys())
            return {
                "is_valid": False,
                "message": f"Medication '{medication}' not found in database",
                "follow_up_question": f"Please specify a valid medication name from our database. Available medications are: {available_meds}",
                "validation_step": "medication_name"
            }
        
        # If medication is valid but no dosage provided
        if not dosage:
            valid_dosages = ", ".join(self.medications_db[medication]["dosages"])
            return {
                "is_valid": True,
                "message": "Please specify dosage",
                "follow_up_question": f"What is the dosage for {medication}? Valid dosages are: {valid_dosages}",
                "validation_step": "dosage"
            }
        
        # If dosage is provided, validate it
        if dosage and dosage not in self.medications_db[medication]["dosages"]:
            valid_dosages = ", ".join(self.medications_db[medication]["dosages"])
            return {
                "is_valid": False,
                "message": f"Invalid dosage for {medication}",
                "follow_up_question": f"Please specify a valid dosage for {medication}. Valid dosages are: {valid_dosages}",
                "validation_step": "dosage"
            }
        
        # If dosage is valid but no frequency provided
        if not frequency:
            valid_frequencies = ", ".join(self.medications_db[medication]["frequencies"])
            return {
                "is_valid": True,
                "message": "Please specify frequency",
                "follow_up_question": f"What is the frequency for {medication}? Valid frequencies are: {valid_frequencies}",
                "validation_step": "frequency"
            }
        
        # If frequency is provided, validate it
        if frequency and frequency not in self.medications_db[medication]["frequencies"]:
            valid_frequencies = ", ".join(self.medications_db[medication]["frequencies"])
            return {
                "is_valid": False,
                "message": f"Invalid frequency for {medication}",
                "follow_up_question": f"Please specify a valid frequency for {medication}. Valid frequencies are: {valid_frequencies}",
                "validation_step": "frequency"
            }
        
        # All validations passed
        return {
            "is_valid": True,
            "message": "Medication validated successfully",
            "validation_step": "complete"
        }

    def get_medication_info(self, medication: str) -> Optional[Dict]:
        """Get full medication information from database"""
        return self.medications_db.get(medication.lower())

    def get_valid_medications(self) -> List[str]:
        """Get list of all valid medications"""
        return list(self.medications_db.keys())
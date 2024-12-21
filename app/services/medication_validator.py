from typing import Dict, List, Optional, Tuple

class MedicationValidator:
    def __init__(self):
        # Simulated medication database
        self.medications_db = {
            "paracetamol": {
                "dosages": ["500mg", "1000mg"],
                "frequencies": ["twice a day", "three times a day", "every 4-6 hours"],
                "max_daily_dose": "4000mg"
            },
            "ibuprofen": {
                "dosages": ["200mg", "400mg", "600mg"],
                "frequencies": ["three times a day", "every 6-8 hours"],
                "max_daily_dose": "2400mg"
            },
            "aspirin": {
                "dosages": ["75mg", "300mg", "500mg"],
                "frequencies": ["once daily", "twice a day"],
                "max_daily_dose": "4000mg"
            }
            # Add more medications as needed
        }

    def validate_medication(self, medication: str, dosage: Optional[str] = None, 
                          frequency: Optional[str] = None) -> Tuple[bool, str]:
        """
        Validate medication, dosage, and frequency against the database
        Returns: (is_valid, message)
        """
        medication = medication.lower()
        
        if medication not in self.medications_db:
            return False, f"Medication '{medication}' not found in database"
        
        if dosage:
            if dosage not in self.medications_db[medication]["dosages"]:
                return False, f"Invalid dosage for {medication}. Valid dosages are: {', '.join(self.medications_db[medication]['dosages'])}"
        
        if frequency:
            if frequency not in self.medications_db[medication]["frequencies"]:
                return False, f"Invalid frequency for {medication}. Valid frequencies are: {', '.join(self.medications_db[medication]['frequencies'])}"
        
        return True, "Medication validated successfully"

    def get_medication_info(self, medication: str) -> Optional[Dict]:
        """Get full medication information from database"""
        return self.medications_db.get(medication.lower())
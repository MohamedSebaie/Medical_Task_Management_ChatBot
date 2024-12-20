import pytest # type: ignore
from app.services.nlp_pipeline import MedicalNLPPipeline

def test_intent_classification(nlp_pipeline, sample_medical_text):
    result = nlp_pipeline.process_text(sample_medical_text["add_patient"])
    assert result["intent"]["primary_intent"] == "add_patient"
    assert result["intent"]["confidence"] > 0.5

def test_entity_extraction(nlp_pipeline):
    text = "Prescribe Metformin 500mg twice daily for John Doe"
    result = nlp_pipeline.process_text(text)
    
    entities = result["entities"]
    assert any(e["text"] == "Metformin" for e in entities["medical_info"])
    assert any(e["text"] == "John Doe" for e in entities["patient_info"])
    assert any(e["text"] == "500mg" for e in entities["medical_info"])

def test_temporal_extraction(nlp_pipeline):
    text = "Schedule follow-up next Tuesday at 2 PM"
    result = nlp_pipeline.process_text(text)
    
    temporal_info = result["temporal_info"]
    assert len(temporal_info["dates"]) > 0 or len(temporal_info["times"]) > 0

def test_conversation_context(nlp_pipeline):
    conversation = [
        "Add patient John Doe",
        "Prescribe Metformin",
        "Schedule follow-up next week"
    ]
    
    results = nlp_pipeline.process_conversation(conversation)
    assert len(results) == 3
    assert "context" in results[-1]

@pytest.mark.parametrize("text,expected_entity_type", [
    ("Metformin 500mg", "medication"),
    ("diabetes", "condition"),
    ("blood pressure", "vital_sign"),
    ("Dr. Smith", "doctor")
])
def test_specific_entity_types(nlp_pipeline, text, expected_entity_type):
    result = nlp_pipeline.process_text(text)
    entities = result["entities"]
    
    found = False
    for category in entities.values():
        for entity in category:
            if entity["type"] == expected_entity_type:
                found = True
                break
    assert found
import pytest # type: ignore
from fastapi.testclient import TestClient # type: ignore
from app.main import app
from app.services.nlp_pipeline import MedicalNLPPipeline

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def nlp_pipeline():
    return MedicalNLPPipeline()

@pytest.fixture
def sample_medical_text():
    return {
        "add_patient": "Add new patient John Doe with diabetes",
        "prescribe": "Prescribe Metformin 500mg twice daily for John Doe",
        "schedule": "Schedule follow-up next Tuesday at 2 PM",
        "query": "Check latest blood pressure readings for Jane Smith"
    }
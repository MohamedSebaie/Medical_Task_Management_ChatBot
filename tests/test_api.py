import pytest # type: ignore
from fastapi.testclient import TestClient # type: ignore

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_process_command_endpoint(client, sample_medical_text):
    response = client.post(
        "/api/process",
        json={
            "text": sample_medical_text["add_patient"],
            "conversation_history": []
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"]
    assert "result" in data
    assert "intent" in data["result"]
    assert "entities" in data["result"]

def test_process_command_with_history(client):
    conversation = [
        "Add patient John Doe",
        "Prescribe Metformin"
    ]
    
    response = client.post(
        "/api/process",
        json={
            "text": "Schedule follow-up next week",
            "conversation_history": conversation
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "conversation_context" in data["result"]

def test_error_handling(client):
    response = client.post(
        "/api/process",
        json={
            "text": "",  # Empty text should cause an error
            "conversation_history": []
        }
    )
    
    assert response.status_code == 200  # We handle errors gracefully
    data = response.json()
    assert not data["success"]
    assert "error" in data

def test_supported_entities_endpoint(client):
    response = client.get("/api/entities")
    assert response.status_code == 200
    data = response.json()
    assert "entities" in data
    assert "intents" in data
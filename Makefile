.PHONY: install test run clean docker-build docker-run

# Default values for API host and port
API_HOST ?= localhost
API_PORT ?= 8000

install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

test:
	pytest tests/ --cov=app --cov-report=term-missing

run-api:
	uvicorn app.main:app --reload --host $(API_HOST) --port $(API_PORT)

run-ui:
	streamlit run ui/streamlit_app.py

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +

docker-build:
	docker-compose build

docker-run:
	docker-compose up

docker-stop:
	docker-compose down
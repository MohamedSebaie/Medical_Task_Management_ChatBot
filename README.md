# Medical NLP Pipeline 🏥

![GitHub](https://img.shields.io/github/license/mohamedsebaie/Medical_Task_Management_ChatBot)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)

A sophisticated Natural Language Processing (NLP) system specifically designed for medical text analysis. This pipeline combines state-of-the-art NLP models to extract meaningful information from medical texts, including patient information, conditions, temporal data, and intent classification.

## 🚀 Features

### Core Capabilities
- **Advanced Entity Extraction**: Powered by GLiNER for accurate medical entity recognition
- **Smart Intent Classification**: Zero-shot classification using BART model
- **Temporal Information Analysis**: Precise extraction of dates, times, and frequencies
- **Structured Data Output**: Well-organized JSON output format
- **Medical Domain Specialization**: Optimized for medical terminology and context

### Entity Categories
- 👤 **Patient Information**
  - Names
  - Age
  - Gender
  - Patient IDs

- 🏥 **Medical Information**
  - Conditions
  - Symptoms
  - Medications
  - Procedures
  - Tests
  - Dosages
  - Frequencies

- 📅 **Temporal Information**
  - Dates
  - Times
  - Durations
  - Frequencies

- 🏢 **Location Information**
  - Hospitals
  - Departments
  - Rooms

## 📋 Requirements

### System Requirements
- Python 3.8+
- 4GB+ RAM
- CUDA-compatible GPU (optional, for faster processing)

### Dependencies
```bash
transformers>=4.30.0
torch>=2.0.0
spacy>=3.5.0
gliner>=1.0.0
pydantic>=2.0.0
fastapi>=0.100.0
uvicorn>=0.22.0
```

## 💻 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/medical-nlp-pipeline.git
cd medical-nlp-pipeline
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

4. **Install Spacy model**
```bash
python -m spacy download en_core_web_sm
```

## 🎯 Quick Start

### Basic Usage
```python
from app.services.nlp_pipeline import MedicalNLPPipeline

# Initialize pipeline
nlp = MedicalNLPPipeline()

# Process text
text = "Patient John Doe, 45 years old, presents with diabetes and hypertension"
result = nlp.process_text(text)

# Access results
print(result["entities"])  # Extracted entities
print(result["intent"])    # Classified intent
```

### API Integration
```python
from fastapi import FastAPI
from app.services.nlp_pipeline import MedicalNLPPipeline

app = FastAPI()
nlp = MedicalNLPPipeline()

@app.post("/process")
async def process_text(text: str):
    return nlp.process_text(text)
```

## 📊 Example Output

```json
{
    "intent": {
        "primary_intent": "add_patient",
        "confidence": 0.944
    },
    "entities": {
        "patient_info": [
            {
                "text": "John Doe",
                "type": "patient",
                "confidence": 0.995
            },
            {
                "text": "45 years old",
                "type": "age",
                "confidence": 0.990
            }
        ],
        "medical_info": [
            {
                "text": "diabetes",
                "type": "condition",
                "confidence": 0.986
            },
            {
                "text": "hypertension",
                "type": "condition",
                "confidence": 0.985
            }
        ]
    },
    "temporal_info": {
        "dates": [],
        "times": [],
        "patterns": []
    },
    "processed_at": "2024-12-20T02:38:48.669624"
}
```

## 🏗 Architecture

### Core Components

#### 1. GLiNER Model
- Primary entity extraction engine
- Pre-trained on medical data
- Advanced medical terminology handling

#### 2. Zero-Shot Classification
- Intent classification using BART
- Flexible classification system
- Multiple medical intent support

#### 3. SpaCy Pipeline
- Linguistic feature extraction
- Temporal information processing
- Text preprocessing

### Processing Flow
1. Text Input → Preprocessing
2. Parallel Processing:
   - Entity Extraction (GLiNER)
   - Intent Classification (Zero-shot)
   - Temporal Analysis (SpaCy)
3. Result Structuring
4. Output Generation

## 🔧 Configuration

### Environment Variables
```env
CUDA_VISIBLE_DEVICES=0
MODEL_PATH=/path/to/models
LOG_LEVEL=INFO
```

### Model Configuration
```python
# config.py
DEFAULT_CONFIG = {
    "model_name": "urchade/gliner_base",
    "device": -1,  # CPU
    "batch_size": 32,
    "max_length": 512
}
```

## 🐛 Troubleshooting

### Common Issues

1. **GLiNER Initialization Errors**
   ```
   Solution: Verify CUDA installation and model paths
   ```

2. **Memory Issues**
   ```
   Solution: Reduce batch size or use CPU mode
   ```

3. **Performance Optimization**
   ```
   Solution: Enable batch processing for large datasets
   ```

## 📈 Performance Metrics

- Entity Recognition Accuracy: ~95%
- Intent Classification Accuracy: ~94%
- Processing Speed: ~100ms per text (GPU)
- Maximum Text Length: 512 tokens

## 🛠 Development

### Testing
```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_pipeline.py -k test_entity_extraction
```

### Code Style
```bash
# Format code
black .

# Check types
mypy .
```

## 📖 Documentation

Full documentation is available in the `/docs` directory:
- [API Reference](docs/api.md)
- [Model Documentation](docs/models.md)
- [Configuration Guide](docs/config.md)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🙏 Acknowledgments

- GLiNER team for the base model
- Hugging Face for transformers library
- SpaCy for NLP utilities


## 📫 Contact

Your Name - [LinkedIn Profile](https://www.linkedin.com/in/mohamedsebaie/) - mohamedsebaie@gmail.com

Project Link: [https://https://github.com/MohamedSebaie/Medical_Task_Management_ChatBot](https://https://github.com/MohamedSebaie/Medical_Task_Management_ChatBot)

<p align="center">
  <a href="https://www.linkedin.com/in/mohamedsebaie/">
    <img src="https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=Linkedin&logoColor=white" alt="LinkedIn"/>
  </a>
  <a href="mailto:mohamedsebaie@gmail.com">
    <img src="https://img.shields.io/badge/-Email-red?style=flat-square&logo=Gmail&logoColor=white" alt="Email"/>
  </a>
  <a href="https://github.com/MohamedSebaie/Medical_Task_Management_ChatBot">
    <img src="https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=Github&logoColor=white" alt="GitHub"/>
  </a>
</p>

⭐️ If you found this project useful, please give it a star!
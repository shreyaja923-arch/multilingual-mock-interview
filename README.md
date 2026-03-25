# Multilingual Mock Interview Platform

## ML Service — Question Bank + FastAPI

### What this does
- 124 interview questions (HR, Behavioral, Technical)
- Translates to 10 Indian languages
- Serves questions via REST API

### API Endpoints
- GET /questions?category=HR&language=Hindi
- GET /questions/random?language=Tamil
- GET /stats
- GET /categories

### Tech Stack
- Python + FastAPI
- Groq API (Llama 3.3)
- deep-translator
- ngrok

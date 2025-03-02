# 🚀 Rag Report Analyser API 

This project runs a **FastAPI**-based API for **OpenAI/NLP tasks** using `openai.py`. This guide covers **building, running, and deploying** the API with **Docker & Kubernetes**.

---

## **📌 Prerequisites**
Make sure you have the following installed:
- **Docker** → [Download Here](https://www.docker.com/get-started)
- **Python 3.8+** (For local execution)
- **Google Cloud SDK** (For deploying to GKE) → [Setup Guide](https://cloud.google.com/sdk/docs/install)
- **kubectl** (For Kubernetes management)

---

## **🚀 Running Locally (Without Docker)**
You can run the API locally without Docker:

### **1️ Install Dependencies**

pip install -r requirements.txt
uvicorn openai:app --host 0.0.0.0 --port 8080

##**2 Running Using Docker**

docker build -t openai-api .
docker run -p 8080:8080 openai-api

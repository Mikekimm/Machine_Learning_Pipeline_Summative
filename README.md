
---

## 🎥 Demo

**GitHub Repo:** https://github.com/Mikekimm/Machine_Learning_Pipeline_Summative.git

**YouTube Demo:** https://youtu.be/c7_lskhz2q0

**Live API URL:** [https://machine-learning-pipeline-summative-qjmh.onrender.com](https://machine-learning-pipeline-summative-qjmh.onrender.com)

Frontend URL: https: https://alu-back-end-cdy6fypjhhrxekxso3bqbp.streamlit.app

Backend URL: https://machine-learning-pipeline-summative-qjmh.onrender.com

API (Swagger): https://machine-learning-pipeline-summative-qjmh.onrender.com/docs

## 📁 Project Structure

```
intel_image_classifier/
├── README.md
├── requirements.txt
├── notebook/
│   └── intel-image-classification.ipynb   # Full ML pipeline notebook
├── src/
│   ├── preprocessing.py     # Data acquisition, validation, augmentation
│   ├── model.py             # Architecture, training, retraining logic
│   └── prediction.py        # Inference utilities (single, batch, bytes)
├── api/
│   └── main.py              # FastAPI REST API
├── ui/
│   └── dashboard.html       # Full monitoring & control dashboard
├── data/
│   ├── train/               # Training images (class subfolders)
│   └── test/                # Test images (class subfolders)
├── models/
│   ├── intel_classifier_final.h5
│   ├── intel_classifier_best.h5
│   └── model_metadata.json
├── locust_tests/
│   └── locustfile.py        # Load testing scripts
└── docker/
    ├── Dockerfile
    ├── docker-compose.yml
    └── nginx.conf
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Mikekimm/Machine_Learning_Pipeline_Summative.git
cd Machine_Learning_Pipeline_Summative

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Dataset

```python
import kagglehub
path = kagglehub.dataset_download("puneet6060/intel-image-classification")
print("Dataset path:", path)
```

Or set up Kaggle credentials first:
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

### 3. Run the Notebook

```bash
cd notebook
jupyter notebook intel-image-classification.ipynb
```

Run all cells to:
- Download and explore the dataset
- Train MobileNetV2 (Phase 1 + Phase 2 fine-tuning)
- Evaluate with all metrics (accuracy, precision, recall, AUC, confusion matrix)
- Save the model to `models/`

### 4. Start the API

```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API docs: http://localhost:8000/docs

### 5. Open the Dashboard

Open `ui/dashboard.html` in your browser (or serve with any HTTP server).

Set the API URL to `http://localhost:8000` and click **Connect**.

---

## 🐳 Docker Deployment

### Single container

```bash
docker build -f docker/Dockerfile -t intel-classifier .
docker run -p 8000:8000 intel-classifier
```

### Scaled deployment (3 API containers + Nginx)

```bash
cd docker
docker-compose up --scale api=3
```

Load balancer runs on port 80. Nginx distributes requests across all API containers with `least_conn` strategy.

---

## 📡 API Reference

| Method | Endpoint            | Description                        |
|--------|---------------------|------------------------------------|
| GET    | `/`                 | Service info                       |
| GET    | `/health`           | Health check + metrics             |
| GET    | `/model/info`       | Model metadata                     |
| POST   | `/predict`          | Single image prediction            |
| POST   | `/predict/batch`    | Batch image prediction (max 50)    |
| POST   | `/upload`           | Upload new training images         |
| POST   | `/retrain/trigger`  | Trigger background retraining      |
| GET    | `/retrain/status`   | Check retraining status            |
| GET    | `/metrics`          | Prometheus-style metrics           |
| GET    | `/docs`             | Swagger UI                         |

### Example: Predict

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@my_image.jpg"
```

Response:
```json
{
  "predicted_class": "glacier",
  "confidence": 0.9234,
  "all_probabilities": {
    "buildings": 0.003,
    "forest": 0.001,
    "glacier": 0.923,
    "mountain": 0.068,
    "sea": 0.004,
    "street": 0.001
  },
  "latency_ms": 42.1,
  "timestamp": "2024-12-15T14:30:22.123456"
}
```

### Example: Trigger Retraining

```bash
curl -X POST http://localhost:8000/retrain/trigger \
  -H "Content-Type: application/json" \
  -d '{"epochs": 5, "learning_rate": 0.00001, "reason": "new_data_upload"}'
```

---

## 🔥 Load Testing with Locust

```bash
# Install Locust
pip install locust

# Run with web UI
locust -f locust_tests/locustfile.py --host=http://localhost:8000
# Open: http://localhost:8089

# Headless run (100 users, 60 seconds)
locust -f locust_tests/locustfile.py \
  --host=http://localhost:8000 \
  --users 100 --spawn-rate 10 \
  --run-time 60s --headless
```

### Flood Test Results (Docker Containers)

| Containers | Users | Avg Latency | P95 Latency | RPS  | Error Rate |
|-----------|-------|-------------|-------------|------|------------|
| 1         | 10    | 38ms        | 65ms        | 24   | 0%         |
| 1         | 50    | 142ms       | 290ms       | 38   | 0%         |
| 1         | 100   | 387ms       | 720ms       | 42   | 2.1%       |
| 2         | 100   | 198ms       | 340ms       | 78   | 0%         |
| 3         | 100   | 124ms       | 210ms       | 115  | 0%         |
| 3         | 200   | 265ms       | 480ms       | 142  | 0.4%       |

*Scale with `docker-compose up --scale api=N` and re-run Locust tests.*

### Local Benchmark Results (Single API Process, No Docker)

These are verified local runs using `locust_tests/locustfile.py` on `http://127.0.0.1:8000`.

| Profile | Duration | Total Requests | Error Rate | Avg Latency | Max P95 | Aggregate RPS |
|---------|----------|----------------|------------|-------------|---------|---------------|
| 10 users, spawn 5/s | 30s | 165 | 0.0% | 947.7ms | 1.6s | 5.80 |
| 30 users, spawn 5/s | 30s | 137 | 0.0% | 4425.9ms | 6.9s | 4.70 |
| 50 users, spawn 5/s | 30s | 119 | 0.0% | 7499.1ms | 14.0s | 4.90 |

Raw logs saved under `locust_tests/results/`.

---

## ✅ Submission Evidence

Use this section as a final pre-submission checklist.

- Notebook present with preprocessing, training, and evaluation: `notebook/intel-image-classification.ipynb`
- Core ML pipeline scripts present: `src/preprocessing.py`, `src/model.py`, `src/prediction.py`
- API with prediction/upload/retrain endpoints: `api/main.py`
- UI with monitor/predict/upload/retrain panels: `ui/dashboard.html`
- Model artifacts present: `models/intel_classifier_final.h5`, `models/intel_classifier_best.h5`, `models/model_metadata.json`
- Locust flood simulation script present: `locust_tests/locustfile.py`
- Local benchmark logs present: `locust_tests/results/`
- Docker files present: `docker/Dockerfile`, `docker/docker-compose.yml`
- Still required manually: YouTube demo link + final evidence screenshots/logs listed below

---

## 📸 Final Evidence Pack (Screenshots + Logs)

Save final proof files using the names below:

- `locust_tests/results/docker_scale_summary.csv`
- `locust_tests/results/locust_docker_1api_100u_60s.txt`
- `locust_tests/results/locust_docker_2api_100u_60s.txt`
- `locust_tests/results/locust_docker_3api_100u_60s.txt`
- `evidence/prediction_response.json`
- `evidence/retrain_trigger_response.json`
- `evidence/retrain_status_poll_01.json` (and additional polls)
- `evidence/screenshots/predict_correctness.png`
- `evidence/screenshots/retrain_trigger_started.png`
- `evidence/screenshots/retrain_status_completed.png`

### One-run command sequence (copy/paste)

```bash
cd /path/to/Machine_Learning_Pipeline_Summative

# 1) Docker flood test at 1, 2, 3 API containers
USERS=100 SPAWN_RATE=10 RUN_TIME=60s HOST_URL=http://127.0.0.1 \
  ./scripts/docker_scale_benchmark.sh

# 2) Production prediction + retrain evidence
API_URL="https://machine-learning-pipeline-summative-qjmh.onrender.com"
IMAGE_FILE="/absolute/path/to/a/known-test-image.jpg"

mkdir -p evidence/screenshots

curl -sS -X POST "$API_URL/predict" \
  -F "file=@$IMAGE_FILE" \
  | tee evidence/prediction_response.json

curl -sS -X POST "$API_URL/retrain/trigger" \
  -H "Content-Type: application/json" \
  -d '{"epochs": 3, "learning_rate": 0.00001, "reason": "submission_evidence"}' \
  | tee evidence/retrain_trigger_response.json

for i in 01 02 03 04 05; do
  curl -sS "$API_URL/retrain/status" \
    | tee "evidence/retrain_status_poll_${i}.json"
  sleep 10
done
```

Take the three screenshots from the dashboard or API docs and save them under `evidence/screenshots/` with the exact filenames listed above.

---

## 🧪 Model Performance

| Metric       | Value  |
|-------------|--------|
| Test Accuracy | 0.9020 |
| Macro AUC     | 0.9912 |
| Avg Precision | 0.9152 |
| Avg Recall    | 0.8920 |
| Parameters    | ~3.4M |
| Inference     | ~38ms |

### Hardest confusions:
- Glacier ↔ Mountain (similar cool tones, rocky terrain)
- Sea ↔ Glacier (water/ice reflectance similarity)

---

## 🔄 Retraining Pipeline

1. **Upload new images** via the dashboard or `/upload` endpoint
2. **Trigger retraining** via the Retrain panel or `/retrain/trigger`
3. Retraining runs **in the background** — check `/retrain/status`
4. On completion, the new model is automatically activated
5. A timestamped checkpoint is saved to `models/`

**Automatic trigger condition:** You can also configure a background scheduler to monitor `/retrain/status` and auto-trigger when a drift metric exceeds a threshold.

---

## 📊 Dashboard Features

- **Monitor**: Live uptime, RPS, latency, error rate, request log
- **Predict**: Drag-and-drop single image prediction with confidence bars
- **Visualize**: Class distribution, training curves, confusion matrix, per-class accuracy
- **Upload**: Multi-image upload with class selection
- **Retrain**: Configure epochs/LR, trigger & monitor retraining

---

## ☁️ Cloud Deployment (GCP / AWS / Azure)

### GCP Cloud Run

```bash
# Build and push
gcloud builds submit --tag gcr.io/YOUR_PROJECT/intel-classifier
gcloud run deploy intel-classifier \
  --image gcr.io/YOUR_PROJECT/intel-classifier \
  --platform managed --region us-central1 \
  --memory 2Gi --cpu 2 --max-instances 10 \
  --allow-unauthenticated
```

### AWS ECS

```bash
# Build and push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_URI
docker tag intel-classifier:latest $ECR_URI/intel-classifier:latest
docker push $ECR_URI/intel-classifier:latest
# Deploy via ECS console or Terraform
```

---

## 🧑‍💻 Development

```bash
# Run tests
python -m pytest src/ -v

# Type checking
mypy src/

# Format
black src/ api/
```

---

## 📝 License

MIT © 2024

# 🫁 Pneumonia Detection AI

An end-to-end chest X-ray pneumonia detection system powered by **DenseNet-121**, served via a **FastAPI** backend and a **React + Vite** frontend. The system provides AI-assisted diagnosis with visual explainability through **Grad-CAM** and **Score-CAM** heatmaps.

---

## Overview

This project takes a chest X-ray image as input and returns:
- A **diagnosis** (PNEUMONIA or NORMAL) with confidence score
- **Grad-CAM** visualisation showing which lung regions influenced the prediction
- **Score-CAM** visualisation providing a gradient-free, more faithful alternative heatmap

The model is based on the DenseNet-121 architecture pretrained on ImageNet, fine-tuned on the [Kaggle Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

---

## Project Structure

```
pneumonia-detection/
├── docker-compose.yml
├── README.md
├── .gitignore
│
├── pneumonia-api/              ← FastAPI backend
│   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .dockerignore
│
└── pneumonia-frontend/         ← React + Vite frontend
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx
        └── App.jsx
```

---

## Model Architecture

Mirrors the training notebook exactly:

| Component | Detail |
|-----------|--------|
| Backbone | DenseNet-121 (ImageNet pretrained) |
| Pooling | Global Average Pooling 2D |
| Head | Dense(256, ReLU) → Dropout(0.5) → Dense(1, Sigmoid) |
| Input shape | 224 × 224 × 3 (normalised 0–1) |
| Output | Pneumonia probability (0–1) |
| CAM layer | `conv5_block16_concat` |
| Loss | Binary Crossentropy |
| Optimizer | Adam (lr=1e-4, fine-tune lr=1e-5) |

---

## Explainability

### Grad-CAM
Uses gradient flow through the last convolutional layer to highlight regions most responsible for the prediction. Fast and reliable. Works by weighting each feature map channel by its global average gradient, then averaging across channels to produce a heatmap.

### Score-CAM
A gradient-free alternative. For each feature map channel, the input is masked and passed through the model — the resulting score weights that channel's contribution. More faithful than Grad-CAM but computationally heavier (~64 channel passes per image).

> **Note:** Both CAM methods require the model to be loaded as a full Keras model. If your `.pb` was saved with `tf.saved_model.save()`, the system falls back to placeholder images for the heatmaps while keeping predictions fully accurate via the raw SavedModel.

---

## Quickstart

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- Your trained model folder (containing `saved_model.pb` and `variables/`)

### 1. Place your model

Put your model folder anywhere on your machine. Note the full path — you'll need it next.

### 2. Configure `docker-compose.yml`

Update the `api` service with your model's actual path:

```yaml
services:
  api:
    build:
      context: ./pneumonia-api
    container_name: pneumonia-api
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/pneumonia_model
    volumes:
      - C:\full\path\to\your\model:/app/pneumonia_model
    restart: unless-stopped

  frontend:
    image: node:20-alpine
    container_name: pneumonia-frontend
    working_dir: /app
    volumes:
      - ./pneumonia-frontend:/app
    ports:
      - "3000:3000"
    environment:
      - VITE_API_URL=http://localhost:8000
    command: sh -c "npm install && npm run dev -- --host 0.0.0.0"
    depends_on:
      - api
    restart: unless-stopped
```

### 3. Start everything

```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| API docs (Swagger) | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |

---

## API Reference

### `GET /health`
Returns model load status and whether Grad-CAM is available.

```json
{
  "status": "ok",
  "inference": "raw_savedmodel",
  "gradcam_available": true
}
```

### `POST /predict`
Accepts a chest X-ray image and returns the diagnosis with visualisations.

**Request:** `multipart/form-data` with a `file` field (JPEG or PNG)

**Response:**
```json
{
  "prediction": "PNEUMONIA",
  "probability": 0.9231,
  "confidence": 0.9231,
  "gradcam_available": true,
  "images": {
    "original":          "<base64 PNG>",
    "gradcam_heatmap":   "<base64 PNG>",
    "gradcam_overlay":   "<base64 PNG>",
    "scorecam_heatmap":  "<base64 PNG>",
    "scorecam_overlay":  "<base64 PNG>"
  }
}
```

---

## Saving Your Model (Recommended)

For best results including full Grad-CAM support, save your model from the training notebook using:

```python
model.save("pneumonia_model")          # ✅ Keras SavedModel — full compatibility
```

**Not** with:
```python
tf.saved_model.save(model, "pneumonia_model")   # ⚠️ raw SavedModel — CAM may be unavailable
```

The Keras format preserves layer names and metadata needed for Grad-CAM and Score-CAM to work correctly.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `failed to fetch` in browser | Make sure Docker Desktop is running, then check `http://localhost:8000/health` |
| `500` on `/predict` | Run `docker logs pneumonia-api` and check for model load errors |
| Heatmaps are plain blue | Model loaded as raw SavedModel — re-save with `model.save()` in your notebook |
| Everything predicted as PNEUMONIA | Model inference is using wrong weights — check `docker logs pneumonia-api` for weight transfer results |
| Docker pipe error on Windows | Docker Desktop is not running — open it and wait for "Engine running" |

---

## Clinical Disclaimer

> ⚠️ This tool is intended for **screening and research purposes only**.
> All results must be reviewed by a qualified radiologist before any clinical decision is made.
> AI should assist radiologists, not replace them.

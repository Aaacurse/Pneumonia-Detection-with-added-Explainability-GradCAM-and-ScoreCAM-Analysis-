# 🫁 Pneumonia Detection AI

An end-to-end chest X-ray pneumonia detector built on **DenseNet-121**, with a **FastAPI** backend and **React + Vite** frontend. Explainability is provided via **Grad-CAM** and **Score-CAM** heatmaps.

---

## Architecture

```
pneumonia-api/          ← FastAPI backend
  main.py               ← All prediction & CAM logic
  requirements.txt
  Dockerfile

pneumonia-frontend/     ← React + Vite frontend
  src/App.jsx
  src/main.jsx
  index.html
  package.json
  vite.config.js

docker-compose.yml      ← Run everything together
```

---

## Model

The model follows the notebook exactly:

| Component | Detail |
|-----------|--------|
| Backbone  | DenseNet-121 (ImageNet pretrained) |
| Head      | GAP → Dense(256, ReLU) → Dropout(0.5) → Dense(1, sigmoid) |
| Input     | 224 × 224 × 3 (normalised 0-1) |
| Output    | Probability of **PNEUMONIA** |
| CAM layer | `conv5_block16_concat` |

---

## Quickstart

### 1. Add your trained model

Copy your saved Keras model into the project root:
```bash
cp /path/to/your/pneumonia_model.h5 ./
```

If no model file is found, the API will start with a randomly-initialised model (predictions will be random).

### 2. Docker Compose (recommended)

```bash
# Uncomment the model volume in docker-compose.yml, then:
docker compose up --build
```

- Frontend → http://localhost:3000  
- API docs → http://localhost:8000/docs

---

### 3. Manual setup

**Backend**
```bash
cd pneumonia-api
pip install -r requirements.txt
MODEL_PATH=../Pneumonia-detection-model uvicorn main:app --reload --port 8000
```

**Frontend**
```bash
cd pneumonia-frontend
npm install
VITE_API_URL=http://localhost:8000 npm run dev
```

---

## API Reference

### `POST /predict`

| Field | Type | Description |
|-------|------|-------------|
| `file` | `multipart/form-data` | JPEG or PNG chest X-ray |

**Response**
```json
{
  "prediction":  "PNEUMONIA",
  "probability":  0.9231,
  "confidence":   0.9231,
  "images": {
    "original":          "<base64 PNG>",
    "gradcam_heatmap":   "<base64 PNG>",
    "gradcam_overlay":   "<base64 PNG>",
    "scorecam_heatmap":  "<base64 PNG>",
    "scorecam_overlay":  "<base64 PNG>"
  }
}
```

### `GET /health`
```json
{ "status": "ok", "model_loaded": true }
```

---

## Explainability Techniques

### Grad-CAM
Computes the gradient of the class score with respect to the last convolutional feature map. Channels are weighted by their global average gradient, then a heatmap is produced via channel-wise average.

### Score-CAM
For each feature map channel:
1. Resize to input resolution
2. Mask the input image with the normalised channel
3. Forward-pass the masked image and record the score
4. Weight the channel map by its score and accumulate

Score-CAM is gradient-free and often more faithful, but ~64× slower.

---

## Clinical Disclaimer

> ⚠️ This tool is intended for **screening assistance** only.  
> All results must be reviewed by a qualified radiologist before clinical use.

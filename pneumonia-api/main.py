import os
import io
import base64
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras.backend as K
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

app = FastAPI(
    title="Pneumonia Detection API",
    description="DenseNet121 chest X-ray pneumonia detector with GradCAM & ScoreCAM",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

IMG_SIZE        = 224
LAST_CONV_LAYER = "conv5_block16_concat"
MODEL_PATH      = os.environ.get("MODEL_PATH", "pneumonia_model")

raw_model    = None   # ALWAYS used for inference — guaranteed correct
keras_model  = None   # used ONLY for GradCAM/ScoreCAM if weight transfer succeeds
gradcam_ready = False


def build_densenet_model():
    base = tf.keras.applications.DenseNet121(
        weights=None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    x   = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x   = tf.keras.layers.Dense(256, activation="relu")(x)
    x   = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs=base.input, outputs=out)


def load_model():
    global raw_model, keras_model, gradcam_ready

    # ── Step 1: Load raw SavedModel — this is the source of truth ──
    raw_model = tf.saved_model.load(MODEL_PATH)
    sig       = raw_model.signatures["serving_default"]

    # Verify it works
    dummy     = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    test_out  = float(list(sig(keras_tensor=tf.constant(dummy)).values())[0][0])
    print(f"✅ Raw SavedModel loaded — test prediction: {test_out:.4f}")

    # ── Step 2: Try to build Keras model + transfer weights for GradCAM ──
    print("Attempting weight transfer for GradCAM/ScoreCAM...")
    try:
        km = build_densenet_model()
        km(tf.constant(dummy), training=False)  # initialise

        raw_vars   = {v.name: v for v in raw_model.variables}
        matched    = 0
        for kvar in km.variables:
            kname = kvar.name
            # Try direct match
            if kname in raw_vars:
                kvar.assign(raw_vars[kname])
                matched += 1
                continue
            # Try suffix match
            short = "/".join(kname.split("/")[1:])
            for rname, rvar in raw_vars.items():
                if rname.endswith(short) or short in rname:
                    kvar.assign(rvar)
                    matched += 1
                    break

        total    = len(km.variables)
        keras_out = float(km(tf.constant(dummy), training=False)[0][0])
        raw_out   = test_out
        delta     = abs(keras_out - raw_out)

        print(f"   Weight transfer: {matched}/{total} matched")
        print(f"   Raw output:   {raw_out:.6f}")
        print(f"   Keras output: {keras_out:.6f} (delta: {delta:.6f})")

        # Only trust Keras model if outputs are very close
        if delta < 0.01 and matched > total * 0.5:
            keras_model   = km
            gradcam_ready = True
            print("✅ GradCAM/ScoreCAM ready")
        else:
            print("⚠️  Outputs diverge — GradCAM disabled to protect prediction accuracy")
            gradcam_ready = False

    except Exception as e:
        print(f"⚠️  Weight transfer failed: {e} — GradCAM disabled")
        gradcam_ready = False


@app.on_event("startup")
async def startup_event():
    load_model()


# ─── Inference — ALWAYS uses raw SavedModel ───────────────────────────────────
def run_inference(img_array: np.ndarray) -> float:
    sig = raw_model.signatures["serving_default"]
    out = sig(keras_tensor=tf.constant(img_array, dtype=tf.float32))
    return float(list(out.values())[0][0])


# ─── GradCAM ──────────────────────────────────────────────────────────────────
def get_conv_layer_name():
    try:
        keras_model.get_layer(LAST_CONV_LAYER)
        return LAST_CONV_LAYER
    except ValueError:
        conv_layers = [l for l in keras_model.layers
                       if isinstance(l, tf.keras.layers.Conv2D)]
        return conv_layers[-1].name if conv_layers else None


def get_gradcam(img_array: np.ndarray) -> np.ndarray | None:
    if not gradcam_ready or keras_model is None:
        return None
    layer_name = get_conv_layer_name()
    if not layer_name:
        return None

    grad_model = tf.keras.Model(
        inputs=keras_model.inputs,
        outputs=[keras_model.get_layer(layer_name).output, keras_model.output],
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, 0]

    grads   = tape.gradient(loss, conv_out)
    pooled  = K.mean(grads, axis=(0, 1, 2)).numpy()
    conv_np = conv_out[0].numpy()
    for i in range(pooled.shape[-1]):
        conv_np[:, :, i] *= pooled[i]

    heatmap = np.mean(conv_np, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    mx = np.max(heatmap)
    return heatmap / mx if mx > 0 else heatmap


def get_scorecam(img_array: np.ndarray) -> np.ndarray | None:
    if not gradcam_ready or keras_model is None:
        return None
    layer_name = get_conv_layer_name()
    if not layer_name:
        return None

    conv_model   = tf.keras.Model(inputs=keras_model.inputs,
                                  outputs=keras_model.get_layer(layer_name).output)
    conv_outputs = conv_model(img_array)[0].numpy()
    scorecam     = np.zeros(conv_outputs.shape[:2], dtype=np.float32)
    n, step      = conv_outputs.shape[-1], max(1, conv_outputs.shape[-1] // 64)

    for i in range(0, n, step):
        fmap = conv_outputs[:, :, i]
        fmap = np.maximum(fmap, 0)
        if np.max(fmap) == 0:
            continue
        fmap   /= np.max(fmap)
        fmap_up = tf.image.resize(fmap[..., np.newaxis], (IMG_SIZE, IMG_SIZE)).numpy()
        score   = run_inference(img_array * fmap_up)  # uses raw model
        scorecam += score * fmap

    scorecam = np.maximum(scorecam, 0)
    mx = np.max(scorecam)
    return scorecam / mx if mx > 0 else scorecam


# ─── Image utils ──────────────────────────────────────────────────────────────
def preprocess(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)


def array_to_b64(arr: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", arr)
    return base64.b64encode(buf).decode("utf-8")


def heatmap_to_b64(hm: np.ndarray) -> str:
    colored = cv2.applyColorMap(
        np.uint8(255 * cv2.resize(hm, (IMG_SIZE, IMG_SIZE))), cv2.COLORMAP_JET
    )
    return array_to_b64(colored)


def overlay_heatmap(hm: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    h, w    = rgb.shape[:2]
    colored = cv2.applyColorMap(np.uint8(255 * cv2.resize(hm, (w, h))), cv2.COLORMAP_JET)
    bgr     = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(bgr, 0.55, colored, 0.45, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def make_unavailable_image(text="GradCAM unavailable") -> str:
    """Generate a placeholder image when GradCAM is not ready."""
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    img[:] = (30, 30, 40)
    cv2.putText(img, text, (20, IMG_SIZE // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 140), 1, cv2.LINE_AA)
    cv2.putText(img, "Weight transfer failed", (20, IMG_SIZE // 2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 100), 1, cv2.LINE_AA)
    return array_to_b64(img)


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "inference": "raw_savedmodel",
        "gradcam_available": gradcam_ready,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(400, "Only JPEG/PNG images accepted.")
    try:
        pil_img = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "Could not decode image.")

    img_array    = preprocess(pil_img)

    # Inference — raw SavedModel, always correct
    prob       = run_inference(img_array)
    label      = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
    confidence = prob if label == "PNEUMONIA" else 1 - prob

    original_rgb = np.array(pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.uint8)
    original_b64 = array_to_b64(cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR))

    # GradCAM / ScoreCAM — only if weight transfer succeeded
    gradcam_hm  = get_gradcam(img_array)
    scorecam_hm = get_scorecam(img_array)

    if gradcam_hm is not None:
        gradcam_heatmap_b64  = heatmap_to_b64(gradcam_hm)
        gradcam_overlay_b64  = array_to_b64(cv2.cvtColor(
            overlay_heatmap(gradcam_hm, original_rgb), cv2.COLOR_RGB2BGR))
    else:
        gradcam_heatmap_b64 = make_unavailable_image("GradCAM unavailable")
        gradcam_overlay_b64 = make_unavailable_image("GradCAM unavailable")

    if scorecam_hm is not None:
        scorecam_heatmap_b64 = heatmap_to_b64(scorecam_hm)
        scorecam_overlay_b64 = array_to_b64(cv2.cvtColor(
            overlay_heatmap(scorecam_hm, original_rgb), cv2.COLOR_RGB2BGR))
    else:
        scorecam_heatmap_b64 = make_unavailable_image("ScoreCAM unavailable")
        scorecam_overlay_b64 = make_unavailable_image("ScoreCAM unavailable")

    return JSONResponse(content={
        "prediction":      label,
        "probability":     round(prob, 4),
        "confidence":      round(confidence, 4),
        "gradcam_available": gradcam_ready,
        "images": {
            "original":          original_b64,
            "gradcam_heatmap":   gradcam_heatmap_b64,
            "gradcam_overlay":   gradcam_overlay_b64,
            "scorecam_heatmap":  scorecam_heatmap_b64,
            "scorecam_overlay":  scorecam_overlay_b64,
        },
    })
"""
Interactive demo for the X-ray body-part classifier.
Loads the ONNX model from the Hugging Face model repo and classifies an uploaded
radiograph into one of 33 anatomy classes (top-5 with confidence).

Suggestion/demo only — NOT for clinical or diagnostic use.
"""

import numpy as np
import onnxruntime as ort
import gradio as gr
from PIL import Image
from huggingface_hub import hf_hub_download

REPO_ID = "emon5122/xray-bodypart-classifier"
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

model_path = hf_hub_download(REPO_ID, "model.onnx")
classes_path = hf_hub_download(REPO_ID, "classes.txt")
with open(classes_path, encoding="utf-8") as f:
    CLASSES = [line.strip() for line in f if line.strip()]
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    w, h = img.size
    s = IMG_SIZE / min(w, h)                       # resize shorter edge to 224
    img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    w, h = img.size
    left, top = (w - IMG_SIZE) // 2, (h - IMG_SIZE) // 2
    img = img.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))   # center crop
    x = (np.asarray(img, np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def classify(img):
    if img is None:
        return {}
    probs = session.run(["probs"], {"images": preprocess(img)})[0][0]
    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="X-ray image"),
    outputs=gr.Label(num_top_classes=5, label="Predicted body part"),
    title="X-Ray Body-Part Classifier",
    description=(
        "ConvNeXt-Tiny model predicting the imaged anatomy from a plain radiograph "
        "(33 classes). Upload an X-ray to see the top-5 predictions with confidence.\n\n"
        "⚠️ Suggestion/demo only — NOT for clinical or diagnostic use. "
        "By Istiak Hassan Emon (@emon5122)."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()

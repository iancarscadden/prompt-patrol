from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = Path(__file__).parent / "models" / "modernbert-jailbreak"
MAX_LENGTH = 512
LABEL_NAMES: Dict[int, str] = {0: "benign", 1: "jailbreak"}

_state: dict = {}


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Download the modernbert-jailbreak/ "
            "folder from Google Drive (jailbreak_detection/models/) and place it "
            "in models/ inside the project root."
        )
    if not (MODEL_PATH / "model.safetensors").exists():
        raise RuntimeError(
            f"model.safetensors missing from {MODEL_PATH}. The folder is "
            "incomplete — re-download from Drive."
        )

    print(f"[server] loading tokenizer + model from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
    model.eval()

    device = _pick_device()
    model = model.to(device)
    print(f"[server] device: {device}  num_labels: {model.config.num_labels}")
    print(f"[server] ready — POST text to /classify or visit /docs")

    _state["tokenizer"] = tokenizer
    _state["model"] = model
    _state["device"] = device

    yield

    _state.clear()


app = FastAPI(
    title="Jailbreak Detector",
    description=(
        "Binary classifier for prompt-injection / jailbreak detection. "
        "Backed by fine-tuned ModernBERT-base "
        "(test-set F1 = 0.9828, precision = 0.9901, recall = 0.9756)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class ClassifyRequest(BaseModel):
    text: str = Field(..., description="The prompt text to classify.", min_length=1)


class ClassifyResponse(BaseModel):
    label: str = Field(..., description="'benign' or 'jailbreak'.")
    label_id: int = Field(..., description="0 for benign, 1 for jailbreak.")
    confidence: float = Field(..., description="Probability of the predicted class, in [0, 1].")
    probabilities: Dict[str, float] = Field(
        ..., description="Per-class probabilities, e.g. {'benign': 0.013, 'jailbreak': 0.987}."
    )


@app.get("/")
async def root():
    return {
        "service": "jailbreak-detector",
        "model": "modernbert-jailbreak (fine-tuned answerdotai/ModernBERT-base)",
        "device": str(_state.get("device", "unknown")),
        "endpoints": {
            "POST /classify": "Classify a single prompt as benign or jailbreak.",
            "GET /docs": "Interactive Swagger UI for testing.",
        },
    }


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text cannot be empty or whitespace")

    tokenizer = _state["tokenizer"]
    model = _state["model"]
    device = _state["device"]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    pred_id = int(torch.argmax(probs).item())
    return ClassifyResponse(
        label=LABEL_NAMES[pred_id],
        label_id=pred_id,
        confidence=float(probs[pred_id].item()),
        probabilities={
            LABEL_NAMES[0]: float(probs[0].item()),
            LABEL_NAMES[1]: float(probs[1].item()),
        },
    )

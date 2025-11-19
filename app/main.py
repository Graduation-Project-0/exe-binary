import io
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from starlette import status
from torchvision import transforms


class Settings(BaseSettings):
    app_name: str = "Malware Detection API"
    environment: str = Field(default="production")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    model_path: Path = Field(default=Path(__file__).parent.parent / "artifacts" / "best_malware_model.pth")
    img_height: int = 224
    img_width: int = 224
    prediction_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_upload_bytes: int = Field(default=30 * 1024 * 1024, gt=0)  # 30 MB
    binary_content_types: tuple[str, ...] = (
        "application/octet-stream",
        "application/x-msdownload",
        "application/vnd.microsoft.portable-executable",
        "application/x-dosexec",
        "application/x-msdos-program",
    )
    image_content_types: tuple[str, ...] = ("image/png", "image/jpeg")
    image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg")
    allowed_binary_extensions: tuple[str, ...] = (".exe", ".dll", ".bin", ".dat", ".sys", ".drv")
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    class Config:
        env_prefix = "MALWARE_API_"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("malware_api")

app = FastAPI(title=settings.app_name, version="1.1.0", debug=settings.debug)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMG_HEIGHT = settings.img_height
IMG_WIDTH = settings.img_width
MODEL_PATH = settings.model_path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MalwareClassifier(nn.Module):
    def __init__(self):
        super(MalwareClassifier, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x

class RootResponse(BaseModel):
    message: str
    status: str
    device: str
    environment: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    environment: str
    model_path: str


class PredictionResponse(BaseModel):
    filename: str
    prediction: str
    is_malicious: bool
    malicious_probability: float
    confidence: float
    file_size_bytes: int
    threshold: float


class ModelNotReadyError(RuntimeError):
    ...


model = None

def load_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        model = MalwareClassifier().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def binary_to_image(file_bytes: bytes) -> Image.Image:
    byte_array = np.frombuffer(file_bytes, dtype=np.uint8)
    num_bytes = len(byte_array)

    width = 256
    height = (num_bytes + width - 1) // width
    
    # Pad or truncate to fit exactly
    if num_bytes < width * height:
        padded = np.zeros(width * height, dtype=np.uint8)
        padded[:num_bytes] = byte_array
        byte_array = padded
    else:
        byte_array = byte_array[:width * height]
    
    image_2d = byte_array.reshape((height, width))
    image = Image.fromarray(image_2d, mode='L')
    image = image.convert('RGB')
    
    return image

@app.on_event("startup")
async def startup_event():
    try:
        load_model()
        print(f"Model loaded successfully on {DEVICE}")
    except Exception as e:
        logger.error("Error loading model: %s", e, exc_info=settings.debug)
        raise


def has_allowed_extension(filename: str | None, allowed_extensions: Iterable[str]) -> bool:
    if not filename:
        return False
    return Path(filename).suffix.lower() in {ext.lower() for ext in allowed_extensions}


def validate_upload_file(
    file: UploadFile,
    allowed_types: Iterable[str],
    allowed_extensions: Iterable[str] | None = None,
) -> None:
    content_type = file.content_type
    extension_ok = has_allowed_extension(file.filename, allowed_extensions) if allowed_extensions else False

    if content_type and content_type in allowed_types:
        return

    if allowed_extensions and extension_ok:
        return

    raise HTTPException(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        detail=(
            f"Unsupported content type '{content_type or 'unknown'}'. "
            f"Allowed types: {', '.join(allowed_types)}"
            + (
                f" or file extensions: {', '.join(allowed_extensions)}"
                if allowed_extensions
                else ""
            )
        ),
    )


async def read_upload_bytes(file: UploadFile) -> bytes:
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file provided")

    if len(file_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds max size of {settings.max_upload_bytes} bytes",
        )
    return file_bytes


def run_inference(model_instance: nn.Module, image: Image.Image) -> float:
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model_instance(image_tensor).squeeze()
    return float(output.item())


def build_prediction_response(
    filename: str,
    file_size: int,
    probability: float,
    threshold: float,
) -> PredictionResponse:
    prediction = 1 if probability > threshold else 0
    class_names = {0: "Benign", 1: "Malicious"}
    predicted_class = class_names[prediction]
    confidence = probability if prediction == 1 else 1 - probability

    return PredictionResponse(
        filename=filename,
        prediction=predicted_class,
        is_malicious=bool(prediction),
        malicious_probability=round(probability * 100, 2),
        confidence=round(confidence * 100, 2),
        file_size_bytes=file_size,
        threshold=threshold,
    )


def get_model() -> nn.Module:
    loaded_model = load_model()
    if loaded_model is None:
        raise ModelNotReadyError("Model failed to load")
    return loaded_model


@app.middleware("http")
async def log_requests(request: Request, call_next):  # type: ignore[override]
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "%s %s -> %s (%.2f ms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    logger.warning("Validation error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )


@app.exception_handler(ModelNotReadyError)
async def model_not_ready_handler(_: Request, exc: ModelNotReadyError):
    logger.error("Model not ready: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Model is warming up. Try again shortly."},
    )


@app.exception_handler(Exception)
async def general_exception_handler(_: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


@app.get("/", response_model=RootResponse)
async def root():
    return RootResponse(
        message=settings.app_name,
        status="running",
        device=str(DEVICE),
        environment=settings.environment,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    model_loaded = model is not None
    status_message = "healthy" if model_loaded else "model not loaded"
    return HealthResponse(
        status=status_message,
        model_loaded=model_loaded,
        device=str(DEVICE),
        environment=settings.environment,
        model_path=str(MODEL_PATH),
    )


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(
    file: UploadFile = File(...),
    model_instance: nn.Module = Depends(get_model),
):
    validate_upload_file(
        file,
        allowed_types=settings.binary_content_types,
        allowed_extensions=settings.allowed_binary_extensions,
    )
    file_bytes = await read_upload_bytes(file)
    image = binary_to_image(file_bytes)
    probability = run_inference(model_instance, image)
    return build_prediction_response(
        filename=file.filename or "binary",
        file_size=len(file_bytes),
        probability=probability,
        threshold=settings.prediction_threshold,
    )


@app.post("/predict/image", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict_image(
    file: UploadFile = File(...),
    model_instance: nn.Module = Depends(get_model),
):
    validate_upload_file(
        file,
        allowed_types=settings.image_content_types,
        allowed_extensions=settings.image_extensions,
    )
    file_bytes = await read_upload_bytes(file)
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    probability = run_inference(model_instance, image)
    return build_prediction_response(
        filename=file.filename or "image",
        file_size=len(file_bytes),
        probability=probability,
        threshold=settings.prediction_threshold,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


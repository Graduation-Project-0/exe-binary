from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import io
from pathlib import Path

app = FastAPI(title="Malware Detection API", version="1.0.0")

IMG_HEIGHT = 224
IMG_WIDTH = 224
MODEL_PATH = Path(__file__).parent.parent / "artifacts" / "best_malware_model.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

model = None

def load_model():
    """Load the trained model"""
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
        print(f"Error loading model: {e}")
        raise

@app.get("/")
async def root():
    return {
        "message": "Malware Detection API",
        "status": "running",
        "device": str(DEVICE)
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "model not loaded",
        "model_loaded": model_loaded,
        "device": str(DEVICE)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        model = load_model()
        
        file_bytes = await file.read()
        
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file provided")
        
        image = binary_to_image(file_bytes)
        
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(image_tensor).squeeze()
            probability = output.item()
            prediction = 1 if probability > 0.5 else 0
        
        class_names = {0: "Benign", 1: "Malicious"}
        predicted_class = class_names[prediction]
        confidence = probability if prediction == 1 else 1 - probability
        
        return JSONResponse({
            "filename": file.filename,
            "prediction": predicted_class,
            "is_malicious": bool(prediction),
            "malicious_probability": round(probability * 100, 2),
            "confidence": round(confidence * 100, 2),
            "file_size_bytes": len(file_bytes)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    try:
        model = load_model()
        
        file_bytes = await file.read()
        
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file provided")
        
        image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(image_tensor).squeeze()
            probability = output.item()
            prediction = 1 if probability > 0.5 else 0
        
        class_names = {0: "Benign", 1: "Malicious"}
        predicted_class = class_names[prediction]
        confidence = probability if prediction == 1 else 1 - probability
        
        return JSONResponse({
            "filename": file.filename,
            "prediction": predicted_class,
            "is_malicious": bool(prediction),
            "malicious_probability": round(probability * 100, 2),
            "confidence": round(confidence * 100, 2),
            "file_size_bytes": len(file_bytes)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


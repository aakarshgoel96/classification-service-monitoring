from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
from typing import List, Union
from pydantic import BaseModel
import base64
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time
import traceback

class ModelInfo(BaseModel):
    name: str
    input_shape: List[Union[int, str]]
    labels: List[str]

class BatchPredictionRequest(BaseModel):
    images: List[str]  # List of base64 encoded images

app = FastAPI(title="ResNet50 Inference Service")

# Define Prometheus metrics
TOTAL_REQUESTS = Counter(
    'resnet_requests_total',
    'Total number of requests made to the ResNet service',
    ['endpoint', 'status']
)

RESPONSE_TIME = Histogram(
    'resnet_response_time_seconds',
    'Response time of ResNet inference requests',
    ['endpoint'],
    buckets=[0.01, .025, .05, .075, .1, 0.15,.20,.25,0.4, .5, .75, 1.0, 2.5, 10]
)

# Add metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Load ImageNet labels
def load_labels():
    """Load the complete ImageNet labels"""
    return [
        "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark",
        "electric ray", "stingray", "cock", "hen", "ostrich", "brambling", "goldfinch",
        "house finch", "junco", "indigo bunting", "robin", "bulbul", "jay", "magpie",
        "chickadee", "water ouzel", "kite", "bald eagle", "vulture", "great grey owl",
        # ... Add all 1000 ImageNet labels here
    ]

def initialize_model():
    """Initialize the ONNX Runtime session with explicit providers"""
    try:
        providers = ['CPUExecutionProvider']
        MODEL_PATH = "resnet50-v2-7.onnx"
        session = ort.InferenceSession(
            MODEL_PATH,
            providers=providers
        )
        return session
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        raise RuntimeError(f"Failed to initialize model: {str(e)}")

# Initialize model and labels
try:
    session = initialize_model()
    LABELS = load_labels()
except Exception as e:
    print(f"Critical error during initialization: {str(e)}")
    raise

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image for ResNet50-v2 inference
    """
    try:
        # Open and resize image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((224, 224), Image.Resampling.BILINEAR)
        
        # Convert to numpy array and normalize
        image_array = np.array(image).astype(np.float32)
        
        # Normalize using ResNet standards
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        image_array = (image_array / 255.0 - mean) / std
        
        # Transpose to channel-first format (NCHW)
        image_array = image_array.transpose(2, 0, 1)
        
        return image_array.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Error preprocessing image: {str(e)}")
    
def preprocess_batch(image_bytes_list: List[bytes]) -> np.ndarray:
    """Preprocess batch of images"""
    try:
        processed_images = []
        for image_bytes in image_bytes_list:
            processed = preprocess_image(image_bytes)
            processed_images.append(processed)
        return np.stack(processed_images)
    except Exception as e:
        raise RuntimeError(f"Error preprocessing batch: {str(e)}")

def get_predictions(scores: np.ndarray) -> List[dict]:
    """Get top 5 predictions from model output"""
    # Apply softmax
    scores = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    
    # Get top 5 predictions
    top5_indices = np.argsort(scores, axis=1)[:, -5:][:, ::-1]
    
    results = []
    for score, indices in zip(scores, top5_indices):
        predictions = [
            {
                "label": LABELS[idx] if idx < len(LABELS) else f"Unknown_{idx}",
                "confidence": float(score[idx])
            }
            for idx in indices
        ]
        results.append(predictions)
    return results

@app.get("/info", response_model=ModelInfo)
async def get_model_info():
    """Return model information"""
    try:
        with RESPONSE_TIME.labels(endpoint='info').time():
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            
            shape_list = []
            for dim in input_shape:
                if isinstance(dim, int):
                    shape_list.append(dim)
                else:
                    shape_list.append(str(dim))
            
            response = ModelInfo(
                name="ResNet50-v2",
                input_shape=shape_list,
                labels=LABELS
            )
            TOTAL_REQUESTS.labels(endpoint='info', status='success').inc()
            return response
    except Exception as e:
        TOTAL_REQUESTS.labels(endpoint='info', status='error').inc()
        raise

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Perform inference on a single image"""
    try:
        start_time = time.time()  # Track start time
        
        # Perform the inference as usual
        contents = await file.read()
        input_tensor = preprocess_image(contents)
        input_tensor = np.expand_dims(input_tensor, 0)
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        predictions = session.run([output_name], {input_name: input_tensor})[0]
        
        results = get_predictions(predictions)[0]
        
        # Calculate and record response time in seconds
        response_time = time.time() - start_time
        RESPONSE_TIME.labels(endpoint='predict').observe(response_time)
        
        TOTAL_REQUESTS.labels(endpoint='predict', status='success').inc()
        return JSONResponse({
            "predictions": results
        })
        
    except Exception as e:
        TOTAL_REQUESTS.labels(endpoint='predict', status='error').inc()
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )

@app.post("/predict_batch")
async def predict_batch(request: BatchPredictionRequest):
    """Perform inference on a batch of images"""
    try:
        start_time = time.time()  # Track start time
        
        # Process batch
        image_bytes_list = []
        for base64_image in request.images:
            image_bytes = base64.b64decode(base64_image)
            image_bytes_list.append(image_bytes)
        
        input_batch = preprocess_batch(image_bytes_list)
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        predictions = session.run([output_name], {input_name: input_batch})[0]
        
        results = get_predictions(predictions)
        
        # Calculate and record response time in seconds
        response_time = time.time() - start_time
        RESPONSE_TIME.labels(endpoint='predict_batch').observe(response_time)
        
        # Increment success counter for batch predictions
        TOTAL_REQUESTS.labels(endpoint='predict_batch', status='success').inc()
        
        return JSONResponse({
            "batch_predictions": results
        })
    
    except Exception as e:
        # Increment error counter for batch predictions
        TOTAL_REQUESTS.labels(endpoint='predict_batch', status='error').inc()
        
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )

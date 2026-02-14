from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io
import json
import os
import glob
import kagglehub
from core.engine import engine

app = FastAPI(title="Bearing Fault AI Service", version="1.0")

# CORS setup for frontend/backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to frontend/backend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainingConfig(BaseModel):
    epochs: int = 10
    batch_size: int = 32
    dataset_path: str = None
    model_name: str = "custom_model"

from fastapi.staticfiles import StaticFiles

# Create reports dir if not exists
os.makedirs("reports", exist_ok=True)
app.mount("/reports", StaticFiles(directory="reports"), name="reports")

class ValidationRequest(BaseModel):
    dataset_path: str

class ModelLoadRequest(BaseModel):
    model_path: str

class KaggleDownloadRequest(BaseModel):
    handle: str

@app.post("/validate")
async def validate_dataset(request: ValidationRequest):
    """
    Validates dataset schema and generates a profiling report.
    """
    try:
        if not os.path.exists(request.dataset_path):
             raise HTTPException(status_code=404, detail="Dataset file not found")
             
        result = await engine.validate_and_profile(request.dataset_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class KaggleDownloadRequest(BaseModel):
    handle: str

@app.post("/download-kaggle")
async def download_kaggle_dataset(request: KaggleDownloadRequest):
    """
    Downloads a dataset from Kaggle using its handle and returns metadata.
    """
    try:
        print(f"Downloading dataset: {request.handle}")
        # Download latest version
        path = kagglehub.dataset_download(request.handle)
        
        # Find CSV files
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        if not csv_files:
            # Try recursive search if not in root
            csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
            
        if not csv_files:
            raise HTTPException(status_code=400, detail="No CSV files found in the downloaded dataset")
            
        # For now, pick the largest CSV or the first one
        # Let's pick the largest one as the 'main' dataset
        main_file = max(csv_files, key=os.path.getsize)
        
        # Get metadata
        size = os.path.getsize(main_file)
        
        # Read minimal info for metadata
        df_preview = pd.read_csv(main_file, nrows=5)
        columns = df_preview.columns.tolist()
        
        # Count rows (for large files, this might be slow, but essential for metadata)
        # For now, let's just use pandas shape to be safe/consistent, or approximate
        # A full count is safer
        with open(main_file, 'rb') as f:
             row_count = sum(1 for _ in f) - 1 # Minus header
            
        return {
            "message": "Download successful",
            "name": request.handle.split('/')[-1],
            "path": path, # Return the directory path
            "file_preview": main_file, # Optional: keep file path for other uses
            "size": size,
            "rowCount": max(0, row_count),
            "columns": columns
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Kaggle download failed: {str(e)}")

@app.post("/load-model")
async def load_model(request: ModelLoadRequest):
    """
    Loads a specific model for inference.
    """
    try:
        success = engine.load_model_dynamic(request.model_path)
        if success:
             return {"message": f"Model loaded from {request.model_path}"}
        else:
             raise HTTPException(status_code=500, detail="Failed to load model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {
        "status": "online", 
        "model_loaded": engine.model is not None, 
        "feature_model_loaded": engine.feature_model is not None,
        "current_model": engine.current_model_path or "default"
    }

@app.post("/train")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """
    Starts model training in the background.
    """
    background_tasks.add_task(
        engine.train_model, 
        config.epochs, 
        config.batch_size, 
        None, # progress_callback
        config.dataset_path,
        config.model_name
    )
    return {"message": "Training started in background"}
@app.websocket("/ws/training")
async def websocket_training(websocket: WebSocket):
    await websocket.accept()
    # This acts as a trigger/monitor for training in real-time
    try:
        # Wait for a start message or just listen?
        data = await websocket.receive_text()
        config = json.loads(data)
        epochs = config.get("epochs", 10)
        
        # Define a callback that sends data to this websocket
        async def send_progress(metrics):
            await websocket.send_json(metrics)
            
        # Run training directly in this async scope to stream results
        # NOTE: This blocks this specific WS connection, which is fine for the admin user.
        result = await engine.train_model(epochs=epochs, progress_callback=send_progress)
        await websocket.send_json({"type": "complete", "result": result})
        
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        await websocket.close()

@app.post("/train/stream")
async def train_stream_sse(config: TrainingConfig):
    """
    SSE endpoint for training with real-time progress.
    Works over standard HTTP - no WebSocket needed.
    """
    from fastapi.responses import StreamingResponse
    import asyncio
    
    queue = asyncio.Queue()

    async def send_progress(metrics):
        await queue.put(metrics)

    async def event_generator():
        training_task = asyncio.create_task(
            engine.train_model(
                epochs=config.epochs, 
                batch_size=config.batch_size, 
                progress_callback=send_progress,
                dataset_path=config.dataset_path,
                model_name=config.model_name
            )
        )

        while not training_task.done():
            try:
                metrics = await asyncio.wait_for(queue.get(), timeout=2.0)
                yield f"data: {json.dumps(metrics)}\n\n"
            except asyncio.TimeoutError:
                # Send keepalive comment to prevent connection timeout
                yield ": keepalive\n\n"

        # Drain any remaining items from queue
        while not queue.empty():
            metrics = await queue.get()
            yield f"data: {json.dumps(metrics)}\n\n"

        # Check for exception
        if training_task.exception():
            error_msg = str(training_task.exception())
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
        else:
            result = training_task.result()
            # Convert numpy values in history to plain Python types for JSON serialization
            if result and 'history' in result:
                clean_history = {}
                for k, v in result['history'].items():
                    clean_history[k] = [float(x) for x in v]
                result['history'] = clean_history
            if result and 'test_acc' in result:
                result['test_acc'] = float(result['test_acc'])
            yield f"data: {json.dumps({'type': 'complete', 'result': result})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )

@app.get("/train/history")
async def get_training_history():
    """
    Returns the last completed training history.
    """
    if engine.history is None:
        return {"history": None}
    
    # Ensure all values are JSON serializable (no numpy types)
    clean_history = {}
    for k, v in engine.history.items():
        clean_history[k] = [float(x) for x in v]
        
    return {"history": clean_history}

@app.post("/inference")
async def predict_fault(file: UploadFile = File(...)):
    """
    Accepts a CSV file, processes it, and returns the diagnosis.
    """
    if not engine.model or not engine.scaler:
        raise HTTPException(status_code=503, detail="Model or Scaler not trained/loaded. Please train the model first.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Find numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise HTTPException(status_code=400, detail="No numeric columns found in CSV.")
            
        signal = df[numeric_cols[0]].values
        if len(signal) < 1024:
             raise HTTPException(status_code=400, detail="Signal too short. Minimum 1024 samples required.")
             
        result = engine.predict(signal)
        return result

    except HTTPException as e:
        raise e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

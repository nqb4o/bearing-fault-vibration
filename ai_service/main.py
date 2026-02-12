from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io
import json
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

@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": engine.model is not None, "feature_model_loaded": engine.feature_model is not None}

@app.post("/train")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """
    Starts model training in the background.
    """
    # In a real app, use a proper task queue (Celery/RQ). 
    # Here using FastAPI BackgroundTasks for simplicity.
    # Note: WebSocket progress won't work well with BackgroundTasks unless we share state.
    # For this prototype, we'll try to support it via the websocket endpoint connecting to a robust state manager.
    # Or, we just run it here if it's not too long (it is long).
    
    # Actually, for the WebSocket to work, we need an active connection. 
    # The user connects via WS first, then triggers training? Or we broadcast.
    
    # Simplified approach: We'll just return "Training Started" and let the WS handle updates 
    # if we had a shared 'TrainingManager'.
    # For now, let's just trigger it.
    background_tasks.add_task(engine.train_model, config.epochs, config.batch_size)
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
            engine.train_model(epochs=config.epochs, batch_size=config.batch_size, progress_callback=send_progress)
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

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

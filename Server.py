import os
import asyncio
import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uuid
import logging

# Configuration
MIN_BOX_AREA = float(os.getenv("MIN_BOX_AREA", 1000))
WS_TIMEOUT = float(os.getenv("WS_TIMEOUT", 30.0))
MODEL_CONFIDENCE = float(os.getenv("MODEL_CONFIDENCE", 0.5))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Load ONNX model
try:
    ort_session = ort.InferenceSession(
    "best.onnx",
    providers=["CPUExecutionProvider"],
    sess_options=ort.SessionOptions())
    logger.info("ONNX model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load ONNX model: {str(e)}")
    raise

warmup_tensor = np.zeros((1, 3, 640, 640), dtype=np.float32)
ort_session.run(None, {"images": warmup_tensor})
logger.info("Model warmup complete")

# FastAPI app
app = FastAPI(title="ONNX Drug Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response schemas
class Detection(BaseModel):
    detection_id: str
    class_id: int
    leftmost: List[float]
    rightmost: List[float]
    notify_point: List[float]

class DetectionResponse(BaseModel):
    request_id: str
    timestamp: float
    processing_time_ms: float = 0
    detections: List[Detection] = []
    error: Optional[str] = None

@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    await websocket.accept()
    client_id = str(uuid.uuid4())[:8]
    logger.info(f"Client {client_id} connected")

    try:
        while True:
            request_id = str(uuid.uuid4())
            start_time = asyncio.get_event_loop().time()

            try:
                frame_bytes = await asyncio.wait_for(websocket.receive_bytes(), timeout=WS_TIMEOUT)
            except asyncio.TimeoutError:
                await websocket.send_json({"request_id": request_id, "error": f"Timeout after {WS_TIMEOUT}s", "timestamp": datetime.utcnow().timestamp()})
                continue
            except WebSocketDisconnect:
                logger.info(f"Client {client_id} disconnected")
                break
            except Exception as e:
                error_msg = f"Error receiving frame: {str(e)}"
                logger.error(f"Client {client_id}: {error_msg}")
                await websocket.send_json({
                    "request_id": request_id,
                    "error": error_msg,
                    "timestamp": datetime.utcnow().timestamp()
                })
                continue    


            try:
                np_arr = np.frombuffer(frame_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is None:
                    error_msg = "Invalid image format - failed to decode"
                    logger.warning(f"Client {client_id}: {error_msg}")
                    await websocket.send_json({
                        "request_id": request_id,
                        "error": error_msg,
                        "timestamp": datetime.utcnow().timestamp()
                    })
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_transposed = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
                input_tensor = np.expand_dims(img_transposed, axis=0)


                # Run inference
                try:
                    outputs = ort_session.run(None, {"images": input_tensor})
                except Exception as e:
                    error_msg = f"Inference error: {str(e)}"
                    logger.error(f"Client {client_id}: {error_msg}")
                    await websocket.send_json({
                        "request_id": request_id,
                        "error": error_msg,
                        "timestamp": datetime.utcnow().timestamp()
                    })
                    continue

                
                boxes = []
                predictions = outputs[0]
                predictions = predictions.squeeze().T
                scores = np.max(predictions[:, 4:], axis=1)
                mask = scores > MODEL_CONFIDENCE
                filtered_preds = predictions[mask]
                indices = cv2.dnn.NMSBoxes(
                filtered_preds[:, :4].tolist(),
                scores[mask].tolist(),
                MODEL_CONFIDENCE,
                0.5  # NMS threshold
                )

                for pred in indices:
                    try:
                        cls_id = np.argmax(filtered_preds[pred, 4:])
                        x, y, w, h = filtered_preds[pred, :4]

                        area = w * h
                        if area < MIN_BOX_AREA:
                            continue

                        leftmost = [x - w / 2, y + h /2]
                        rightmost = [x + w / 2, y - h /2]
                        notify_point = [x - w / 2 + 0.25 * w, y - h / 2 + 0.25 * h]

                        boxes.append(Detection(
                            detection_id=str(uuid.uuid4()),
                            class_id=cls_id,
                            leftmost=[float(leftmost[0]), float(leftmost[1])],
                            rightmost=[float(rightmost[0]), float(rightmost[1])],
                            notify_point=[float(notify_point[0]), float(notify_point[1])]
                        ))
                    except Exception as e:
                            logger.error(f"Client {client_id}: Error processing box: {str(e)}")
                            continue




                response = DetectionResponse(
                    request_id=request_id,
                    timestamp=datetime.utcnow().timestamp(),
                    processing_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000,
                    detections=boxes
                )
                await websocket.send_json(response.model_dump())

            except Exception as e:
                error_msg = f"Processing error: {str(e)}"
                logger.error(f"Client {client_id}: {error_msg}", exc_info=True)
                await websocket.send_json({
                    "request_id": request_id,
                    "error": error_msg,
                    "timestamp": datetime.utcnow().timestamp()
                })
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"Client {client_id} fatal error: {str(e)}")
    finally:
        logger.info(f"Client {client_id} connection closed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 3000)))

import asyncio
from fastapi import APIRouter, UploadFile, WebSocket, File, WebSocketDisconnect ,Depends
from fastapi.responses import JSONResponse
import logging
import cv2
import numpy as np
import base64
from helpers.configs import get_settings, Settings
from controllers.EmbeddingController import EmbeddingController

logger = logging.getLogger('uvicorn.error')

data_router = APIRouter(
    prefix=f"/AutoProctor/{get_settings().APP_VARIENT}/data",
    tags=["AutoProctor_v1"]
)

# Initialize the embedding controller globally
embedding_controller = None


def get_embedding_controller():
    global embedding_controller
    if embedding_controller is None:
        try:
            logger.info("Initializing EmbeddingController...")
            embedding_controller = EmbeddingController(
                DETECTION_MODEL=get_settings().DETECTION_MODEL,
                YOLOFACE_MODEL_PATH=get_settings().YOLOFACE_MODEL_PATH
            )
            logger.info("EmbeddingController initialized successfully")
            if not hasattr(embedding_controller, 'collection') or embedding_controller.collection is None:
                logger.error("Collection not initialized in EmbeddingController!")
                raise Exception("Collection initialization failed")
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingController: {e}")
            raise
    return embedding_controller


@data_router.post("/embed/{user_id}")
async def embed_frame_api(user_id: str, file: UploadFile):
    try:
        controller = get_embedding_controller()
        image = await file.read()
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("Failed to decode image")
            return JSONResponse(status_code=400, content={"message": "Invalid image format"})
        faces = controller.detect_faces(img)
        if not faces:
            return JSONResponse(status_code=404, content={"message": "No faces detected."})
        logger.info(f"Detected {len(faces)} face(s) for user_id: {user_id}")
        for idx, face in enumerate(faces):
            try:
                embedding = controller.get_embedding(face)
                metadata = {"user_id": user_id}
                controller.add_embedding(face, embedding, metadata)
                logger.info(f"Added embedding {idx + 1}/{len(faces)} for user_id: {user_id}")
            except Exception as e:
                logger.error(f"Error adding embedding {idx + 1} for user_id {user_id}: {e}")
                raise
        return {
            "message": f"Embeddings added for user_id: {user_id}",
            "num_faces": len(faces)
        }
    except Exception as e:
        logger.error(f"Error in embed_frame_api: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": f"Internal server error: {str(e)}"})



@data_router.post("/delete/{user_id}")
async def delete_embeddings_api(user_id: str):
    try:
        controller = get_embedding_controller()
        delete_result = controller.delete_embeddings_by_user(user_id)
        return {
            "message": f"Deleted embeddings for user_id: {user_id}",
            "details": delete_result
        }
    except Exception as e:
        logger.error(f"Error in delete_embeddings_api: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": f"Internal server error: {str(e)}"})


@data_router.post("/update/{user_id}")
async def update_embeddings_api(user_id: str, file: UploadFile, app_settings: Settings = Depends(get_settings)):
    try:
        controller = get_embedding_controller()
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(status_code=400, content={"message": "Invalid image format"})
        
        faces = controller.detect_faces(img)
        if not faces:
            return JSONResponse(status_code=404, content={"message": "No faces detected."})
        embeddings = [controller.get_embedding(face) for face in faces]
        metadata = {"user_id": user_id}
        controller.update_embeddings(
            user_id=user_id,
            faces=faces,
            embeddings=embeddings,
            metadata=metadata
        )
        return {
            "message": f"Embeddings updated for user_id: {user_id}",
            "num_faces": len(faces)
        }
    except Exception as e:
        logger.error(f"Error in update_embeddings_api: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": f"Internal server error: {str(e)}"})



@data_router.post("/detect/frame")
async def detect_frame_api(file: UploadFile = File(...), app_settings: Settings = Depends(get_settings)):
    controller = get_embedding_controller()
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"message": "Invalid image"})
    faces = controller.detect_faces(img)
    if not faces:
        return JSONResponse(status_code=404, content={"message": "No faces detected"})
    results = []
    for face in faces:
        embedding = controller.get_embedding(face)
        result = controller.query_embedding(
            embedding,
            n_results=app_settings.MAX_RESULTS,
            threshold=app_settings.SIMILARITY_THRESHOLD
        )
        results.append(result)
    return {
        "num_faces": len(faces),
        "results": results
    }


@data_router.websocket("/detect/stream")
async def detect_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected")
    controller = get_embedding_controller()
    frame_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    stop_event = asyncio.Event()
    frame_count = 0

    async def receiver():
        """Receive frames and keep ONLY the latest one"""
        try:
            while not stop_event.is_set():
                msg = await websocket.receive()

                if msg.get("type") == "websocket.disconnect":
                    break

                data = None
                if msg.get("bytes"):
                    data = msg["bytes"]
                elif msg.get("text"):
                    text = msg["text"]
                    if text.startswith("data:image"):
                        text = text.split(",", 1)[1]
                    data = base64.b64decode(text)

                if not data:
                    continue

                # Drop old frame if queue is full
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass

                await frame_queue.put(data)

        except WebSocketDisconnect:
            logger.info("Receiver: client disconnected")
        except Exception as e:
            logger.error(f"Receiver error: {e}", exc_info=True)
        finally:
            stop_event.set()

    async def processor():
        """Process ONLY the latest frame"""
        nonlocal frame_count
        try:
            while not stop_event.is_set():
                try:
                    data = await asyncio.wait_for(frame_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                if websocket.client_state.name != "CONNECTED":
                    break

                # Decode image
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                frame_count += 1

                # Detect faces
                try:
                    faces = controller.detect_faces(frame)
                    if not faces:
                        await websocket.send_json({
                            "frame": frame_count,
                            "faces_detected": 0
                        })
                        continue
                except Exception as e:
                    await websocket.send_json({"error": f"detection failed: {e}"})
                    continue

                # Process faces
                results = []
                for face in faces:
                    try:
                        emb = controller.get_embedding(face)
                        res = controller.query_embedding(
                            emb,
                            n_results=get_settings().MAX_RESULTS,
                            threshold=get_settings().SIMILARITY_THRESHOLD
                        )
                        results.append(res)
                    except Exception as e:
                        results.append({"error": str(e)})

                # Send results
                try:
                    await websocket.send_json({
                        "frame": frame_count,
                        "faces_detected": len(faces),
                        "results": results
                    })
                except Exception:
                    break

        except Exception as e:
            logger.error(f"Processor error: {e}", exc_info=True)
        finally:
            stop_event.set()

    # Run tasks
    recv_task = asyncio.create_task(receiver())
    proc_task = asyncio.create_task(processor())

    # Wait for receiver to finish (disconnect)
    await recv_task

    # Stop processor immediately
    proc_task.cancel()

    try:
        await proc_task
    except asyncio.CancelledError:
        pass

    try:
        await websocket.close()
    except Exception:
        pass

    logger.info(f"WebSocket closed (processed {frame_count} frames)")



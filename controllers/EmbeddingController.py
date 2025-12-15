import base64
import cv2
import os
from datetime import datetime
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
from helpers.db import get_chroma
import uuid

from helpers.Augmentions import FaceAugmentor

class EmbeddingController:
    def __init__(self, DETECTION_MODEL: str, YOLOFACE_MODEL_PATH=None):
        self.client, self.collection = get_chroma()
        self.detection_model = DETECTION_MODEL
        if DETECTION_MODEL == "yoloface":
            self.detector = YOLO(model=YOLOFACE_MODEL_PATH)
        else:
            self.detector = MTCNN(
                image_size=160,
                margin=10,                 # tight crop, small context
                min_face_size=20,          # allow smaller faces
                thresholds=[0.6, 0.7, 0.8], # higher recall, fewer misses
                factor=0.709,
                post_process=True,
                keep_all=True,        
                device=torch.device('cpu')
            )

        self.facenet = InceptionResnetV1(pretrained="vggface2").eval().to("cpu")
        self.augmentor = FaceAugmentor()

    def detect_faces(self, image):
        if isinstance(self.detector, YOLO):
            results = self.detector(image)
            boxes = results[0].boxes.xyxy.cpu().numpy()
        else:
            boxes, _ = self.detector.detect(image)
            if boxes is None:
                return []
        faces = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = image[y1:y2, x1:x2]
            if face.size > 0:
                faces.append(face)
        return faces

    def get_embedding(self, face):
        try:
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        except Exception:
            face_rgb = face
        face_resized = cv2.resize(face_rgb, (160, 160))
        face_tensor = torch.tensor(face_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            embedding = self.facenet(face_tensor.to("cpu")).cpu().numpy()
        return embedding.flatten()

    def face_to_base64(self, face):
        _, buffer = cv2.imencode('.jpg', face)
        return base64.b64encode(buffer).decode("utf-8")

    def save_cropped_face(self, face, user_id: str = None, idx: int = 0):
        try:
            out_dir = os.path.join(os.getcwd(), 'static', 'crops')
            os.makedirs(out_dir, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            user_part = user_id if user_id else 'unknown'
            filename = f"{user_part}_{self.detection_model}_{idx}_{ts}.jpg"
            path = os.path.join(out_dir, filename)
            cv2.imwrite(path, face)
            return path
        except Exception:
            return None

    def add_embedding(self, face, embedding, metadata: dict):
        user_id = metadata["user_id"]
        record_id = f"{user_id}_{uuid.uuid4().hex}"
        face_b64 = self.face_to_base64(face)

        # try:
        #     self.save_cropped_face(face, user_id=user_id, idx=0)
        # except Exception:
        #     pass

        embedding = embedding / np.linalg.norm(embedding)
        self.collection.add(
            ids=[record_id],
            embeddings=[embedding.tolist()],
            documents=[face_b64],
            metadatas=[metadata]
        )


        aug_faces = self.augmentor.generate(face)

        for i, aug_face in enumerate(aug_faces):
            aug_embedding = self.get_embedding(aug_face)
            aug_metadata = metadata.copy()
            aug_metadata["augmented"] = True
            aug_id = f"{user_id}_aug_{i}_{uuid.uuid4().hex}"
            # try:
            #     self.save_cropped_face(aug_face, user_id=aug_id, idx=i)
            # except Exception:
            #     pass

            aug_embedding = aug_embedding / np.linalg.norm(aug_embedding)
            self.collection.add(
                ids=[aug_id],
                embeddings=[aug_embedding.tolist()],
                documents=[self.face_to_base64(aug_face)],
                metadatas=[aug_metadata]
            )

    def update_embeddings(self, user_id: str, faces: list, embeddings: list, metadata: dict = None):
        try:
            self.collection.delete(where={"user_id": user_id})
        except Exception:
            pass

        for idx, (face, emb) in enumerate(zip(faces, embeddings)):
            meta = metadata.copy() if metadata else {}
            meta.update({"user_id": user_id})
            # try:
            #     self.save_cropped_face(face, user_id=user_id, idx=idx)
            # except Exception:
            #     pass

            record_id = f"{user_id}_{idx}_{datetime.now().timestamp()}"

            emb = emb / np.linalg.norm(emb)
            self.collection.add(
                ids=[record_id],
                embeddings=[emb.tolist()],
                documents=[self.face_to_base64(face)],
                metadatas=[meta]
            )

            aug_faces = self.augmentor.generate(face)
            for j, aug_face in enumerate(aug_faces):
                aug_embedding = self.get_embedding(aug_face)
                aug_meta = meta.copy()
                aug_meta["augmented"] = True
                aug_id = f"{user_id}_upd_aug_{j}_{uuid.uuid4().hex}"
                aug_embedding = aug_embedding / np.linalg.norm(aug_embedding)
                self.collection.add(
                    ids=[aug_id],
                    embeddings=[aug_embedding.tolist()],
                    documents=[self.face_to_base64(aug_face)],
                    metadatas=[aug_meta]
                )

    def delete_embeddings_by_user(self, user_id: str):
        try:
            self.collection.delete(where={"user_id": user_id})
            return True
        except Exception as e:
            print("Deletion error:", e)
            return False


    def query_embedding(self, embedding, n_results=5, threshold=0.6):
        embedding = embedding / np.linalg.norm(embedding)
        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=n_results
        )
        if not results or not results.get("distances"):
            return {
                "match": False,
                "reason": "No results from database"
            }

        distances = results["distances"][0]
        metadatas = results["metadatas"][0]
        if not distances or not metadatas:
            return {
                "match": False,
                "reason": "Empty results from database"
            }
        best_distance = min(distances)
        best_index = distances.index(best_distance)
        best_metadata = metadatas[best_index]
        similarity = 1 - best_distance
        if similarity >= threshold:
            return {
                "match": True,
                "user_id": best_metadata.get("user_id"),
                "similarity": round(similarity, 5),
                "metadata": best_metadata
            }

        return {
            "match": False,
            "similarity": round(similarity, 5)
        }


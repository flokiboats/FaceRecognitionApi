from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str 
    APP_VERSION: str 
    APP_VARIENT: str
    
    
    DETECTION_MODEL:str = "mtcnn"  # Options: mtcnn, yoloface

    YOLOFACE_MODEL_PATH: str = "assets/yolov12n-face.pt"
    CHROMA_DB_PATH:str = "./chroma_data"
    COLLECTION_NAME:str = "face_embeddings_collection"
    SIMILARITY_THRESHOLD:float = 0.4
    MAX_RESULTS:int = 2
    class Config: 
        env_file = ".env" 

def get_settings(): ## this makes any got by "get_settings().APP_NAME" 
    return Settings()
from fastapi import APIRouter , Depends
from helpers.configs import Settings , get_settings
from helpers.db import get_chroma
from fastapi.responses import FileResponse


base_router = APIRouter(
    prefix=f"/AutoProctor/{get_settings().APP_VARIENT}",
    tags=["AutoProctor_v1"]) 


@base_router.get("/") 
async def welcome(app_settings: Settings = Depends(get_settings)):
        return FileResponse('static/index.html')
        # app_name = app_settings.APP_NAME
        # app_varient=app_settings.APP_VARIENT
        # app_version = app_settings.APP_VERSION
        # return {"app_name": app_name, "app_version": app_version ,"app_varient":app_varient, "status": "healthy"}


@base_router.get('/config')
async def config(app_settings: Settings = Depends(get_settings)):
        return {
                'app_name': app_settings.APP_NAME,
                'app_version': app_settings.APP_VERSION,
                'app_variant': app_settings.APP_VARIENT,
                'detection_model': app_settings.DETECTION_MODEL,
                'yoloface_model_path': app_settings.YOLOFACE_MODEL_PATH,
                'chroma_db_path': app_settings.CHROMA_DB_PATH,
                'collection_name': app_settings.COLLECTION_NAME,
                'similarity_threshold': app_settings.SIMILARITY_THRESHOLD,
                'max_results': app_settings.MAX_RESULTS
        }


@base_router.get("/health") 
async def health(app_settings: Settings = Depends(get_settings)):
    app_name = app_settings.APP_NAME
    app_version = app_settings.APP_VERSION
    return {"app_name": app_name, "app_version": app_version , "status": "healthy"}


@base_router.get("/count")
def count_documents():
    _, collection = get_chroma()
    return {"documents": collection.count()}
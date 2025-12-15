from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager

from routes import base, data
from helpers.db import init_chroma, chroma_client
from helpers.configs import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔵 Initializing ChromaDB...")
    init_chroma()
    print("🟢 ChromaDB Ready.")

    yield
    if chroma_client is not None:
        try:
            chroma_client.persist()
            print("🔴 ChromaDB persisted & closed.")
        except:
            pass


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(base.base_router)
app.include_router(data.data_router)

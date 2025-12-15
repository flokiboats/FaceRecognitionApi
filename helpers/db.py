import chromadb
from chromadb.config import Settings
from helpers.configs import get_settings

# Globals
chroma_client = None
chroma_collection = None



def init_chroma():
    global chroma_client, chroma_collection
    settings = get_settings()

    chroma_client = chromadb.PersistentClient(
        path=settings.CHROMA_DB_PATH
    )

    chroma_collection = chroma_client.get_or_create_collection(
        name=settings.COLLECTION_NAME,
        #        metadata={"hnsw:space": "cosine"}
        metadata={"hnsw:space": "cosine"}
    )

    return chroma_client, chroma_collection


def get_chroma():
    return chroma_client, chroma_collection

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rag_chat_app.backend.routes.chat import router as chat_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG Chat API"}
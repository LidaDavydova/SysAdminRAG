from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from rag_chat_app.backend.models.rag_model import RAGModel
import logging

router = APIRouter()

# Initialize the RAG system
rag_system = RAGModel()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    user_input: str

class ChatResponse(BaseModel):
    answer: str
    sources: list


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received user input: {request.user_input}")
        # Use the RAG system to get an answer
        result = rag_system.generate_response(request.user_input)
        logger.info(f"Generated response: {result}")
        return ChatResponse(answer=result["answer"],
                            sources=result['sources'])
    except Exception as e:
        logger.error(f"Error during chat processing: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": "An error occurred while processing your request."})
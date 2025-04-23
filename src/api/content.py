"""Content generation API endpoints"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from pydantic import BaseModel
from ..core.content_generation import content_generator
from ..core.circuit_breaker import CircuitBreaker

router = APIRouter()
circuit_breaker = CircuitBreaker()

class ContentRequest(BaseModel):
    """Content generation request model"""
    query: str
    tone: str = "professional"
    format_type: str = "blog"
    target_audience: str = "general"
    length: str = "medium"

class ContentResponse(BaseModel):
    """Content generation response model"""
    content: str
    metadata: Dict[str, Any]

@router.post("/generate", response_model=ContentResponse)
async def generate_content(request: ContentRequest):
    """
    Generate content using RAG and knowledge graph
    
    Args:
        request: Content generation request
        
    Returns:
        Generated content and metadata
        
    Raises:
        HTTPException: If generation fails
    """
    try:
        result = circuit_breaker.execute(
            content_generator.generate_content,
            request.query,
            request.tone,
            request.format_type,
            request.target_audience,
            request.length
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return ContentResponse(
            content=result["content"],
            metadata=result["metadata"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"} 
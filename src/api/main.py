"""Main FastAPI application"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .content import router as content_router

app = FastAPI(
    title="SocialSynth AI API",
    description="AI-powered content generation API",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(content_router, prefix="/api/content", tags=["content"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to SocialSynth AI API",
        "version": "1.0.0",
        "docs_url": "/docs"
    } 
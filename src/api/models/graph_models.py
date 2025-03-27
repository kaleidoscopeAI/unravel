from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class GraphAnalysisRequest(BaseModel):
    """Request model for code graph analysis"""
    decompiled_dir: str = Field(..., description="Directory containing decompiled files to analyze")
    use_llm: bool = Field(True, description="Whether to use LLM for enhanced analysis")
    
    class Config:
        schema_extra = {
            "example": {
                "decompiled_dir": "/path/to/decompiled/files",
                "use_llm": True
            }
        }

class GraphAnalysisResponse(BaseModel):
    """Response model for code graph analysis"""
    analysis_id: str = Field(..., description="Unique ID for this analysis job")
    status: str = Field(..., description="Status of the analysis (processing, completed, failed)")
    message: str = Field(..., description="Status message")
    results_url: Optional[str] = Field(None, description="URL to fetch results when completed")
    
    class Config:
        schema_extra = {
            "example": {
                "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "processing",
                "message": "Analysis in progress",
                "results_url": None
            }
        }

class GraphNode(BaseModel):
    """Model for a node in the code graph"""
    id: str
    label: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    
class GraphEdge(BaseModel):
    """Model for an edge in the code graph"""
    source: str
    target: str
    type: str
    
class CodeGraph(BaseModel):
    """Model for the complete code graph"""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    properties: Dict[str, Any] = Field(default_factory=dict)

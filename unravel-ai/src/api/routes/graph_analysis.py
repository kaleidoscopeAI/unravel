from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Dict, Any, Optional
import os
import uuid

from ...core.code_graph_analyzer import analyze_codebase, CodeGraphBuilder
from ...utils.config import Config
from ..models.graph_models import GraphAnalysisRequest, GraphAnalysisResponse

router = APIRouter(
    prefix="/api/v1/graph",
    tags=["code-graph"],
    responses={404: {"description": "Not found"}},
)

config = Config()

@router.post("/analyze", response_model=GraphAnalysisResponse)
async def analyze_code(
    request: GraphAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Analyze code files and generate a code structure graph.
    This endpoint initiates the analysis as a background task.
    """
    # Generate unique ID for this analysis job
    analysis_id = str(uuid.uuid4())
    
    # Create output directory
    output_dir = os.path.join(config.WORK_DIR, "graph_analysis", analysis_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Run analysis in background
    background_tasks.add_task(
        analyze_codebase,
        request.decompiled_dir,
        output_dir,
        request.use_llm
    )
    
    return GraphAnalysisResponse(
        analysis_id=analysis_id,
        status="processing",
        message="Analysis started. Use the /api/v1/graph/status/{analysis_id} endpoint to check status."
    )

@router.get("/status/{analysis_id}", response_model=GraphAnalysisResponse)
async def get_analysis_status(
    analysis_id: str = Path(..., description="ID of the analysis job")
):
    """Get the status of a code graph analysis job"""
    output_dir = os.path.join(config.WORK_DIR, "graph_analysis", analysis_id)
    
    # Check if the directory exists
    if not os.path.exists(output_dir):
        raise HTTPException(status_code=404, detail=f"Analysis job {analysis_id} not found")
    
    # Check for report file to determine if analysis is complete
    report_path = os.path.join(output_dir, "code_analysis_report.md")
    if os.path.exists(report_path):
        return GraphAnalysisResponse(
            analysis_id=analysis_id,
            status="completed",
            message="Analysis completed",
            results_url=f"/api/v1/graph/results/{analysis_id}"
        )
    
    return GraphAnalysisResponse(
        analysis_id=analysis_id,
        status="processing",
        message="Analysis in progress"
    )

@router.get("/results/{analysis_id}")
async def get_analysis_results(
    analysis_id: str = Path(..., description="ID of the analysis job"),
    format: str = Query("json", description="Format of the results (json, report, visualization)")
):
    """Get the results of a code graph analysis job"""
    output_dir = os.path.join(config.WORK_DIR, "graph_analysis", analysis_id)
    
    # Check if the directory exists
    if not os.path.exists(output_dir):
        raise HTTPException(status_code=404, detail=f"Analysis job {analysis_id} not found")
    
    # Check for report file to determine if analysis is complete
    report_path = os.path.join(output_dir, "code_analysis_report.md")
    if not os.path.exists(report_path):
        return JSONResponse(
            status_code=202,
            content={"message": "Analysis still in progress"}
        )
    
    # Return requested format
    if format == "report":
        return FileResponse(report_path)
    elif format == "visualization":
        vis_path = os.path.join(output_dir, "code_graph.png")
        if os.path.exists(vis_path):
            return FileResponse(vis_path)
        else:
            raise HTTPException(status_code=404, detail="Visualization not found")
    else:  # JSON is default
        graph_path = os.path.join(output_dir, "code_graph.json")
        if os.path.exists(graph_path):
            return FileResponse(graph_path)
        else:
            raise HTTPException(status_code=404, detail="Graph data not found")

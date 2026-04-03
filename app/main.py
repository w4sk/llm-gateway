from fastapi import FastAPI, Request, HTTPException, Security
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import httpx
import os
import glob
import json

app = FastAPI(
    title="LLM Gateway",
    description="Proxy gateway for vLLM server. Provides OpenAI-compatible API for models like Qwen.",
    version="1.0.0"
)

VLLM_URL = os.getenv("VLLM_URL", "http://vllm:8000")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# Configure HTTPBearer for Swagger UI
security = HTTPBearer(auto_error=False)

client = httpx.AsyncClient(base_url=VLLM_URL, timeout=None)

# --- Data Models (OpenAI Compatible) ---

class Message(BaseModel):
    role: str = Field(..., description="roles: system, user, assistant")
    content: Union[str, List[Dict[str, Any]]] = Field(..., description="Message content")

class ChatCompletionRequest(BaseModel):
    model: str = Field(DEFAULT_MODEL, description="Model repository name")
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "model": DEFAULT_MODEL,
                "messages": [
                    {"role": "user", "content": "Hello, please introduce yourself."}
                ],
                "stream": False
            }
        }

# --- Internal Helper ---

def get_proxy_headers(raw_headers: Dict[str, str], auth: Optional[HTTPAuthorizationCredentials] = None) -> Dict[str, str]:
    headers = {k.lower(): v for k, v in raw_headers.items()}
    headers.pop("host", None)
    headers.pop("content-length", None)
    
    # 1. Custom headers (e.g. from curl)
    if "authorization" in headers:
        return headers
    
    # 2. From Swagger UI Authorize button
    if auth:
        headers["authorization"] = f"Bearer {auth.credentials}"
        return headers
    
    # 3. Fallback to Gateway's internal API key
    if VLLM_API_KEY:
        headers["authorization"] = f"Bearer {VLLM_API_KEY}"
        
    return headers

# --- Endpoints ---

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/v1/models")
async def list_models(
    raw_request: Request,
    auth: Optional[HTTPAuthorizationCredentials] = Security(security)
):
    """List available models."""
    headers = get_proxy_headers(dict(raw_request.headers), auth)
    try:
        response = await client.get("/v1/models", headers=headers)
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest, 
    raw_request: Request,
    auth: Optional[HTTPAuthorizationCredentials] = Security(security)
):
    """
    OpenAI-compatible Chat Completions endpoint.
    Allows testing Qwen models directly from /docs.
    """
    body = request.dict(exclude_none=True)
    headers = get_proxy_headers(dict(raw_request.headers), auth)
    
    try:
        req = client.build_request(
            method="POST",
            url="/v1/chat/completions",
            json=body,
            headers=headers
        )
        rp = await client.send(req, stream=True)
        
        if request.stream:
            return StreamingResponse(
                rp.aiter_raw(),
                status_code=rp.status_code,
                headers=dict(rp.headers)
            )
        else:
            await rp.aread()
            return rp.json()
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/scenefun3d/results")
async def list_scenefun3d_results():
    """List available SceneFun3D results."""
    # Use relative path from /app
    output_dir = "output/scenefun3d"
    pattern = os.path.join(output_dir, "*_selected.json")
    files = glob.glob(pattern)
    results = []
    for f in files:
        basename = os.path.basename(f)
        # Expected filename: {visit_id}_{video_id}_selected.json
        name_part = basename.replace("_selected.json", "")
        parts = name_part.split("_")
        if len(parts) >= 2:
            results.append({
                "visit_id": parts[0],
                "video_id": parts[1]
            })
    return results

@app.get("/v1/scenefun3d/results/{visit_id}/{video_id}")
async def get_scenefun3d_result(visit_id: str, video_id: str):
    """Get the detail of a specific SceneFun3D result."""
    file_path = f"output/scenefun3d/{visit_id}_{video_id}_selected.json"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Result file not found")
    
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading result: {str(e)}")

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"], include_in_schema=False)
async def proxy_fallback(request: Request, path: str):
    """Forward requests to undefined endpoints directly."""
    url = f"/{path}"
    params = dict(request.query_params)
    content = await request.body()
    headers = get_proxy_headers(dict(request.headers)) # Fallback auth included
    
    try:
        req = client.build_request(
            method=request.method,
            url=url,
            content=content,
            headers=headers,
            params=params
        )
        rp = await client.send(req, stream=True)
        return StreamingResponse(
            rp.aiter_raw(),
            status_code=rp.status_code,
            headers=dict(rp.headers)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()

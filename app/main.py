from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List
import base64
import time

from app.process import process_id_image

app = FastAPI(
    title="KYC ID Verification API",
    description="Automated KYC & ID verification using OpenCV + Tesseract OCR",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class KYCResponse(BaseModel):
    success: bool
    processing_time_ms: float
    document_type: str
    extracted_name: str
    id_number: str
    dates_found: List[str]
    # address: str
    card_detected: bool
    raw_text: Optional[str] = None
    error: Optional[str] = None


@app.get("/")
async def root():
    return FileResponse("templates/index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/verify", response_model=KYCResponse)
async def verify_id(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Please upload JPEG or PNG.",
        )

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max 10 MB.")

    start = time.perf_counter()
    result = process_id_image(image_bytes)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if "error" in result:
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "processing_time_ms": elapsed_ms,
                "error": result["error"],
                "document_type": "Unknown",
                "extracted_name": "Not Found",
                "id_number": "Not Found",
                "dates_found": [],
                "card_detected": False,
            },
        )

    return KYCResponse(
        success=True,
        processing_time_ms=round(elapsed_ms, 2),
        document_type=result.get("document_type", "Unknown"),
        extracted_name=result.get("extracted_name", "Not Found"),
        id_number=result.get("id_number", "Not Found"),
        dates_found=result.get("dates_found", []),
        card_detected=result.get("card_detected", False),
        raw_text=result.get("raw_text", ""),
    )

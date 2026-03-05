

import structlog
from fastapi import APIRouter, HTTPException

from app.models.analysis import AnalysisRequest, AnalysisResult
from app.services.intelligence_service import IntelligenceService

log = structlog.get_logger()
router = APIRouter(tags=["Intelligence"])
svc = IntelligenceService()

@router.post("/analyze", response_model=AnalysisResult, status_code=200)
async def analyze(req: AnalysisRequest):

    try:
        result = await svc.analyze(req)
        log.info("analysis.complete",
                 call_id=req.call_id,
                 sentiment=result.sentiment.label,
                 intent=result.intent.intent,
                 escalation=result.escalation_risk)
        return result
    except Exception as e:
        log.error("analysis.error", call_id=req.call_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/batch", response_model=list[AnalysisResult])
async def analyze_batch(requests: list[AnalysisRequest]):

    import asyncio
    results = await asyncio.gather(*[svc.analyze(r) for r in requests], return_exceptions=True)
    out = []
    for req, res in zip(requests, results):
        if isinstance(res, Exception):
            log.error("batch.analysis.error", call_id=req.call_id, error=str(res))
            raise HTTPException(status_code=500, detail=f"Error on call {req.call_id}: {res}")
        out.append(res)
    return out

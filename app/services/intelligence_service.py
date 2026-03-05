

import asyncio
import re
from functools import lru_cache

import structlog
from transformers import pipeline

from app.models.analysis import (
    AnalysisRequest,
    AnalysisResult,
    Entity,
    EscalationRisk,
    Intent,
    IntentResult,
    Sentiment,
    SentimentResult,
)

log = structlog.get_logger()

_ESCALATION_CRITICAL = [
    "cancelar contrato", "hablar con el gerente", "voy a denunciar",
    "ya me cansé", "es un fraude", "voy a publicar",
]
_ESCALATION_HIGH = [
    "no me han resuelto", "llevo días llamando", "quiero hablar con supervisor",
    "esto es inaceptable", "exijo",
]

_INTENT_LABELS = {
    "soporte técnico": Intent.TECHNICAL_SUPPORT,
    "problema con factura o cobro": Intent.BILLING,
    "cancelar servicio": Intent.CANCELLATION,
    "comprar o contratar": Intent.SALES,
    "queja o reclamación": Intent.COMPLAINT,
    "consulta general": Intent.GENERAL_INQUIRY,
    "transferir a otro agente": Intent.TRANSFER_REQUEST,
}

@lru_cache(maxsize=1)
def _load_sentiment_pipeline():
    log.info("intelligence.loading_sentiment_model")
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    )

@lru_cache(maxsize=1)
def _load_zero_shot_pipeline():
    log.info("intelligence.loading_zero_shot_model")
    return pipeline(
        "zero-shot-classification",
        model="joeddav/xlm-roberta-large-xnli",
    )

@lru_cache(maxsize=1)
def _load_ner_pipeline():
    log.info("intelligence.loading_ner_model")
    return pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple",
    )

def _map_sentiment(label: str, score: float) -> SentimentResult:
    mapping = {
        "LABEL_0": Sentiment.NEGATIVE,
        "negative": Sentiment.NEGATIVE,
        "LABEL_1": Sentiment.NEUTRAL,
        "neutral": Sentiment.NEUTRAL,
        "LABEL_2": Sentiment.POSITIVE,
        "positive": Sentiment.POSITIVE,
    }
    return SentimentResult(
        label=mapping.get(label.upper(), Sentiment.NEUTRAL),
        score=round(score, 4),
    )

def _detect_escalation(text: str, sentiment: SentimentResult) -> EscalationRisk:
    text_lower = text.lower()
    if any(phrase in text_lower for phrase in _ESCALATION_CRITICAL):
        return EscalationRisk.CRITICAL
    if any(phrase in text_lower for phrase in _ESCALATION_HIGH):
        return EscalationRisk.HIGH
    if sentiment.label == Sentiment.NEGATIVE and sentiment.score > 0.85:
        return EscalationRisk.MEDIUM
    return EscalationRisk.LOW

def _extract_key_phrases(text: str) -> list[str]:

    words = re.findall(r'\b[A-Za-záéíóúñÁÉÍÓÚÑ]{5,}\b', text)
    stop = {"quiero", "necesito", "tengo", "puede", "podría", "favor", "buenas", "hola"}
    return list({w.lower() for w in words if w.lower() not in stop})[:10]

class IntelligenceService:
    def __init__(self):
        self._loop = asyncio.get_event_loop()

    async def analyze(self, req: AnalysisRequest) -> AnalysisResult:

        text = req.transcript
        if not text or not text.strip():
            return self._empty_result(req)

        sentiment_res, intent_res, entities = await asyncio.gather(
            self._run_sentiment(text),
            self._run_intent(text),
            self._run_ner(text),
        )

        return AnalysisResult(
            call_id=req.call_id,
            chunk_id=req.chunk_id,
            transcript=text,
            language=req.language,
            sentiment=sentiment_res,
            intent=intent_res,
            entities=entities,
            escalation_risk=_detect_escalation(text, sentiment_res),
            key_phrases=_extract_key_phrases(text),
        )

    async def _run_sentiment(self, text: str) -> SentimentResult:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: _load_sentiment_pipeline()(text[:512]))
        r = result[0]
        return _map_sentiment(r["label"], r["score"])

    async def _run_intent(self, text: str) -> IntentResult:
        loop = asyncio.get_event_loop()
        labels = list(_INTENT_LABELS.keys())
        result = await loop.run_in_executor(
            None,
            lambda: _load_zero_shot_pipeline()(text[:512], labels, multi_label=True)
        )
        scores = dict(zip(result["labels"], result["scores"]))
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        secondary = [
            (_INTENT_LABELS.get(lbl, Intent.UNKNOWN), round(s, 4))
            for lbl, s in list(scores.items())[1:4]
        ]
        return IntentResult(
            intent=_INTENT_LABELS.get(top_label, Intent.UNKNOWN),
            confidence=round(top_score, 4),
            secondary_intents=secondary,
        )

    async def _run_ner(self, text: str) -> list[Entity]:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, lambda: _load_ner_pipeline()(text[:512])
        )
        return [
            Entity(
                text=r["word"],
                label=r["entity_group"],
                start=r["start"],
                end=r["end"],
                confidence=round(r["score"], 4),
            )
            for r in results
        ]

    def _empty_result(self, req: AnalysisRequest) -> AnalysisResult:
        return AnalysisResult(
            call_id=req.call_id,
            chunk_id=req.chunk_id,
            transcript=req.transcript,
            language=req.language,
            sentiment=SentimentResult(label=Sentiment.NEUTRAL, score=0.0),
            intent=IntentResult(intent=Intent.UNKNOWN, confidence=0.0),
        )

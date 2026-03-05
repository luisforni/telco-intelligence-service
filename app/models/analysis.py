

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class Intent(str, Enum):
    TECHNICAL_SUPPORT = "technical_support"
    BILLING = "billing"
    CANCELLATION = "cancellation"
    SALES = "sales"
    COMPLAINT = "complaint"
    GENERAL_INQUIRY = "general_inquiry"
    TRANSFER_REQUEST = "transfer_request"
    UNKNOWN = "unknown"

class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    confidence: float = Field(ge=0.0, le=1.0)

class SentimentResult(BaseModel):
    label: Sentiment
    score: float = Field(ge=0.0, le=1.0)
    mixed: bool = False

class IntentResult(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    secondary_intents: list[tuple[Intent, float]] = []

class EscalationRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AnalysisResult(BaseModel):
    call_id: str
    chunk_id: Optional[str] = None
    transcript: str
    language: str = "es"
    sentiment: SentimentResult
    intent: IntentResult
    entities: list[Entity] = []
    escalation_risk: EscalationRisk = EscalationRisk.LOW
    key_phrases: list[str] = []
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)

class AnalysisRequest(BaseModel):
    call_id: str
    chunk_id: Optional[str] = None
    transcript: str
    language: str = "es"
    context_chunks: list[str] = []

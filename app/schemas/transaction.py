from pydantic import BaseModel, Field
from typing import Optional


class TransactionRequest(BaseModel):
    TransactionAmt: float = Field(..., gt=0)
    ProductCD: str
    card4: Optional[str] = None
    card6: Optional[str] = None
    P_emaildomain: Optional[str] = None


class TransactionResponse(BaseModel):
    fraud_probability: float
    risk_bucket: str
    model_version: str
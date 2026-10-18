from fastapi import APIRouter, Request
from app.schemas.transaction import TransactionRequest, TransactionResponse
from app.services.fraud_service import predict_transaction

router = APIRouter()


@router.post("/", response_model=TransactionResponse)
def predict(payload: TransactionRequest, request: Request, model: str = "xgboost"):
    predictor = request.app.state.predictor
    return predict_transaction(payload, predictor, model_choice=model)
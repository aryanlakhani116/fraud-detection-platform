from app.schemas.transaction import TransactionRequest, TransactionResponse


def predict_transaction(payload: TransactionRequest, predictor, model_choice: str) -> TransactionResponse:
    transaction_dict = payload.model_dump()

    if model_choice == "ann":
        result = predictor.predict_ann(transaction_dict)
    else:
        result = predictor.predict_xgboost(transaction_dict)

    return TransactionResponse(**result)
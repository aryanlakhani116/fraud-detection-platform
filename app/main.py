from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.inference.tabular_predictor import TabularFraudPredictor
from app.routes import transaction


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")
    app.state.predictor = TabularFraudPredictor()
    print("Models loaded.")

    yield

    print("Shutting down.")
    app.state.predictor = None


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(transaction.router, prefix="/predict/transaction", tags=["transaction"])
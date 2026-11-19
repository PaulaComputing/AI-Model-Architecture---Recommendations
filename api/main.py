import os
import json
import time
import torch
import torch.nn as nn
import redis
import pika
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from pydantic import BaseModel, Field

app = FastAPI(
    title="API Sistema de Recomendación E-commerce (AWS EC2)",
    description="Inferencia PyTorch NeuMF en tiempo real con Caché Redis y Eventos RabbitMQ",
    version="3.0.0"
)

# Arquitectura de la red neuronal (debe coincidir con train_and_package.py)
class RecommenderNet(nn.Module):
    def __init__(self, num_users: int, num_items: int, emb_dim: int = 32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, user_idx, item_idx):
        u = self.user_emb(user_idx)
        i = self.item_emb(item_idx)
        return self.fc(torch.cat([u, i], dim=-1)).squeeze(-1)

# Carga del modelo y metadatos locales al iniciar la API
with open("metadata.json", "r", encoding="utf-8") as f:
    META = json.load(f)

MODEL = RecommenderNet(
    num_users=META["num_users"],
    num_items=META["num_items"],
    emb_dim=META.get("emb_dim", 32)
)
MODEL.load_state_dict(torch.load("model.pth", map_location="cpu"))
MODEL.eval()

API_KEY = os.getenv("API_KEY", "duoc-ep3-secret-key-2026")
cache = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=6379,
    decode_responses=True
)

def verify_security(request: Request, x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida")
    rate_key = f"rate:{request.client.host}:{int(time.time() // 60)}"
    if cache.incr(rate_key) == 1:
        cache.expire(rate_key, 60)
    if int(cache.get(rate_key)) > 300:
        raise HTTPException(status_code=429, detail="Rate limit excedido")

class RecommendationRequest(BaseModel):
    user_id: int = Field(..., ge=1, le=100000, example=42)
    top_k: int = Field(5, ge=1, le=20, example=5)

class EventRequest(BaseModel):
    user_id: int = Field(..., ge=1, example=42)
    product_id: str = Field(..., example="SKU-1004")
    event_type: str = Field(..., pattern="^(view|add_to_cart|purchase)$", example="purchase")

@app.get("/health", tags=["Operaciones"])
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "catalog_size": len(META.get("catalog", []))
    }

@app.post("/api/v1/recommendations", tags=["Inferencia IA"], dependencies=[Depends(verify_security)])
def get_recommendations(payload: RecommendationRequest):
    start = time.perf_counter()
    cache_key = f"rec:user:{payload.user_id}:k:{payload.top_k}"

    # 1. Revisar si ya está en caché Redis
    cached = cache.get(cache_key)
    if cached:
        data = json.loads(cached)
        data["source"] = "Redis-Cache"
        data["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return data

    # 2. Inferencia en tiempo real con PyTorch en CPU
    uid = payload.user_id % META["num_users"]
    catalog = META["catalog"]
    num_catalog = len(catalog)

    with torch.no_grad():
        user_tensor = torch.full((num_catalog,), uid, dtype=torch.long)
        item_tensor = torch.arange(num_catalog, dtype=torch.long)
        scores = MODEL(user_tensor, item_tensor).numpy()

    ranked = sorted(zip(catalog, scores.tolist()), key=lambda x: x[1], reverse=True)[:payload.top_k]

    result = {
        "user_id": payload.user_id,
        "source": "PyTorch-Inference-EC2",
        "recommendations": [
            {
                "product_id": item["id"],
                "name": item["name"],
                "category": item["category"],
                "price": item["price"],
                "score": round(score, 4),
                "reason": f"Afinidad neuronal ({round(score * 100, 1)}%) con {item['category']}"
            }
            for item, score in ranked
        ]
    }

    cache.setex(cache_key, 300, json.dumps(result))
    result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
    return result

@app.post("/api/v1/events", tags=["Eventos Distribuidos"], dependencies=[Depends(verify_security)])
def publish_user_event(event: EventRequest):
    conn = pika.BlockingConnection(
        pika.ConnectionParameters(host=os.getenv("RABBITMQ_HOST", "rabbitmq"))
    )
    ch = conn.channel()
    ch.queue_declare(queue="user_events", durable=True)
    ch.basic_publish(
        exchange="",
        routing_key="user_events",
        body=event.model_dump_json(),
        properties=pika.BasicProperties(delivery_mode=2)
    )
    conn.close()
    return {"status": "queued", "broker": "RabbitMQ", "event": event.model_dump()}
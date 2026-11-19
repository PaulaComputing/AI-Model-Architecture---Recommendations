# Script para el Endpoint de AWS SageMaker

import json
import os
import torch
import torch.nn as nn

class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=32):
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
        x = torch.cat([u, i], dim=-1)
        return self.fc(x).squeeze(-1)

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(model_dir, "metadata.json"), "r") as f:
        meta = json.load(f)
    model = RecommenderNet(meta["num_users"], meta["num_items"], meta["emb_dim"])
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth"), map_location=device))
    model.eval()
    return {"model": model, "meta": meta}

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        return json.loads(request_body)
    raise ValueError(f"Content-type no soportado: {request_content_type}")

def predict_fn(input_data, model_dict):
    model = model_dict["model"]
    meta = model_dict["meta"]
    user_id = int(input_data.get("user_id", 1)) % meta["num_users"]
    top_k = int(input_data.get("top_k", 5))
    catalog = meta["catalog"]

    with torch.no_grad():
        item_indices = torch.arange(len(catalog), dtype=torch.long)
        user_indices = torch.full((len(catalog),), user_id, dtype=torch.long)
        scores = model(user_indices, item_indices).numpy()

    ranked = sorted(zip(catalog, scores.tolist()), key=lambda x: x[1], reverse=True)[:top_k]
    return {
        "user_id": user_id,
        "recommendations": [
            {
                "product_id": item["id"],
                "name": item["name"],
                "category": item["category"],
                "price": item["price"],
                "score": round(score, 4),
                "reason": f"Afinidad ({round(score*100,1)}%) con historial en {item['category']}"
            }
            for item, score in ranked
        ]
    }

def output_fn(prediction, response_content_type):
    return json.dumps(prediction), "application/json"
#Entrenamiento, Métricas y Despliegue en SageMaker

import os, json, tarfile, shutil
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from inference_sagemaker import RecommenderNet

# 1. Catálogo simulado representativo del retail tecnológico (50.000 productos escalables)
CATEGORIES = ["Laptops", "Smartphones", "Audio", "Monitores", "Periféricos", "Componentes"]
CATALOG = [
    {"id": f"SKU-{1000+i}", "name": f"{CATEGORIES[i%6]} Pro Modelo {i+1}",
     "category": CATEGORIES[i%6], "price": round(50 + (i * 17.5) % 1400, 2)}
    for i in range(200)
]
NUM_USERS, NUM_ITEMS, EMB_DIM = 1000, len(CATALOG), 32

# 2. Generación de dataset sintético de interacciones (usuario, producto, compra/clic)
np.random.seed(42)
torch.manual_seed(42)
users = np.random.randint(0, NUM_USERS, 12000)
items = np.random.randint(0, NUM_ITEMS, 12000)
# Regla de afinidad: usuarios pares prefieren categorías pares
labels = ((users % 2 == items % 2) & (np.random.rand(12000) > 0.15)).astype(np.float32)

split = 9600
u_train, i_train, y_train = torch.tensor(users[:split]), torch.tensor(items[:split]), torch.tensor(labels[:split])
u_test, i_test, y_test = torch.tensor(users[split:]), torch.tensor(items[split:]), labels[split:]

# 3. Entrenamiento del modelo PyTorch
model = RecommenderNet(NUM_USERS, NUM_ITEMS, EMB_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.BCELoss()

for epoch in range(15):
    model.train()
    optimizer.zero_grad()
    preds = model(u_train, i_train)
    loss = criterion(preds, y_train)
    loss.backward()
    optimizer.step()

# 4. Cálculo de Métricas exigidas en EP3 (IE10)
model.eval()
with torch.no_grad():
    y_prob = model(u_test, i_test).numpy()
    y_pred = (y_prob >= 0.5).astype(int)

metrics = {
    "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
    "precision": round(float(precision_score(y_test, y_pred)), 4),
    "recall": round(float(recall_score(y_test, y_pred)), 4),
    "f1_score": round(float(f1_score(y_test, y_pred)), 4)
}
print("Métricas del Modelo PyTorch:", json.dumps(metrics, indent=2))

# 5. Empaquetado para AWS SageMaker (model.tar.gz)
os.makedirs("build/code", exist_ok=True)
torch.save(model.state_dict(), "build/model.pth")
with open("build/metadata.json", "w") as f:
    json.dump({"num_users": NUM_USERS, "num_items": NUM_ITEMS, "emb_dim": EMB_DIM, "catalog": CATALOG}, f)
shutil.copy("inference_sagemaker.py", "build/code/inference.py")

with tarfile.open("model.tar.gz", "w:gz") as tar:
    tar.add("build/model.pth", arcname="model.pth")
    tar.add("build/metadata.json", arcname="metadata.json")
    tar.add("build/code", arcname="code")

print("Artefacto model.tar.gz listo para subir a S3 y desplegar en SageMaker.")
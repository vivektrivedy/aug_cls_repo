CLS + Register + ROI — Vision-Transformer Multi-vector Retrieval
==============================================================

This repository contains a **research-grade re-implementation** of the paper Augmenting the CLS Token with Region of Interest Tokens for Efficient Multi-Vector Image Retrieval.

All vectors are **indexed in Weaviate** and every training / evaluation job is launched on **Modal GPU containers** (A10G by default).

---


## 1 · Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.9 – 3.11 | run the Modal CLI locally |
| **Modal CLI** | ≥ 0.56 | build, deploy, and invoke GPU functions |
| **Weaviate** | cloud cluster **or** Docker compose | vector store |
| **CUDA driver** | matches the A10G runtime | only needed when you run locally |

---

## 2 · Python dependencies and Setup Weaviate

`requirements.txt`

```text
torch>=2.2.0
torchvision>=0.17.0
timm>=0.9.16
einops>=0.7.0
faiss-gpu>=1.8.0
modal>=0.56
weaviate-client>=4.5
pillow


Log in to Weaviate Cloud (WCD) → Clusters → Create Cluster
Choose the free Sandbox tier, name & region, hit Create. 
Weaviate
Weaviate

Open Details → Authentication and copy:

https://<CLUSTER>.weaviate.network

API key (Write & Query)

## 3 · Register secrets on Modal

modal secret create weaviate-creds \
     WEAVIATE_URL=https://<CLUSTER>.weaviate.network \
     WEAVIATE_API_KEY=<YOUR-KEY> \
     WEAVIATE_CLASS=ClsRegRoiToken


## 4 · Dataset structure
my-data/
├── train/
│   ├── class_a/  img001.jpg ...
│   └── class_b/  ...
└── test/
    ├── class_c/  ...
    └── class_d/  ...

## 5 · Troubleshooting
| Symptom                         | Likely cause                                               | Fix                  |
| ------------------------------- | ---------------------------------------------------------- | -------------------- |
| `SecretNotFoundError` on deploy | secret typo                                                | `modal secret list`  |
| Weaviate 403                    | wrong API key scope                                        | regenerate key       |
| CUDA OOM on A10G                | batch too large                                            | lower `batch_size`   |
| Empty Recall values             | train/test overlap or wrong class name in `WEAVIATE_CLASS` | verify dataset & env |

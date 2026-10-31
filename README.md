
# Experiment setup

## Environment

- Jupyter kernel: The generator and judge run locally in four-bit form.

## Notebook order

1. `Legal_FAQ_Retrieval_Experiment.ipynb` — main retrieval
2. `Experiment_1_Retrieval_Supplement.ipynb` — retrieval checks
3. `Experiment_2_RAG_Generation_Evaluation.ipynb` — answer generation and evaluation
4. `Experiment_3_PolicyBench_Validation.ipynb` — second dataset validation

## Device

- Windows laptop; RTX 3060 Laptop GPU with **6 GiB VRAM**; about 16 GB RAM.

## Approximate time

| Run mode | Time |
|---|---:|
| Review saved outputs in all four notebooks | About 2–5 min |
| Rebuild retrieval and checks | Allow 30 min–2 h (estimate) |
| Regenerate and judge Experiment 2 | About 9 h |
| Regenerate and judge Experiment 3 | About 13 h; variable |

Retrieval time is an estimate. Experiment 2 and 3 times come from saved per-answer inference records. Downloads, manual review, and unexpected stalls add time.

## Package versions

```text
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.5.1+cu121
transformers==4.57.6
bitsandbytes==0.48.2
accelerate==1.15.0
huggingface-hub==0.36.2
hf_xet==1.7.0
safetensors==0.8.0
datasets==4.8.5
sentence-transformers==5.7.0
rank-bm25==0.2.2
numpy==2.4.4
pandas==3.0.3
matplotlib==3.10.9
nbformat==5.10.4
nbclient==0.10.4
ipykernel==7.2.0
jupyterlab==4.5.7
```

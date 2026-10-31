"""Local 4-bit inference and pinned model deployment for the FAQ study."""

import gc
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_XET_CACHE", r"D:\CodexModels\LegalFAQ\xet")
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "deployment.json"
MODELS = {"generator": "Qwen/Qwen2.5-3B-Instruct",
          "judge": "microsoft/Phi-3.5-mini-instruct"}


def deploy(cache_dir=r"D:\CodexModels\LegalFAQ\hub"):
    from huggingface_hub import HfApi, snapshot_download

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {"models": {}}
    config["cache_dir"] = str(cache)
    for role, repo in MODELS.items():
        revision = config["models"].get(role, {}).get("revision") or HfApi().model_info(repo).sha
        print(f"Downloading {role}: {repo}@{revision}", flush=True)
        snapshot = snapshot_download(repo, revision=revision, cache_dir=str(cache),
                                     allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.md", "LICENSE*"],
                                     max_workers=2)
        config["models"][role] = {"repo": repo, "revision": revision, "path": snapshot}
        CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


class LocalModel:
    def __init__(self, role):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not CONFIG_PATH.exists():
            raise FileNotFoundError("Run model_runtime.py --deploy first.")
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.info = config["models"][role]
        if not torch.cuda.is_available():
            raise RuntimeError("Select the Legal FAQ CUDA kernel; this configuration requires NVIDIA CUDA.")
        self.torch = torch
        self.role = role
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.info["path"], local_files_only=True,
                                                       trust_remote_code=False)
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_use_double_quant=True,
                                         bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.info["path"], local_files_only=True, trust_remote_code=False,
            quantization_config=quantization, dtype=torch.float16,
            device_map={"": 0}, attn_implementation="sdpa")
        self.model.eval()
        self.model.generation_config.temperature = 1.0
        self.model.generation_config.top_p = 1.0
        self.model.generation_config.top_k = 50
        torch.cuda.reset_peak_memory_stats()
        print(f"Loaded {role}: {self.info['repo']} in NF4 on {torch.cuda.get_device_name(0)}", flush=True)

    def input_tokens(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

    def generate(self, messages, max_new_tokens=160, max_input_tokens=2048):
        token_ids = self.input_tokens(messages)
        if len(token_ids) > max_input_tokens:
            raise ValueError(f"Prompt has {len(token_ids)} tokens; budget is {max_input_tokens}. No silent truncation.")
        inputs = self.torch.tensor([token_ids], device="cuda")
        started = time.perf_counter()
        self.torch.cuda.synchronize()
        with self.torch.inference_mode():
            output = self.model.generate(input_ids=inputs, attention_mask=self.torch.ones_like(inputs),
                                         do_sample=False, max_new_tokens=max_new_tokens,
                                         pad_token_id=self.tokenizer.eos_token_id, use_cache=True)
        self.torch.cuda.synchronize()
        new_tokens = output[0, inputs.shape[1]:].tolist()
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return {"answer": answer, "input_tokens": len(token_ids), "output_tokens": len(new_tokens),
                "seconds": time.perf_counter() - started,
                "hit_token_limit": len(new_tokens) == max_new_tokens,
                "peak_allocated_mb": self.torch.cuda.max_memory_allocated() / 2**20}

    def close(self):
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()


def deployment_report():
    import torch

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["hardware"] = {"name": torch.cuda.get_device_name(0),
                          "total_vram_mb": torch.cuda.get_device_properties(0).total_memory / 2**20}
    config["versions"] = {package: importlib.metadata.version(package) for package in
                          ["torch", "transformers", "bitsandbytes", "accelerate", "huggingface-hub"]}
    config["quantization"] = "NF4 double quantization, FP16 compute, SDPA, batch=1"
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--smoke", choices=list(MODELS))
    args = parser.parse_args()
    if args.deploy:
        deploy()
    if args.smoke:
        runtime = LocalModel(args.smoke)
        result = runtime.generate([{"role": "user", "content": "Reply with exactly: Local model is working."}], 30)
        print(json.dumps(result, indent=2))
        (ROOT / f"{args.smoke}_smoke_test.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        runtime.close()
        deployment_report()

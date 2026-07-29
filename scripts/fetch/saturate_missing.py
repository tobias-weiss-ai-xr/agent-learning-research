#!/usr/bin/env python3
"""Saturate 12 empty categories with targeted arXiv queries.

Targets: federated, curriculum, neurosymbolic, causal, rlhf-alignment,
diffusion, world-model, multimodal, reasoning, retrieval-augmented,
imitation, reward-modeling.

Each category gets 20+ targeted queries aiming for 200-400 papers.
24-month window (2024-07 to 2026-07), max 100 results per query.
Deduplicates against all existing papers in papers.yaml.
"""

import re
import sys
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import requests
import yaml

ARXIV_ID_PATTERN = re.compile(r"(\d{4}\.\d{4,5})")
ARXIV_SEARCH_API = (
    "http://export.arxiv.org/api/query?search_query={}&start={}&max_results={}"
)
API_DELAY = 3
MAX_RESULTS = 100
DATE_START = "202407010000"
DATE_END = "202607312359"

CATEGORY_QUERIES = {
    "federated": [
        'cat:cs.LG AND abs:"federated learning" AND abs:"deep"',
        'cat:cs.LG AND abs:"federated learning" AND abs:"privacy"',
        'cat:cs.LG AND abs:"federated learning" AND abs:"communication"',
        'cat:cs.LG AND abs:"federated learning" AND abs:"heterogeneous"',
        'cat:cs.LG AND abs:"federated learning" AND abs:"convergence"',
        'cat:cs.LG AND abs:"federated learning" AND abs:"personalization"',
        'cat:cs.LG AND abs:"federated learning" AND abs:"aggregation"',
        'cat:cs.LG AND abs:"federated learning" AND abs:"robust"',
        'cat:cs.LG AND abs:"federated learning" AND abs:"non-IID"',
        'cat:cs.CR AND abs:"federated learning" AND abs:"differential privacy"',
        'cat:cs.LG AND abs:"differential privacy" AND abs:"machine learning" AND abs:"deep"',
        'cat:cs.LG AND abs:"distributed learning" AND abs:"privacy" AND abs:"deep"',
        'cat:cs.DC AND abs:"distributed" AND abs:"privacy" AND abs:"learning"',
        'cat:cs.LG AND abs:"secure aggregation" AND abs:"federated"',
        'cat:cs.LG AND abs:"federated" AND abs:"client selection"',
        'cat:cs.LG AND abs:"federated" AND abs:"compression" AND abs:"communication"',
        'cat:cs.LG AND abs:"federated" AND abs:" Byzantine"',
        'cat:cs.LG AND abs:"federated" AND abs:"scaffold"',
        'cat:cs.LG AND abs:"federated" AND abs:"federated averaging"',
        'cat:cs.LG AND abs:"federated" AND abs:"edge computing"',
        'cat:cs.LG AND abs:"horizontal federated" AND abs:"vertical federated"',
    ],
    "curriculum": [
        'cat:cs.LG AND abs:"curriculum learning" AND abs:"deep"',
        'cat:cs.LG AND abs:"curriculum learning" AND abs:"training"',
        'cat:cs.LG AND abs:"self-paced learning" AND abs:"deep"',
        'cat:cs.LG AND abs:"self-paced learning" AND abs:"training"',
        'cat:cs.LG AND abs:"hard example mining" AND abs:"training" AND abs:"deep"',
        'cat:cs.LG AND abs:"easy to hard" AND abs:"training" AND abs:"deep"',
        'cat:cs.CV AND abs:"curriculum" AND abs:"learning" AND abs:"image"',
        'cat:cs.CL AND abs:"curriculum" AND abs:"learning" AND abs:"language"',
        'cat:cs.LG AND abs:"training curriculum" AND abs:"data" AND abs:"deep"',
        'cat:cs.LG AND abs:"data ordering" AND abs:"training" AND abs:"curriculum"',
        'cat:cs.LG AND abs:"curriculum" AND abs:"reinforcement learning"',
        'cat:cs.LG AND abs:"curriculum" AND abs:"sample selection"',
        'cat:cs.LG AND abs:"curriculum" AND abs:"difficulty" AND abs:"learning"',
        'cat:cs.LG AND abs:"hard negative mining" AND abs:"training"',
        'cat:cs.LG AND abs:"online curriculum" AND abs:"learning"',
        'cat:cs.LG AND abs:"self-paced" AND abs:"regularization"',
        'cat:cs.LG AND abs:"curriculum" AND abs:"teacher" AND abs:"student"',
        'cat:cs.LG AND abs:"curriculum" AND abs:"continual learning"',
        'cat:cs.CV AND abs:"hard example" AND abs:"object detection"',
        'cat:cs.LG AND abs:"curriculum" AND abs:"multi-task" AND abs:"learning"',
    ],
    "neurosymbolic": [
        'cat:cs.AI AND abs:"neurosymbolic" AND abs:"learning"',
        'cat:cs.AI AND abs:"neurosymbolic" AND abs:"reasoning"',
        'cat:cs.LG AND abs:"neurosymbolic" AND abs:"neural"',
        'cat:cs.LG AND abs:"neural-symbolic" AND abs:"learning"',
        'cat:cs.LG AND abs:"neural-symbolic" AND abs:"reasoning"',
        'cat:cs.LG AND abs:"logic" AND abs:"neural" AND abs:"differentiable"',
        'cat:cs.LG AND abs:"logic" AND abs:"deep learning" AND abs:"reasoning"',
        'cat:cs.LG AND abs:"neural theorem prover" AND abs:"learning"',
        'cat:cs.AI AND abs:"symbolic reasoning" AND abs:"neural network"',
        'cat:cs.AI AND abs:"symbolic" AND abs:"neural" AND abs:"integration"',
        'cat:cs.LG AND abs:"program synthesis" AND abs:"neural"',
        'cat:cs.CL AND abs:"neurosymbolic" AND abs:"language"',
        'cat:cs.LG AND abs:"probabilistic logic" AND abs:"neural"',
        'cat:cs.LG AND abs:"differentiable reasoning" AND abs:"learning"',
        'cat:cs.LG AND abs:"logical neural network"',
        'cat:cs.AI AND abs:"knowledge graph" AND abs:"neural" AND abs:"reasoning"',
        'cat:cs.LG AND abs:"neural" AND abs:"theorem proving"',
        'cat:cs.LG AND abs:"semantic parser" AND abs:"neural" AND abs:"symbolic"',
        'cat:cs.AI AND abs:"abductive reasoning" AND abs:"neural"',
        'cat:cs.LG AND abs:"neuro-symbolic" AND abs:"explainability"',
        'cat:cs.LG AND abs:"algebraic" AND abs:"neural" AND abs:"learning"',
    ],
    "causal": [
        'cat:cs.LG AND abs:"causal inference" AND abs:"machine learning"',
        'cat:cs.LG AND abs:"causal inference" AND abs:"deep"',
        'cat:cs.LG AND abs:"causal discovery" AND abs:"structure"',
        'cat:cs.LG AND abs:"causal discovery" AND abs:"graph"',
        'cat:cs.LG AND abs:"counterfactual" AND abs:"learning"',
        'cat:cs.LG AND abs:"counterfactual" AND abs:"deep learning"',
        'cat:cs.LG AND abs:"structural causal model" AND abs:"neural"',
        'cat:stat.ML AND abs:"causal inference" AND abs:"machine learning"',
        'cat:cs.AI AND abs:"causal" AND abs:"reasoning"',
        'cat:cs.LG AND abs:"intervention" AND abs:"causal" AND abs:"learning"',
        'cat:cs.LG AND abs:"treatment effect" AND abs:"estimation" AND abs:"neural"',
        'cat:cs.LG AND abs:"do-calculus" AND abs:"causal"',
        'cat:cs.LG AND abs:"causal representation" AND abs:"learning"',
        'cat:cs.LG AND abs:"causal" AND abs:"reinforcement learning"',
        'cat:cs.LG AND abs:"neural" AND abs:"causal" AND abs:"inference"',
        'cat:cs.LG AND abs:"causal" AND abs:"transformer"',
        'cat:cs.LG AND abs:"causal" AND abs:"graph neural network"',
        'cat:cs.LG AND abs:"causal" AND abs:"variational"',
        'cat:cs.LG AND abs:"instrumental variable" AND abs:"neural"',
        'cat:cs.LG AND abs:"causal" AND abs:"fairness" AND abs:"learning"',
        'cat:cs.LG AND abs:"causal" AND abs:"time series"',
    ],
    "rlhf-alignment": [
        'cat:cs.CL AND abs:"RLHF" AND abs:"language model"',
        'cat:cs.CL AND abs:"RLHF" AND abs:"alignment"',
        'cat:cs.LG AND abs:"reinforcement learning from human feedback"',
        'cat:cs.CL AND abs:"preference optimization" AND abs:"language"',
        'cat:cs.CL AND abs:"DPO" AND abs:"direct preference optimization"',
        'cat:cs.CL AND abs:"constitutional AI" AND abs:"alignment"',
        'cat:cs.CL AND abs:"reward model" AND abs:"language" AND abs:"training"',
        'cat:cs.CL AND abs:"alignment" AND abs:"instruction" AND abs:"tuning"',
        'cat:cs.CL AND abs:"KTO" AND abs:"preference" AND abs:"learning"',
        'cat:cs.CL AND abs:"value learning" AND abs:"language model"',
        'cat:cs.CL AND abs:"preference" AND abs:"RLHF" AND abs:"large language"',
        'cat:cs.CL AND abs:"alignment tax" AND abs:"RLHF"',
        'cat:cs.CL AND abs:"online" AND abs:"DPO" AND abs:"preference"',
        'cat:cs.CL AND abs:"reward hacking" AND abs:"RLHF"',
        'cat:cs.LG AND abs:"human preference" AND abs:"reinforcement learning"',
        'cat:cs.CL AND abs:"self-rewarding" AND abs:"language model"',
        'cat:cs.CL AND abs:"RLAIF" AND abs:"alignment"',
        'cat:cs.CL AND abs:"safety" AND abs:"alignment" AND abs:"RLHF"',
        'cat:cs.CL AND abs:"preference" AND abs:"model" AND abs:"alignment"',
        'cat:cs.CL AND abs:"helpful" AND abs:"harmless" AND abs:"honest" AND abs:"RLHF"',
        'cat:cs.CL AND abs:"contrastive" AND abs:"preference" AND abs:"learning"',
    ],
    "diffusion": [
        'cat:cs.LG AND abs:"diffusion model" AND abs:"training"',
        'cat:cs.LG AND abs:"diffusion model" AND abs:"image generation"',
        'cat:cs.LG AND abs:"diffusion model" AND abs:"sampling"',
        'cat:cs.LG AND abs:"flow matching" AND abs:"generative"',
        'cat:cs.LG AND abs:"score matching" AND abs:"generative"',
        'cat:cs.LG AND abs:"score-based" AND abs:"generative model"',
        'cat:cs.LG AND abs:"rectified flow" AND abs:"generation"',
        'cat:cs.CV AND abs:"diffusion" AND abs:"image" AND abs:"generation"',
        'cat:cs.CV AND abs:"diffusion" AND abs:"video" AND abs:"generation"',
        'cat:cs.LG AND abs:"diffusion" AND abs:"text-to-image"',
        'cat:cs.LG AND abs:"diffusion" AND abs:"conditional generation"',
        'cat:cs.LG AND abs:"denoising" AND abs:"diffusion" AND abs:"training"',
        'cat:cs.LG AND abs:"stochastic differential equation" AND abs:"generative"',
        'cat:cs.LG AND abs:"diffusion" AND abs:"guidance"',
        'cat:cs.LG AND abs:"latent diffusion" AND abs:"training"',
        'cat:cs.LG AND abs:"consistency model" AND abs:"distillation"',
        'cat:cs.LG AND abs:"flow" AND abs:"matching" AND abs:"training"',
        'cat:cs.LG AND abs:"noise prediction" AND abs:"diffusion"',
        'cat:cs.SD AND abs:"diffusion" AND abs:"audio" AND abs:"generation"',
        'cat:cs.CL AND abs:"diffusion" AND abs:"language" AND abs:"generation"',
        'cat:cs.LG AND abs:"diffusion" AND abs:"3D" AND abs:"generation"',
    ],
    "world-model": [
        'cat:cs.LG AND abs:"world model" AND abs:"learning"',
        'cat:cs.LG AND abs:"world model" AND abs:"reinforcement learning"',
        'cat:cs.LG AND abs:"world model" AND abs:"planning"',
        'cat:cs.LG AND abs:"latent dynamics" AND abs:"model"',
        'cat:cs.LG AND abs:"latent dynamics" AND abs:"prediction"',
        'cat:cs.LG AND abs:"environment model" AND abs:"reinforcement"',
        'cat:cs.LG AND abs:"predictive model" AND abs:"state" AND abs:"learning"',
        'cat:cs.LG AND abs:"world model" AND abs:"robotics"',
        'cat:cs.LG AND abs:"model-based" AND abs:"reinforcement" AND abs:"world"',
        'cat:cs.LG AND abs:"latent space" AND abs:"dynamics" AND abs:"prediction"',
        'cat:cs.LG AND abs:"dreamer" AND abs:"world model"',
        'cat:cs.LG AND abs:"simulated" AND abs:"environment" AND abs:"learning"',
        'cat:cs.RO AND abs:"world model" AND abs:"robot"',
        'cat:cs.LG AND abs:"state-space model" AND abs:"sequential" AND abs:"prediction"',
        'cat:cs.LG AND abs:"transition model" AND abs:"reinforcement" AND abs:"learning"',
        'cat:cs.LG AND abs:"imagination" AND abs:"reinforcement" AND abs:"planning"',
        'cat:cs.LG AND abs:"video prediction" AND abs:"world"',
        'cat:cs.LG AND abs:"generative" AND abs:"environment" AND abs:"model"',
        'cat:cs.LG AND abs:"recurrent" AND abs:"world model"',
        'cat:cs.LG AND abs:"world model" AND abs:"autonomous"',
    ],
    "multimodal": [
        'cat:cs.CV AND abs:"vision language model" AND abs:"training"',
        'cat:cs.CV AND abs:"vision language model" AND abs:"VLM"',
        'cat:cs.CL AND abs:"multimodal" AND abs:"language" AND abs:"vision"',
        'cat:cs.CL AND abs:"CLIP" AND abs:"training" AND abs:"contrastive"',
        'cat:cs.LG AND abs:"multimodal learning" AND abs:"representation"',
        'cat:cs.CV AND abs:"cross-modal" AND abs:"learning" AND abs:"vision"',
        'cat:cs.CL AND abs:"multimodal" AND abs:"reasoning"',
        'cat:cs.CV AND abs:"multimodal" AND abs:"image" AND abs:"text"',
        'cat:cs.MM AND abs:"multimodal" AND abs:"learning" AND abs:"deep"',
        'cat:cs.LG AND abs:"multimodal" AND abs:"fusion" AND abs:"representation"',
        'cat:cs.CL AND abs:"multimodal" AND abs:"instruction" AND abs:"tuning"',
        'cat:cs.CV AND abs:"VLM" AND abs:"benchmark" AND abs:"evaluation"',
        'cat:cs.LG AND abs:"vision encoder" AND abs:"language decoder"',
        'cat:cs.CV AND abs:"multimodal" AND abs:"alignment" AND abs:"vision"',
        'cat:cs.CL AND abs:"multimodal" AND abs:"large language model"',
        'cat:cs.SD AND abs:"audio-visual" AND abs:"learning"',
        'cat:cs.CV AND abs:"multimodal" AND abs:"segmentation"',
        'cat:cs.LG AND abs:"multimodal" AND abs:"retrieval"',
        'cat:cs.CL AND abs:"multimodal" AND abs:"chain-of-thought"',
        'cat:cs.CV AND abs:"multimodal" AND abs:"grounding"',
        'cat:cs.LG AND abs:"multimodal" AND abs:"embodied"',
    ],
    "reasoning": [
        'cat:cs.CL AND abs:"test-time compute" AND abs:"language"',
        'cat:cs.CL AND abs:"chain-of-thought" AND abs:"training"',
        'cat:cs.CL AND abs:"chain-of-thought" AND abs:"reasoning"',
        'cat:cs.CL AND abs:"reasoning model" AND abs:"language"',
        'cat:cs.CL AND abs:"inference-time scaling" AND abs:"language"',
        'cat:cs.CL AND abs:"system-2" AND abs:"thinking" AND abs:"language"',
        'cat:cs.CL AND abs:"process reward" AND abs:"reasoning"',
        'cat:cs.CL AND abs:"outcome reward" AND abs:"reasoning"',
        'cat:cs.CL AND abs:"search" AND abs:"reasoning" AND abs:"language"',
        'cat:cs.CL AND abs:"reasoning" AND abs:"verification" AND abs:"language"',
        'cat:cs.CL AND abs:"o1" AND abs:"reasoning" AND abs:"language model"',
        'cat:cs.CL AND abs:"distillation" AND abs:"reasoning"',
        'cat:cs.CL AND abs:"math reasoning" AND abs:"language model"',
        'cat:cs.AI AND abs:"reasoning" AND abs:"neural" AND abs:"logic"',
        'cat:cs.CL AND abs:"scratchpad" AND abs:"reasoning"',
        'cat:cs.CL AND abs:"tree-of-thought" AND abs:"reasoning"',
        'cat:cs.CL AND abs:"self-consistency" AND abs:"reasoning"',
        'cat:cs.CL AND abs:"reinforcement" AND abs:"reasoning" AND abs:"language"',
        'cat:cs.CL AND abs:"abstract reasoning" AND abs:"neural"',
        'cat:cs.CL AND abs:"code reasoning" AND abs:"language model"',
        'cat:cs.CL AND abs:"long reasoning" AND abs:"chain"',
    ],
    "retrieval-augmented": [
        'cat:cs.CL AND abs:"retrieval-augmented generation" AND abs:"RAG"',
        'cat:cs.CL AND abs:"RAG" AND abs:"retrieval" AND abs:"generation"',
        'cat:cs.IR AND abs:"retrieval augmented" AND abs:"language"',
        'cat:cs.CL AND abs:"retrieval-augmented" AND abs:"language model"',
        'cat:cs.CL AND abs:"memory-augmented" AND abs:"language" AND abs:"learning"',
        'cat:cs.CL AND abs:"retrieval" AND abs:"augmented" AND abs:"training"',
        'cat:cs.CL AND abs:"RAG" AND abs:"knowledge" AND abs:"generation"',
        'cat:cs.IR AND abs:"retriever" AND abs:"generator" AND abs:"training"',
        'cat:cs.CL AND abs:"self-RAG" AND abs:"retrieval"',
        'cat:cs.CL AND abs:"active retrieval" AND abs:"generation"',
        'cat:cs.CL AND abs:"retrieval" AND abs:"fine-tuning" AND abs:"RAG"',
        'cat:cs.CL AND abs:"knowledge-intensive" AND abs:"retrieval" AND abs:"NLP"',
        'cat:cs.IR AND abs:"dense retrieval" AND abs:"training" AND abs:"learning"',
        'cat:cs.CL AND abs:"retrieval" AND abs:"augmented" AND abs:"instruction"',
        'cat:cs.CL AND abs:"RAG" AND abs:"chunking" AND abs:"retrieval"',
        'cat:cs.LG AND abs:"retrieval-augmented" AND abs:"learning"',
        'cat:cs.CL AND abs:"adaptive retrieval" AND abs:"language model"',
        'cat:cs.CL AND abs:"multi-hop" AND abs:"retrieval" AND abs:"generation"',
        'cat:cs.CL AND abs:"RAG" AND abs:"evaluation" AND abs:"benchmark"',
        'cat:cs.CL AND abs:"retrieval" AND abs:"query" AND abs:"reformulation"',
        'cat:cs.CL AND abs:"real-time" AND abs:"retrieval" AND abs:"generation"',
    ],
    "imitation": [
        'cat:cs.LG AND abs:"imitation learning" AND abs:"robotics"',
        'cat:cs.LG AND abs:"behavior cloning" AND abs:"deep"',
        'cat:cs.LG AND abs:"behavior cloning" AND abs:"policy"',
        'cat:cs.RO AND abs:"imitation learning" AND abs:"robot"',
        'cat:cs.LG AND abs:"inverse reinforcement learning"',
        'cat:cs.LG AND abs:"learning from demonstration" AND abs:"robot"',
        'cat:cs.LG AND abs:"learning from demonstration" AND abs:"deep"',
        'cat:cs.LG AND abs:"inverse reinforcement learning" AND abs:"deep"',
        'cat:cs.LG AND abs:"imitation" AND abs:"expert" AND abs:"policy"',
        'cat:cs.LG AND abs:"generative adversarial imitation learning"',
        'cat:cs.LG AND abs:"DAgger" AND abs:"imitation"',
        'cat:cs.RO AND abs:"behavior cloning" AND abs:"autonomous"',
        'cat:cs.LG AND abs:"imitation" AND abs:"transformer" AND abs:"policy"',
        'cat:cs.LG AND abs:"diffusion policy" AND abs:"imitation"',
        'cat:cs.LG AND abs:"imitation" AND abs:"manipulation"',
        'cat:cs.LG AND abs:"demonstration" AND abs:"policy" AND abs:"learning"',
        'cat:cs.LG AND abs:"reward inference" AND abs:"demonstration"',
        'cat:cs.LG AND abs:"action chunking" AND abs:"imitation"',
        'cat:cs.SY AND abs:"imitation learning" AND abs:"control"',
        'cat:cs.LG AND abs:"offline imitation" AND abs:"behavior"',
        'cat:cs.LG AND abs:"multi-modal" AND abs:"imitation" AND abs:"policy"',
    ],
    "reward-modeling": [
        'cat:cs.LG AND abs:"reward model" AND abs:"training"',
        'cat:cs.LG AND abs:"reward model" AND abs:"reinforcement learning"',
        'cat:cs.LG AND abs:"reward prediction" AND abs:"learning"',
        'cat:cs.LG AND abs:"preference learning" AND abs:"reward"',
        'cat:cs.LG AND abs:"preference learning" AND abs:"feedback"',
        'cat:cs.LG AND abs:"feedback learning" AND abs:"reward"',
        'cat:cs.LG AND abs:"reward model" AND abs:"human"',
        'cat:cs.LG AND abs:"reward function" AND abs:"learning" AND abs:"inverse"',
        'cat:cs.CL AND abs:"reward model" AND abs:"language"',
        'cat:cs.CL AND abs:"reward model" AND abs:"alignment"',
        'cat:cs.LG AND abs:"preference" AND abs:"comparison" AND abs:"learning"',
        'cat:cs.LG AND abs:"Bradley-Terry" AND abs:"reward" AND abs:"model"',
        'cat:cs.LG AND abs:"multi-objective reward" AND abs:"model"',
        'cat:cs.LG AND abs:"reward" AND abs:"shaping" AND abs:"learning"',
        'cat:cs.RO AND abs:"reward model" AND abs:"robotics"',
        'cat:cs.LG AND abs:"implicit reward" AND abs:"model"',
        'cat:cs.LG AND abs:"reward modeling" AND abs:"sparse"',
        'cat:cs.LG AND abs:"contrastive" AND abs:"reward" AND abs:"modeling"',
        'cat:cs.LG AND abs:"reward" AND abs:"extrapolation" AND abs:"generalization"',
        'cat:cs.LG AND abs:"ensemble" AND abs:"reward" AND abs:"model"',
        'cat:cs.LG AND abs:"active learning" AND abs:"reward" AND abs:"feedback"',
    ],
}

SUB_KW_MAP = {
    "theory": [
        "generalization bound",
        "convergence",
        "information theory",
        "pac",
        "vc dimension",
        "rademacher",
        "sample complexity",
        "theoretical",
        "proof",
        "theorem",
        "complexity",
        "optimal",
        "lower bound",
        "upper bound",
        "rate",
        "minimax",
    ],
    "algorithm": [
        "algorithm",
        "method",
        "approach",
        "framework",
        "proposed",
        "technique",
        "procedure",
        "strategy",
        "model",
        "our",
    ],
    "architecture": [
        "transformer",
        "cnn",
        "graph neural",
        "attention mechanism",
        "architecture",
        "network design",
        "residual",
        "encoder",
        "decoder",
        "gan",
        "diffusion",
        "encoder-decoder",
        "bert",
        "gpt",
        "mamba",
        "unet",
        "vit",
        "mlp",
        "state-space",
        "module",
        "block",
        "layer",
    ],
    "optimization": [
        "optimizer",
        "adam",
        "sgd",
        "learning rate",
        "regularization",
        "loss function",
        "gradient descent",
        "scheduler",
        "training objective",
        "gradient",
        "backpropagation",
        "weight decay",
        "momentum",
        "clip",
        "normalization",
    ],
    "scaling": [
        "scaling law",
        "compute-optimal",
        "large-scale",
        "scale",
        "chinchilla",
        "billion parameter",
        "trillion",
        "compute",
        "flops",
        "model size",
        "data scaling",
    ],
    "efficient": [
        "compression",
        "quantization",
        "distillation",
        "pruning",
        "efficient",
        "speedup",
        "low-rank",
        "lora",
        "adapter",
        "sparsity",
        "sparse",
        "accelerat",
        "latency",
        "memory efficient",
        "parameter efficient",
    ],
    "robust": [
        "adversarial",
        "out-of-distribution",
        "robustness",
        "robust",
        "fairness",
        "safety",
        "certified",
        "perturbation",
        "privacy",
        "secure",
        "distribution shift",
        "noise",
        "corruption",
        "adversarial example",
    ],
    "application": [
        "nlp",
        "natural language",
        "computer vision",
        "robotics",
        "healthcare",
        "science",
        "medical",
        "autonomous driving",
        "protein",
        "clinical",
        "drug",
        "recommendation",
        "speech",
        "code generation",
        "autonomous",
        "self-driving",
        "agriculture",
        "climate",
        "finance",
        "education",
    ],
}

ALL_CATEGORIES = [
    "supervised",
    "unsupervised",
    "reinforcement",
    "self-supervised",
    "meta-learning",
    "continual",
    "transfer",
    "multi-agent",
    "active",
    "online",
    "federated",
    "curriculum",
    "neurosymbolic",
    "causal",
    "rlhf-alignment",
    "diffusion",
    "world-model",
    "multimodal",
    "reasoning",
    "retrieval-augmented",
    "imitation",
    "reward-modeling",
]

ALL_SUBCATEGORIES = [
    "theory",
    "algorithm",
    "architecture",
    "optimization",
    "scaling",
    "efficient",
    "robust",
    "application",
]


def load_existing_papers(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    papers = data.get("papers", [])
    by_id = {}
    titles_lower = []
    for p in papers:
        url = p.get("url", "")
        match = ARXIV_ID_PATTERN.search(url)
        if match:
            by_id[match.group(1)] = p
        titles_lower.append(p.get("title", "").lower().strip())
    return data, papers, by_id, titles_lower


def title_similarity(a, b):
    a_clean = re.sub(r"[^\w\s]", "", a.lower())
    b_clean = re.sub(r"[^\w\s]", "", b.lower())
    return SequenceMatcher(None, a_clean, b_clean).ratio()


def search_arxiv(query, start=0, max_results=100):
    full_query = f"({query}) AND submittedDate:[{DATE_START} TO {DATE_END}]"
    for attempt in range(4):
        try:
            resp = requests.get(
                ARXIV_SEARCH_API.format(
                    requests.utils.quote(full_query), start, max_results
                ),
                timeout=30,
            )
            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                print(
                    f"    429, waiting {wait}s (attempt {attempt + 1}/4)...", flush=True
                )
                time.sleep(wait)
                continue
            if resp.status_code == 503:
                print(f"    503, waiting 60s...", flush=True)
                time.sleep(60)
                continue
            resp.raise_for_status()
            entries = []
            for match in re.finditer(r"<entry>(.*?)</entry>", resp.text, re.DOTALL):
                entry_xml = match.group(1)
                entry = {}
                title_m = re.search(r"<title>(.*?)</title>", entry_xml, re.DOTALL)
                if title_m:
                    entry["title"] = re.sub(r"\s+", " ", title_m.group(1).strip())
                id_m = re.search(r"<id>(.*?)</id>", entry_xml)
                if id_m:
                    entry["url"] = id_m.group(1).strip().replace("http://", "https://")
                published_m = re.search(r"<published>(.*?)</published>", entry_xml)
                if published_m:
                    entry["date"] = published_m.group(1).strip()[:7]
                summary_m = re.search(r"<summary>(.*?)</summary>", entry_xml, re.DOTALL)
                if summary_m:
                    entry["abstract"] = re.sub(r"\s+", " ", summary_m.group(1).strip())
                if entry.get("title") and entry.get("url"):
                    entries.append(entry)
            return entries
        except requests.exceptions.Timeout:
            print(f"    Timeout, waiting 30s (attempt {attempt + 1}/4)...", flush=True)
            time.sleep(30)
        except Exception as e:
            print(
                f"    WARNING: arXiv search error: {e} (attempt {attempt + 1}/4)",
                flush=True,
            )
            time.sleep(15)
    return []


def classify_subcategory(title, abstract, category_hint):
    text = f"{title} {abstract}".lower()
    sub_scores = {}
    for sub_name, kw_list in SUB_KW_MAP.items():
        sub_scores[sub_name] = sum(1 for kw in kw_list if kw in text)
    best = max(sub_scores.values())
    if best > 0:
        return max(sub_scores, key=sub_scores.get)
    return "algorithm"


def dedup_title(title, titles_lower, threshold=0.75):
    title_clean = title.lower().strip()
    for existing in titles_lower:
        if title_similarity(title_clean, existing) >= threshold:
            return True
    return False


def save_papers(yaml_path, data, papers):
    data["papers"] = papers
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(
            data, f, default_flow_style=False, allow_unicode=True, sort_keys=False
        )


def main():
    yaml_path = Path(__file__).resolve().parent.parent.parent / "papers.yaml"
    data, papers, by_id, titles_lower = load_existing_papers(yaml_path)
    print(f"Loaded {len(papers)} existing papers", flush=True)

    cat_counter = Counter(p.get("category", "?") for p in papers)
    print("\nCurrent category distribution:", flush=True)
    for cat in ALL_CATEGORIES:
        count = cat_counter.get(cat, 0)
        marker = " [EMPTY]" if count == 0 else ""
        print(f"  {cat}: {count}{marker}", flush=True)

    MIN_THRESHOLD = 150
    to_run = {
        cat: qs
        for cat, qs in CATEGORY_QUERIES.items()
        if cat_counter.get(cat, 0) < MIN_THRESHOLD
    }
    skipped = [
        cat for cat in CATEGORY_QUERIES if cat_counter.get(cat, 0) >= MIN_THRESHOLD
    ]
    if skipped:
        print(
            f"\nSkipping saturated categories (>= {MIN_THRESHOLD}): {skipped}",
            flush=True,
        )

    total_queries = sum(len(qs) for qs in to_run.values())
    print(
        f"\nRunning {total_queries} targeted queries across {len(to_run)} remaining categories",
        flush=True,
    )

    total_new = 0
    cat_new = defaultdict(int)
    seen_ids = set(by_id.keys())
    seen_titles = set(titles_lower)

    for cat, queries in to_run.items():
        print(f"\n{'=' * 60}", flush=True)
        print(
            f"CATEGORY: {cat} ({len(queries)} queries, existing: {cat_counter.get(cat, 0)})",
            flush=True,
        )
        print(f"{'=' * 60}", flush=True)

        cat_batch_new = 0
        for qi, query in enumerate(queries):
            print(f"\n  Query {qi + 1}/{len(queries)}...", flush=True)
            entries = search_arxiv(query)
            print(f"    arXiv returned {len(entries)} entries", flush=True)

            batch_new = 0
            for entry in entries:
                arxiv_id_match = ARXIV_ID_PATTERN.search(entry.get("url", ""))
                arxiv_id = arxiv_id_match.group(1) if arxiv_id_match else None
                if arxiv_id and arxiv_id in seen_ids:
                    continue
                title = entry.get("title", "")
                title_lower = title.lower().strip()
                if title_lower in seen_titles:
                    continue
                if dedup_title(title, list(seen_titles)):
                    continue
                abstract = entry.get("abstract", "")
                subcategory = classify_subcategory(title, abstract, cat)

                new_paper = {
                    "title": title,
                    "date": entry.get("date", ""),
                    "url": entry.get("url", ""),
                    "category": cat,
                    "subcategory": subcategory,
                    "authors": [],
                    "venue": "",
                    "code_url": "",
                    "project_url": "",
                    "abstract": abstract,
                    "tags": [f"auto-{cat}", f"auto-{subcategory}"],
                }
                if arxiv_id:
                    seen_ids.add(arxiv_id)
                seen_titles.add(title_lower)
                titles_lower.append(title_lower)
                papers.append(new_paper)
                batch_new += 1
                total_new += 1
                cat_batch_new += 1
                cat_new[cat] += 1
                if arxiv_id:
                    by_id[arxiv_id] = new_paper
                print(f"    NEW [{cat}/{subcategory}] {title[:70]}", flush=True)

            print(f"    batch added: {batch_new}", flush=True)
            time.sleep(API_DELAY)

        print(f"\n  >> {cat}: {cat_batch_new} new papers this category", flush=True)

        save_papers(yaml_path, data, papers)
        print(f"  >> checkpoint saved ({len(papers)} total)", flush=True)

    save_papers(yaml_path, data, papers)
    print(f"\n{'=' * 60}", flush=True)
    print(f"FINAL: Saved {len(papers)} total papers ({total_new} new)", flush=True)
    print(f"{'=' * 60}", flush=True)

    final_counter = Counter()
    final_sub = Counter()
    for p in papers:
        final_counter[p.get("category", "unknown")] += 1
        final_sub[p.get("subcategory", "unknown")] += 1

    print(f"\nBy category:", flush=True)
    for cat in ALL_CATEGORIES:
        old = cat_counter.get(cat, 0)
        new = cat_new.get(cat, 0)
        total = final_counter.get(cat, 0)
        print(f"  {cat}: {total} (was {old}, +{new})", flush=True)
    print(f"\nBy subcategory:", flush=True)
    for sub in ALL_SUBCATEGORIES:
        print(f"  {sub}: {final_sub.get(sub, 0)}", flush=True)

    print(f"\nTotal papers: {len(papers)}", flush=True)
    print(f"New papers added: {total_new}", flush=True)

    empty_remaining = [c for c in ALL_CATEGORIES if final_counter.get(c, 0) == 0]
    if empty_remaining:
        print(f"\nWARNING: Still empty: {empty_remaining}", flush=True)
    else:
        print(f"\nAll {len(ALL_CATEGORIES)} categories populated!", flush=True)


if __name__ == "__main__":
    main()

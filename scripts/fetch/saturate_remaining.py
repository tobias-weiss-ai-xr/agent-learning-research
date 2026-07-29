#!/usr/bin/env python3
"""Run remaining queries (61-132) for categories not yet populated."""

import re
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import requests
import yaml

ARXIV_ID_PATTERN = re.compile(r"(\d{4}\.\d{4,5})")
ARXIV_SEARCH_API = (
    "http://export.arxiv.org/api/query?search_query={}&start={}&max_results={}"
)
API_DELAY = 3
MAX_RESULTS_PER_QUERY = 100
DATE_START = "202407010000"
DATE_END = "202607312359"

QUERIES = [
    # --- multi-agent (remaining 4 queries, 61-64) ---
    'cat:cs.LG AND abs:"multi-agent reinforcement learning" AND abs:"cooperative"',
    'cat:cs.AI AND abs:"multi-agent" AND abs:"emergent"',
    'cat:cs.LG AND abs:"social learning" AND abs:"multi-agent"',
    'cat:cs.LG AND abs:"multi-agent" AND abs:"decentralized"',
    # --- active (8 queries, 65-72) ---
    'cat:cs.LG AND abs:"active learning" AND abs:"query strategy"',
    'cat:cs.LG AND abs:"human-in-the-loop" AND abs:"machine learning"',
    'cat:cs.LG AND abs:"active learning" AND abs:"deep"',
    'cat:cs.LG AND abs:"annotation" AND abs:"active"',
    'cat:cs.CV AND abs:"active learning" AND abs:"image"',
    'cat:cs.LG AND abs:"uncertainty sampling" AND abs:"active learning"',
    'cat:cs.CL AND abs:"active learning" AND abs:"NLP"',
    'cat:cs.LG AND abs:"semi-supervised" AND abs:"active"',
    # --- online (8 queries, 73-80) ---
    'cat:cs.LG AND abs:"online learning" AND abs:"streaming"',
    'cat:cs.LG AND abs:"bandit" AND abs:"non-stationary"',
    'cat:cs.LG AND abs:"multi-armed bandit" AND abs:"contextual"',
    'cat:cs.LG AND abs:"online learning" AND abs:"regret"',
    'cat:cs.LG AND abs:"concept drift" AND abs:"online"',
    'cat:stat.ML AND abs:"online learning" AND abs:"bandit"',
    'cat:cs.LG AND abs:"continual" AND abs:"online" AND abs:"streaming"',
    'cat:cs.LG AND abs:"stochastic optimization" AND abs:"online"',
    # --- federated (8 queries, 81-88) ---
    'cat:cs.LG AND abs:"federated learning"',
    'cat:cs.LG AND abs:"differential privacy" AND abs:"machine learning"',
    'cat:cs.LG AND abs:"distributed learning" AND abs:"privacy"',
    'cat:cs.LG AND abs:"federated" AND abs:"communication" AND abs:"efficient"',
    'cat:cs.CR AND abs:"federated" AND abs:"privacy"',
    'cat:cs.LG AND abs:"secure aggregation" AND abs:"federated"',
    'cat:cs.LG AND abs:"heterogeneous" AND abs:"federated learning"',
    'cat:cs.DC AND abs:"distributed" AND abs:"privacy" AND abs:"learning"',
    # --- curriculum (8 queries, 89-96) ---
    'cat:cs.LG AND abs:"curriculum learning"',
    'cat:cs.LG AND abs:"self-paced learning"',
    'cat:cs.LG AND abs:"hard example mining" AND abs:"training"',
    'cat:cs.LG AND abs:"easy to hard" AND abs:"training"',
    'cat:cs.CV AND abs:"curriculum" AND abs:"learning"',
    'cat:cs.CL AND abs:"curriculum" AND abs:"learning"',
    'cat:cs.LG AND abs:"training curriculum" AND abs:"data"',
    'cat:cs.LG AND abs:"course learning" AND abs:"sample"',
    # --- neurosymbolic (8 queries, 97-104) ---
    'cat:cs.LG AND abs:"neurosymbolic" AND abs:"learning"',
    'cat:cs.AI AND abs:"neurosymbolic" AND abs:"reasoning"',
    'cat:cs.LG AND abs:"neural theorem prover"',
    'cat:cs.LG AND abs:"logic" AND abs:"neural" AND abs:"differentiable"',
    'cat:cs.AI AND abs:"symbolic reasoning" AND abs:"neural"',
    'cat:cs.LG AND abs:"neural-symbolic" AND abs:"learning"',
    'cat:cs.CL AND abs:"neurosymbolic" AND abs:"language"',
    'cat:cs.LG AND abs:"program synthesis" AND abs:"neural"',
    # --- causal (8 queries, 105-112) ---
    'cat:cs.LG AND abs:"causal inference" AND abs:"machine learning"',
    'cat:cs.LG AND abs:"causal discovery" AND abs:"structure"',
    'cat:cs.LG AND abs:"counterfactual" AND abs:"learning"',
    'cat:cs.LG AND abs:"structural causal model" AND abs:"neural"',
    'cat:stat.ML AND abs:"causal inference" AND abs:"machine learning"',
    'cat:cs.AI AND abs:"causal" AND abs:"reasoning"',
    'cat:cs.LG AND abs:"intervention" AND abs:"causal" AND abs:"learning"',
    'cat:cs.LG AND abs:"treatment effect" AND abs:"estimation" AND abs:"neural"',
    # --- cross-cutting supplementary (113-132) ---
    'cat:cs.LG AND abs:"deep learning" AND abs:"theory" AND abs:"generalization"',
    'cat:cs.LG AND abs:"graph neural network" AND abs:"learning"',
    'cat:cs.LG AND abs:"transformer" AND abs:"attention" AND abs:"architecture"',
    'cat:cs.LG AND abs:"optimizer" AND abs:"learning rate" AND abs:"neural"',
    'cat:cs.LG AND abs:"regularization" AND abs:"deep learning"',
    'cat:cs.LG AND abs:"adversarial robustness" AND abs:"deep"',
    'cat:cs.LG AND abs:"model compression" AND abs:"quantization"',
    'cat:cs.LG AND abs:"knowledge distillation" AND abs:"neural"',
    'cat:cs.LG AND abs:"scaling law" AND abs:"language model"',
    'cat:cs.LG AND abs:"efficient inference" AND abs:"neural"',
    'cat:cs.LG AND abs:"out-of-distribution" AND abs:"generalization"',
    'cat:cs.IT AND abs:"information theory" AND abs:"deep learning"',
    'cat:cs.LG AND abs:"deep learning" AND abs:"healthcare"',
    'cat:cs.LG AND abs:"reinforcement learning" AND abs:"NLP"',
    'cat:cs.LG AND abs:"RLHF" AND abs:"alignment" AND abs:"language"',
    'cat:cs.LG AND abs:"generative adversarial" AND abs:"training"',
    'cat:cs.LG AND abs:"diffusion model" AND abs:"training"',
    'cat:cs.LG AND abs:"spiking neural" AND abs:"learning"',
    'cat:cs.SY AND abs:"neural" AND abs:"learning" AND abs:"control"',
    'cat:cs.LG AND abs:"loss function" AND abs:"deep learning"',
]

CATEGORY_HINTS = {
    "multi-agent": range(0, 4),
    "active": range(4, 12),
    "online": range(12, 20),
    "federated": range(20, 28),
    "curriculum": range(28, 36),
    "neurosymbolic": range(36, 44),
    "causal": range(44, 52),
}


def get_hint(idx):
    for cat, rng in CATEGORY_HINTS.items():
        if idx in rng:
            return cat
    return "auto"


SUPERVISED_KW = [
    "supervised",
    "classification",
    "regression",
    "labeled data",
    "label",
    "prediction task",
    "object detection",
    "segmentation",
]
UNSUPERVISED_KW = [
    "unsupervised",
    "clustering",
    "dimensionality reduction",
    "generative model",
    "vae",
    "gan",
    "density estimation",
    "normalizing flow",
    "anomaly detection",
]
REINFORCEMENT_KW = [
    "reinforcement learning",
    "policy gradient",
    "reward",
    "q-learning",
    "actor-critic",
    "value function",
    "planning",
]
SELF_SUPERVISED_KW = [
    "self-supervised",
    "contrastive learning",
    "masked prediction",
    "pretext task",
    "simclr",
    "mae",
    "bert",
    "masked language",
    "masked image",
]
META_LEARNING_KW = [
    "meta-learning",
    "few-shot",
    "maml",
    "hyperparameter",
    "neural architecture search",
    "automl",
    "learning to learn",
]
CONTINUAL_KW = [
    "continual learning",
    "catastrophic forgetting",
    "lifelong learning",
    "incremental learning",
    "experience replay",
    "elastic weight",
]
TRANSFER_KW = [
    "transfer learning",
    "domain adaptation",
    "pretraining",
    "fine-tuning",
    "foundation model",
    "pre-trained",
    "pretrained",
    "downstream",
]
MULTI_AGENT_KW = [
    "multi-agent",
    "emergent communication",
    "cooperative learning",
    "competitive learning",
    "multi-agent reinforcement",
    "decentralized",
    "social learning",
]
ACTIVE_KW = [
    "active learning",
    "query strategy",
    "human-in-the-loop",
    "annotation",
    "label query",
    "uncertainty sampling",
]
ONLINE_KW = [
    "online learning",
    "streaming",
    "non-stationary",
    "concept drift",
    "bandit",
    "regret",
    "stochastic optimization",
]
FEDERATED_KW = [
    "federated learning",
    "privacy-preserving",
    "differential privacy",
    "distributed learning",
    "secure aggregation",
    "heterogeneous",
]
CURRICULUM_KW = [
    "curriculum learning",
    "self-paced learning",
    "hard example mining",
    "training schedule",
    "easy to hard",
    "course learning",
]
NEUROSYMBOLIC_KW = [
    "neurosymbolic",
    "neural-symbolic",
    "logic",
    "differentiable reasoning",
    "theorem prover",
    "symbolic reasoning",
    "program synthesis",
]
CAUSAL_KW = [
    "causal inference",
    "causal discovery",
    "counterfactual",
    "structural causal model",
    "intervention",
    "treatment effect",
    "do-calculus",
]
THEORY_KW = [
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
]
ALGORITHM_KW = ["algorithm", "method", "approach", "framework", "proposed"]
ARCHITECTURE_KW = [
    "transformer",
    "cnn",
    "graph neural",
    "attention mechanism",
    "architecture",
    "network design",
    "residual",
    "u-net",
    "gan",
    "diffusion",
    "encoder-decoder",
    "bert",
    "gpt",
]
OPTIMIZATION_KW = [
    "optimizer",
    "adam",
    "sgd",
    "learning rate",
    "regularization",
    "loss function",
    "gradient descent",
    "scheduler",
    "training objective",
]
SCALING_KW = ["scaling law", "compute-optimal", "large-scale", "scale", "chinchilla"]
EFFICIENT_KW = [
    "compression",
    "quantization",
    "distillation",
    "pruning",
    "efficient",
    "efficient inference",
    "speedup",
    "spiking",
    "federated",
]
ROBUST_KW = [
    "adversarial",
    "out-of-distribution",
    "robustness",
    "fairness",
    "safety",
    "certified",
    "perturbation",
    "privacy",
]
APPLICATION_KW = [
    "nlp",
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
]

CAT_KW_MAP = {
    "supervised": SUPERVISED_KW,
    "unsupervised": UNSUPERVISED_KW,
    "reinforcement": REINFORCEMENT_KW,
    "self-supervised": SELF_SUPERVISED_KW,
    "meta-learning": META_LEARNING_KW,
    "continual": CONTINUAL_KW,
    "transfer": TRANSFER_KW,
    "multi-agent": MULTI_AGENT_KW,
    "active": ACTIVE_KW,
    "online": ONLINE_KW,
    "federated": FEDERATED_KW,
    "curriculum": CURRICULUM_KW,
    "neurosymbolic": NEUROSYMBOLIC_KW,
    "causal": CAUSAL_KW,
}
SUB_KW_MAP = {
    "theory": THEORY_KW,
    "algorithm": ALGORITHM_KW,
    "architecture": ARCHITECTURE_KW,
    "optimization": OPTIMIZATION_KW,
    "scaling": SCALING_KW,
    "efficient": EFFICIENT_KW,
    "robust": ROBUST_KW,
    "application": APPLICATION_KW,
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


def classify_paper(title, abstract, query_hint="auto"):
    text = f"{title} {abstract}".lower()
    cat_scores = {}
    for cat_name, kw_list in CAT_KW_MAP.items():
        cat_scores[cat_name] = sum(1 for kw in kw_list if kw in text)
    if query_hint != "auto" and query_hint in cat_scores:
        cat_scores[query_hint] += 3
    best_score = max(cat_scores.values())
    category = max(cat_scores, key=cat_scores.get) if best_score > 0 else "supervised"

    sub_scores = {}
    for sub_name, kw_list in SUB_KW_MAP.items():
        sub_scores[sub_name] = sum(1 for kw in kw_list if kw in text)
    sub_best = max(sub_scores.values())
    subcategory = max(sub_scores, key=sub_scores.get) if sub_best > 0 else "algorithm"
    return category, subcategory


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
    print(f"Running {len(QUERIES)} remaining queries", flush=True)

    total_new = 0
    seen_ids = set(by_id.keys())
    seen_titles = set(titles_lower)

    for qi, query in enumerate(QUERIES):
        hint = get_hint(qi)
        print(f"\n  Query {qi + 1}/{len(QUERIES)} [{hint}]...", flush=True)
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
            category, subcategory = classify_paper(title, abstract, hint)
            new_paper = {
                "title": title,
                "date": entry.get("date", ""),
                "url": entry.get("url", ""),
                "category": category,
                "subcategory": subcategory,
                "authors": [],
                "venue": "",
                "code_url": "",
                "project_url": "",
                "abstract": abstract,
                "tags": [f"auto-{category}", f"auto-{subcategory}"],
            }
            if arxiv_id:
                seen_ids.add(arxiv_id)
            seen_titles.add(title_lower)
            titles_lower.append(title_lower)
            papers.append(new_paper)
            batch_new += 1
            total_new += 1
            if arxiv_id:
                by_id[arxiv_id] = new_paper
            print(f"    NEW [{category}/{subcategory}] {title[:70]}", flush=True)

        print(f"    batch added: {batch_new}", flush=True)
        time.sleep(API_DELAY)

        if (qi + 1) % 20 == 0:
            save_papers(yaml_path, data, papers)
            cat_counter = Counter(p.get("category", "?") for p in papers)
            print(f"    [checkpoint] saved {len(papers)} papers", flush=True)

    save_papers(yaml_path, data, papers)
    print(f"\nSaved {len(papers)} total papers", flush=True)

    cat_counter = Counter()
    sub_counter = Counter()
    for p in papers:
        cat_counter[p.get("category", "unknown")] += 1
        sub_counter[p.get("subcategory", "unknown")] += 1

    print(f"\n{'=' * 60}", flush=True)
    print("FINAL DISTRIBUTION", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"Total papers: {len(papers)}", flush=True)
    print(f"New papers added this run: {total_new}", flush=True)
    print(f"\nBy category:", flush=True)
    for cat in ALL_CATEGORIES:
        print(f"  {cat}: {cat_counter.get(cat, 0)}", flush=True)
    print(f"\nBy subcategory:", flush=True)
    for sub in ALL_SUBCATEGORIES:
        print(f"  {sub}: {sub_counter.get(sub, 0)}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Saturate papers.yaml by searching arXiv comprehensively for learning papers.

Runs 50+ diverse queries across cs.AI, cs.CL, cs.LG, cs.RO, cs.CV, cs.IT
within a 48-month window. Auto-classifies, deduplicates, and loops until
saturation (<5 new). Saves after each round to survive timeouts.
"""

import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher
from pathlib import Path

import requests
import yaml

ARXIV_ID_PATTERN = re.compile(r"(\d{4}\.\d{4,5})")
ARXIV_SEARCH_API = (
    "http://export.arxiv.org/api/query?search_query={}&start={}&max_results={}"
)
API_DELAY = 3
SATURATION_THRESHOLD = 5
MAX_RESULTS_PER_QUERY = 100
MONTHS_BACK = 48
MAX_ROUNDS = 3

QUERIES = [
    'cat:cs.LG AND abs:"supervised learning" AND abs:"deep"',
    'cat:cs.LG AND abs:"classification" AND abs:"neural network"',
    'cat:cs.LG AND abs:"unsupervised learning" AND abs:"deep"',
    'cat:cs.LG AND abs:"clustering" AND abs:"deep learning"',
    'cat:cs.LG AND abs:"generative model" AND abs:"unsupervised"',
    'cat:cs.LG AND abs:"variational autoencoder"',
    'cat:cs.LG AND abs:"normalizing flow" AND abs:"density estimation"',
    'cat:cs.LG AND abs:"reinforcement learning" AND abs:"policy optimization"',
    'cat:cs.LG AND abs:"reinforcement learning" AND abs:"reward shaping"',
    'cat:cs.LG AND abs:"model-based reinforcement learning"',
    'cat:cs.LG AND abs:"offline reinforcement learning"',
    'cat:cs.LG AND abs:"multi-agent reinforcement learning"',
    'cat:cs.RO AND abs:"reinforcement learning" AND abs:"robot"',
    'cat:cs.LG AND abs:"self-supervised learning"',
    'cat:cs.CV AND abs:"self-supervised learning"',
    'cat:cs.CL AND abs:"self-supervised learning"',
    'cat:cs.LG AND abs:"contrastive learning" AND abs:"representation"',
    'cat:cs.LG AND abs:"masked language modeling"',
    'cat:cs.LG AND abs:"pretext task" AND abs:"self-supervised"',
    'cat:cs.LG AND abs:"meta-learning" AND abs:"few-shot"',
    'cat:cs.LG AND abs:"MAML" AND abs:"meta-learning"',
    'cat:cs.LG AND abs:"neural architecture search"',
    'cat:cs.LG AND abs:"hyperparameter optimization" AND abs:"neural"',
    'cat:cs.LG AND abs:"continual learning" AND abs:"catastrophic forgetting"',
    'cat:cs.LG AND abs:"lifelong learning"',
    'cat:cs.LG AND abs:"incremental learning" AND abs:"class"',
    'cat:cs.LG AND abs:"experience replay" AND abs:"continual"',
    'cat:cs.LG AND abs:"transfer learning" AND abs:"domain adaptation"',
    'cat:cs.LG AND abs:"pretraining" AND abs:"fine-tuning"',
    'cat:cs.LG AND abs:"foundation model" AND abs:"downstream"',
    'cat:cs.CL AND abs:"pretraining" AND abs:"fine-tuning" AND abs:"language model"',
    'cat:cs.LG AND abs:"multi-agent learning" AND abs:"emergent communication"',
    'cat:cs.LG AND abs:"cooperative learning" AND abs:"multi-agent"',
    'cat:cs.LG AND abs:"competitive learning" AND abs:"multi-agent"',
    'cat:cs.LG AND abs:"active learning" AND abs:"query strategy"',
    'cat:cs.LG AND abs:"human-in-the-loop" AND abs:"machine learning"',
    'cat:cs.LG AND abs:"online learning" AND abs:"streaming"',
    'cat:cs.LG AND abs:"bandit" AND abs:"non-stationary"',
    'cat:cs.LG AND abs:"multi-armed bandit" AND abs:"contextual"',
    'cat:cs.LG AND abs:"federated learning"',
    'cat:cs.LG AND abs:"differential privacy" AND abs:"machine learning"',
    'cat:cs.LG AND abs:"distributed learning" AND abs:"privacy"',
    'cat:cs.LG AND abs:"curriculum learning"',
    'cat:cs.LG AND abs:"self-paced learning"',
    'cat:cs.LG AND abs:"hard example mining" AND abs:"training"',
    'cat:cs.LG AND abs:"neurosymbolic" AND abs:"learning"',
    'cat:cs.AI AND abs:"neurosymbolic" AND abs:"reasoning"',
    'cat:cs.LG AND abs:"neural theorem prover"',
    'cat:cs.LG AND abs:"causal inference" AND abs:"machine learning"',
    'cat:cs.LG AND abs:"causal discovery" AND abs:"structure"',
    'cat:cs.LG AND abs:"counterfactual" AND abs:"learning"',
    'cat:cs.LG AND abs:"structural causal model" AND abs:"neural"',
    'cat:cs.LG AND abs:"scaling law" AND abs:"language model"',
    'cat:cs.LG AND abs:"compute-optimal" AND abs:"training"',
    'cat:cs.LG AND abs:"transformer" AND abs:"attention" AND abs:"architecture"',
    'cat:cs.LG AND abs:"graph neural network" AND abs:"learning"',
    'cat:cs.LG AND abs:"optimizer" AND abs:"learning rate" AND abs:"neural"',
    'cat:cs.LG AND abs:"regularization" AND abs:"deep learning"',
    'cat:cs.LG AND abs:"loss function" AND abs:"deep learning"',
    'cat:cs.LG AND abs:"adversarial robustness" AND abs:"deep"',
    'cat:cs.LG AND abs:"out-of-distribution" AND abs:"generalization"',
    'cat:cs.LG AND abs:"fairness" AND abs:"machine learning"',
    'cat:cs.LG AND abs:"model compression" AND abs:"quantization"',
    'cat:cs.LG AND abs:"knowledge distillation"',
    'cat:cs.LG AND abs:"neural network pruning"',
    'cat:cs.LG AND abs:"efficient inference" AND abs:"deep learning"',
    'cat:cs.LG AND abs:"generalization bound" AND abs:"deep learning"',
    'cat:cs.IT AND abs:"information theory" AND abs:"deep learning"',
    'cat:cs.LG AND abs:"convergence" AND abs:"neural network" AND abs:"theory"',
    'cat:cs.LG AND abs:"deep learning" AND abs:"healthcare"',
    'cat:cs.LG AND abs:"deep learning" AND abs:"scientific discovery"',
    'cat:cs.LG AND abs:"reinforcement learning" AND abs:"NLP"',
    'cat:cs.LG AND abs:"reinforcement learning from human feedback"',
    'cat:cs.CL AND abs:"RLHF" AND abs:"alignment"',
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


def search_arxiv(query, months_back):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cutoff = now - timedelta(days=months_back * 30)
    date_start = cutoff.strftime("%Y%m%d0000")
    date_end = now.strftime("%Y%m%d") + "2359"

    full_query = f"({query}) AND submittedDate:[{date_start} TO {date_end}]"
    try:
        resp = requests.get(
            ARXIV_SEARCH_API.format(
                requests.utils.quote(full_query), 0, MAX_RESULTS_PER_QUERY
            ),
            timeout=30,
        )
        resp.raise_for_status()
        entries = []
        root = resp.text
        for match in re.finditer(r"<entry>(.*?)</entry>", root, re.DOTALL):
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
    except Exception as e:
        print(f"  WARNING: arXiv search error: {e}", flush=True)
        return []


SUPERVISED_KW = [
    "supervised",
    "classification",
    "regression",
    "labeled data",
    "label",
    "prediction task",
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
]
REINFORCEMENT_KW = [
    "reinforcement learning",
    "policy gradient",
    "reward",
    "q-learning",
    "actor-critic",
    "RL",
    "value function",
    "bandit",
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
]
MULTI_AGENT_KW = [
    "multi-agent",
    "emergent communication",
    "cooperative learning",
    "competitive learning",
    "multi-agent reinforcement",
]
ACTIVE_KW = [
    "active learning",
    "query strategy",
    "human-in-the-loop",
    "annotation",
    "label query",
]
ONLINE_KW = [
    "online learning",
    "streaming",
    "non-stationary",
    "concept drift",
    "bandit",
]
FEDERATED_KW = [
    "federated learning",
    "privacy-preserving",
    "differential privacy",
    "distributed learning",
]
CURRICULUM_KW = [
    "curriculum learning",
    "self-paced learning",
    "hard example mining",
    "training schedule",
    "easy to hard",
]
NEUROSYMBOLIC_KW = [
    "neurosymbolic",
    "neural-symbolic",
    "logic",
    "differentiable reasoning",
    "theorem prover",
    "symbolic reasoning",
]
CAUSAL_KW = [
    "causal inference",
    "causal discovery",
    "counterfactual",
    "structural causal model",
    "intervention",
    "causal",
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
]
ROBUST_KW = [
    "adversarial",
    "out-of-distribution",
    "robustness",
    "fairness",
    "safety",
    "certified",
    "perturbation",
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
]


def classify_paper(title, abstract):
    text = f"{title} {abstract}".lower()

    cat_scores = {}
    for cat_name, kw_list in [
        ("supervised", SUPERVISED_KW),
        ("unsupervised", UNSUPERVISED_KW),
        ("reinforcement", REINFORCEMENT_KW),
        ("self-supervised", SELF_SUPERVISED_KW),
        ("meta-learning", META_LEARNING_KW),
        ("continual", CONTINUAL_KW),
        ("transfer", TRANSFER_KW),
        ("multi-agent", MULTI_AGENT_KW),
        ("active", ACTIVE_KW),
        ("online", ONLINE_KW),
        ("federated", FEDERATED_KW),
        ("curriculum", CURRICULUM_KW),
        ("neurosymbolic", NEUROSYMBOLIC_KW),
        ("causal", CAUSAL_KW),
    ]:
        cat_scores[cat_name] = sum(1 for kw in kw_list if kw in text)

    category = (
        max(cat_scores, key=cat_scores.get)
        if max(cat_scores.values()) > 0
        else "supervised"
    )

    sub_scores = {}
    for sub_name, kw_list in [
        ("theory", THEORY_KW),
        ("algorithm", ALGORITHM_KW),
        ("architecture", ARCHITECTURE_KW),
        ("optimization", OPTIMIZATION_KW),
        ("scaling", SCALING_KW),
        ("efficient", EFFICIENT_KW),
        ("robust", ROBUST_KW),
        ("application", APPLICATION_KW),
    ]:
        sub_scores[sub_name] = sum(1 for kw in kw_list if kw in text)

    subcategory = (
        max(sub_scores, key=sub_scores.get)
        if max(sub_scores.values()) > 0
        else "algorithm"
    )

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
            data,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )


def run_round(yaml_path, data, papers, by_id, titles_lower, queries, round_num):
    print(f"\n{'=' * 60}", flush=True)
    print(f"ROUND {round_num}", flush=True)
    print(f"{'=' * 60}", flush=True)

    round_new = []
    seen_ids = set()
    seen_titles = set(titles_lower)

    for qi, query in enumerate(queries):
        cat_match = re.search(r"cat:(\S+)", query)
        cat = cat_match.group(1) if cat_match else "?"
        print(f"\n  Query {qi + 1}/{len(queries)} [{cat}]...", flush=True)

        entries = search_arxiv(query, MONTHS_BACK)
        print(f"    arXiv returned {len(entries)} entries", flush=True)

        for entry in entries:
            arxiv_id_match = ARXIV_ID_PATTERN.search(entry.get("url", ""))
            arxiv_id = arxiv_id_match.group(1) if arxiv_id_match else None

            if arxiv_id and arxiv_id in by_id:
                continue

            if arxiv_id and arxiv_id in seen_ids:
                continue

            title = entry.get("title", "")
            title_lower = title.lower().strip()

            if title_lower in seen_titles:
                continue

            if dedup_title(title, titles_lower):
                continue

            abstract = entry.get("abstract", "")
            category, subcategory = classify_paper(title, abstract)

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
            round_new.append(new_paper)
            by_id[arxiv_id] = new_paper

            print(f"    NEW [{category}/{subcategory}] {title[:70]}", flush=True)

        time.sleep(API_DELAY)

        if (qi + 1) % 20 == 0:
            save_papers(yaml_path, data, papers + round_new)
            print(
                f"    [checkpoint] saved {len(papers) + len(round_new)} papers",
                flush=True,
            )

    print(f"\n  Round {round_num} found {len(round_new)} new papers", flush=True)
    return round_new


def main():
    yaml_path = Path(__file__).resolve().parent.parent.parent / "papers.yaml"
    data, papers, by_id, titles_lower = load_existing_papers(yaml_path)

    print(f"Loaded {len(papers)} existing papers", flush=True)
    print(f"Using {len(QUERIES)} queries", flush=True)
    print(f"Search window: {MONTHS_BACK} months", flush=True)

    total_new = 0
    round_num = 1

    while round_num <= MAX_ROUNDS:
        round_new = run_round(
            yaml_path, data, papers, by_id, titles_lower, QUERIES, round_num
        )

        papers.extend(round_new)
        total_new += len(round_new)

        save_papers(yaml_path, data, papers)
        print(f"  Saved {len(papers)} total papers to {yaml_path}", flush=True)

        if len(round_new) < SATURATION_THRESHOLD:
            print(
                f"\nSATURATED: Round {round_num} found only {len(round_new)} "
                f"new papers (< {SATURATION_THRESHOLD} threshold)",
                flush=True,
            )
            break

        print(
            f"\n  Total new so far: {total_new}, starting round {round_num + 1}...",
            flush=True,
        )
        round_num += 1

    if round_num > MAX_ROUNDS:
        print(f"\nReached max rounds ({MAX_ROUNDS}). Stopping.", flush=True)

    if total_new == 0:
        print("\nNo new papers found. papers.yaml unchanged.", flush=True)

    cat_counter = Counter()
    sub_counter = Counter()
    for p in papers:
        cat_counter[p.get("category", "unknown")] += 1
        sub_counter[p.get("subcategory", "unknown")] += 1

    print(f"\n{'=' * 60}", flush=True)
    print("FINAL DISTRIBUTION", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"Total papers: {len(papers)}", flush=True)
    print(f"New papers added: {total_new}", flush=True)
    print(f"Rounds: {round_num}", flush=True)
    print(f"\nBy category:", flush=True)
    for cat, count in cat_counter.most_common():
        print(f"  {cat}: {count}", flush=True)
    print(f"\nBy subcategory:", flush=True)
    for sub, count in sub_counter.most_common():
        print(f"  {sub}: {count}", flush=True)


if __name__ == "__main__":
    main()

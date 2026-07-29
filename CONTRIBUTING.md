# Contributing to Agent Learning Research

Thank you for your interest in contributing! This guide explains how to add papers to the list.

## Overview

This repository maintains a curated list of papers related to **learning in AI**, covering the full spectrum of machine learning paradigms — from supervised learning to causal inference.

The source of truth is `papers.yaml`. The `README.md` is **auto-generated** from `papers.yaml` — never edit it directly.

## Quick Start: Adding a Paper

1. **Check for duplicates** — search `papers.yaml` by title and URL. If the paper already exists, skip.
2. **Edit `papers.yaml`** — add your entry following the schema below.
3. **Validate** — run `python3 scripts/validate_papers.py`
4. **Regenerate README** — run `python3 scripts/generate_readme.py`
5. **Commit and open a PR** — see the PR checklist below.

## papers.yaml Schema

```yaml
papers:
  - title: "Paper Title"            # Required
    date: "2026-01"                  # Required, YYYY-MM format
    url: "https://arxiv.org/abs/XXXX"  # Required, normalized arXiv URL
    category: "supervised"           # Required: see taxonomy below
    subcategory: "algorithm"         # Required: see taxonomy below
    # Optional fields:
    authors: ["Author1", "Author2"]
    venue: "NeurIPS 2025"
    code_url: "https://github.com/..."
    project_url: "https://..."
    abstract: "..."
    tags: ["tag1", "tag2"]
```

## URL Normalization Rules

- **arXiv papers**: always use `https://arxiv.org/abs/XXXX` format
  - Do NOT use `https://doi.org/10.48550/arXiv.XXXX`
  - Do NOT use `https://www.arxiv.org/abs/XXXX`
  - Do NOT use `https://arxiv.org/pdf/XXXX`
- **Non-arXiv papers**: keep URLs as-is (e.g., `aclanthology.org`, `openreview.net`, `papers.nips.cc`)

## Taxonomy Guide

### Category (learning paradigm)

| Category | Description | Examples |
|----------|-------------|----------|
| **supervised** | Supervised learning, classification, regression, labeled data | Image classification, sentiment analysis, object detection |
| **unsupervised** | Unsupervised learning, clustering, dimensionality reduction, generative modeling | k-means, VAEs, GANs, normalizing flows |
| **reinforcement** | Reinforcement learning, policy optimization, reward learning | PPO, SAC, model-based RL, offline RL |
| **self-supervised** | Self-supervised learning, contrastive learning, masked prediction, pretext tasks | SimCLR, MAE, BERT, BYOL |
| **meta-learning** | Learning to learn, few-shot, MAML, hyperparameter optimization, NAS | ProtoNet, ENAS, Bayesian optimization |
| **continual** | Continual/lifelong learning, catastrophic forgetting avoidance, incremental learning | EWC, PackNet, progressive networks |
| **transfer** | Transfer learning, domain adaptation, pretraining + finetuning, foundation models | GPT, BERT, CLIP, domain adaptation |
| **multi-agent** | Multi-agent learning, emergent communication, cooperative/competitive learning | MARL, communication protocols, social dilemmas |
| **active** | Active learning, query strategies, human-in-the-loop | Uncertainty sampling, query-by-committee |
| **online** | Online/streaming learning, bandits, non-stationary environments | Online SGD, contextual bandits, concept drift |
| **federated** | Federated learning, privacy-preserving distributed learning, differential privacy | FedAvg, FedProx, secure aggregation |
| **curriculum** | Curriculum learning, self-paced learning, hard example mining | Training schedule, self-paced, OHEM |
| **neurosymbolic** | Neurosymbolic learning, logic + neural integration, differentiable reasoning | Neural theorem provers, probabilistic programming |
| **causal** | Causal inference, causal discovery, counterfactual learning, structural causal models | Do-calculus, causal graphs, invariant prediction |

### Subcategory (scope/approach)

| Subcategory | Description | Examples |
|-------------|-------------|----------|
| **theory** | Theoretical foundations, convergence, generalization bounds, information theory | PAC-Bayes, VC dimension, NTK theory |
| **algorithm** | New algorithms, methods, architectures | New optimizer, novel RL algorithm |
| **architecture** | Neural architectures, transformers, CNNs, GNNs, attention mechanisms | Vision transformer, graph attention network |
| **optimization** | Optimizers, regularization, loss functions, learning rate schedules | AdamW, weight decay, label smoothing |
| **scaling** | Scaling laws, compute-optimal training, large-scale training | Chinchilla, scaling behavior |
| **efficient** | Efficient learning, compression, quantization, distillation, pruning | Knowledge distillation, model pruning |
| **robust** | Robustness, adversarial, out-of-distribution, fairness, safety | Adversarial training, OOD detection |
| **application** | Applied learning in specific domains (NLP, CV, robotics, science, healthcare) | Medical imaging, protein folding, NLP tasks |

A paper may belong to one category/subcategory combination. If a paper spans multiple, choose the **primary** contribution.

> **Note on paper counts:** The total paper count in the header and README counts each unique (title, category, subcategory) triple. If the same paper appears under multiple categories, it is counted multiple times.

## Deduplication Checklist

Before adding a paper, check that it is not already in the list:

1. Search `papers.yaml` by **title** (case-insensitive)
2. Search `papers.yaml` by **URL**
3. If the same paper appears under a different category, that is acceptable — each unique (title, category, subcategory) triple is valid

## Local Development Setup

```bash
pip install -r requirements.txt
```

### Useful Commands

| Command | Description |
|---------|-------------|
| `python3 scripts/validate_papers.py` | Validate `papers.yaml` for errors |
| `python3 scripts/validate_papers.py --fix` | Validate and auto-fix URL normalization |
| `python3 scripts/generate_readme.py` | Regenerate `README.md` from `papers.yaml` |
| `python3 scripts/generate_readme.py --check` | Check if README is up-to-date (CI use) |
| `python3 scripts/fetch/fetch_metadata_bulk.py` | Bulk-fetch authors/venue/abstract from arXiv |
| `python3 scripts/fetch/fetch_new_papers.py` | Discover new learning papers from arXiv |
| `python3 scripts/fetch/fetch_new_papers.py --dry-run` | Preview new papers without creating anything |
| `python3 scripts/fetch/saturate_papers.py` | Saturate with many arXiv queries |
| `python3 scripts/export_bibtex.py` | Export all papers to BibTeX format |
| `python3 scripts/analysis/generate_analysis.py` | Generate D3.js graph analysis |

## PR Process

1. Fork this repository
2. Create a branch: `git checkout -b add-paper-name`
3. Edit `papers.yaml` to add your paper entry
4. Run the validator: `python3 scripts/validate_papers.py`
5. Run the README generator: `python3 scripts/generate_readme.py`
6. Commit your changes
7. Open a pull request

## PR Checklist

- [ ] Added entry to `papers.yaml` (not `README.md`)
- [ ] Used normalized URL format (`https://arxiv.org/abs/XXXX`)
- [ ] Checked for duplicates (searched by title and URL)
- [ ] Ran `python3 scripts/validate_papers.py` — no errors
- [ ] Ran `python3 scripts/generate_readme.py` — README updated
- [ ] Used correct date format (YYYY-MM)

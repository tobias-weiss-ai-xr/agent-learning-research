#!/usr/bin/env python3
"""Discover new learning papers from arXiv API across all major learning paradigms."""

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
import yaml

ARXIV_ID_PATTERN = re.compile(r"(\d{4}\.\d{4,5})")
ARXIV_SEARCH_API = (
    "http://export.arxiv.org/api/query?search_query={}&start={}&max_results={}"
)
QUERIES = [
    'cat:cs.LG AND abs:"supervised learning" AND abs:"deep learning"',
    'cat:cs.LG AND abs:"classification" AND abs:"deep neural"',
    'cat:cs.LG AND abs:"regression" AND abs:"neural network"',
    'cat:cs.LG AND abs:"unsupervised learning" AND abs:"deep learning"',
    'cat:cs.LG AND abs:"clustering" AND abs:"deep"',
    'cat:cs.LG AND abs:"generative model" AND abs:"unsupervised"',
    'cat:cs.LG AND abs:"dimensionality reduction" AND abs:"neural"',
    'cat:cs.LG AND abs:"reinforcement learning" AND abs:"policy"',
    'cat:cs.LG AND abs:"reinforcement learning" AND abs:"reward"',
    'cat:cs.LG AND abs:"model-based reinforcement learning"',
    'cat:cs.LG AND abs:"offline reinforcement learning"',
    'cat:cs.LG AND abs:"self-supervised learning" AND abs:"pretraining"',
    'cat:cs.LG AND abs:"contrastive learning" AND abs:"self-supervised"',
    'cat:cs.CV AND abs:"self-supervised learning"',
    'cat:cs.LG AND abs:"masked prediction" AND abs:"pretraining"',
    'cat:cs.LG AND abs:"meta-learning" AND abs:"few-shot"',
    'cat:cs.LG AND abs:"MAML" AND abs:"meta-learning"',
    'cat:cs.LG AND abs:"neural architecture search" AND abs:"meta"',
    'cat:cs.LG AND abs:"hyperparameter optimization" AND abs:"neural"',
    'cat:cs.LG AND abs:"continual learning" AND abs:"catastrophic forgetting"',
    'cat:cs.LG AND abs:"lifelong learning" AND abs:"incremental"',
    'cat:cs.LG AND abs:"transfer learning" AND abs:"domain adaptation"',
    'cat:cs.LG AND abs:"pretraining" AND abs:"fine-tuning" AND abs:"foundation"',
    'cat:cs.LG AND abs:"foundation model" AND abs:"transfer"',
    'cat:cs.LG AND abs:"multi-agent learning" AND abs:"emergent"',
    'cat:cs.LG AND abs:"multi-agent" AND abs:"cooperative" AND abs:"learning"',
    'cat:cs.LG AND abs:"multi-agent" AND abs:"competitive" AND abs:"learning"',
    'cat:cs.LG AND abs:"active learning" AND abs:"query strategy"',
    'cat:cs.LG AND abs:"human-in-the-loop" AND abs:"learning"',
    'cat:cs.LG AND abs:"online learning" AND abs:"streaming"',
    'cat:cs.LG AND abs:"bandit" AND abs:"non-stationary"',
    'cat:cs.LG AND abs:"federated learning" AND abs:"privacy"',
    'cat:cs.LG AND abs:"differential privacy" AND abs:"federated"',
    'cat:cs.LG AND abs:"distributed learning" AND abs:"privacy-preserving"',
    'cat:cs.LG AND abs:"curriculum learning" AND abs:"self-paced"',
    'cat:cs.LG AND abs:"hard example mining" AND abs:"training"',
    'cat:cs.LG AND abs:"neurosymbolic" AND abs:"learning"',
    'cat:cs.AI AND abs:"neurosymbolic" AND abs:"reasoning"',
    'cat:cs.LG AND abs:"logic" AND abs:"neural" AND abs:"differentiable"',
    'cat:cs.LG AND abs:"causal inference" AND abs:"machine learning"',
    'cat:cs.LG AND abs:"causal discovery" AND abs:"structural"',
    'cat:cs.LG AND abs:"counterfactual" AND abs:"learning"',
    'cat:cs.LG AND abs:"structural causal model" AND abs:"learning"',
    'cat:cs.AI AND abs:"scaling law" AND abs:"neural network"',
    'cat:cs.LG AND abs:"compute-optimal training"',
    'cat:cs.LG AND abs:"neural architecture" AND abs:"transformer"',
    'cat:cs.LG AND abs:"attention mechanism" AND abs:"learning"',
    'cat:cs.LG AND abs:"optimization" AND abs:"learning rate" AND abs:"neural"',
    'cat:cs.LG AND abs:"regularization" AND abs:"deep learning"',
    'cat:cs.LG AND abs:"adversarial robustness" AND abs:"deep learning"',
    'cat:cs.LG AND abs:"out-of-distribution" AND abs:"generalization"',
    'cat:cs.LG AND abs:"model compression" AND abs:"quantization" AND abs:"distillation"',
    'cat:cs.LG AND abs:"knowledge distillation" AND abs:"neural network"',
    'cat:cs.RO AND abs:"reinforcement learning" AND abs:"robot"',
    'cat:cs.CL AND abs:"pretraining" AND abs:"language model" AND abs:"learning"',
    'cat:cs.CL AND abs:"instruction tuning" AND abs:"RLHF"',
    'cat:cs.CL AND abs:"reward learning" AND abs:"language"',
]


def load_existing_papers(yaml_path):
    if not yaml_path.exists():
        return {}, []
    with open(yaml_path, "r") as f:
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
    return by_id, titles_lower


def search_arxiv(query, months, start=0, max_results=100):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cutoff = now - timedelta(days=months * 30)
    date_start = cutoff.strftime("%Y%m%d0000")
    date_end = now.strftime("%Y%m%d") + "2359"

    full_query = f"({query}) AND submittedDate:[{date_start} TO {date_end}]"
    try:
        resp = requests.get(
            ARXIV_SEARCH_API.format(
                requests.utils.quote(full_query), start, max_results
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


def format_yaml_entry(entry):
    title = entry["title"].replace('"', '\\"')
    lines = [
        f'  - title: "{title}"',
        f'    date: "{entry.get("date", "")}"',
        f'    url: "{entry.get("url", "")}"',
        f'    category: ""  # TODO: supervised|unsupervised|reinforcement|self-supervised|meta-learning|continual|transfer|multi-agent|active|online|federated|curriculum|neurosymbolic|causal',
        f'    subcategory: ""  # TODO: theory|algorithm|architecture|optimization|scaling|efficient|robust|application',
    ]
    if entry.get("abstract"):
        abstract = entry["abstract"][:200].replace('"', '\\"')
        lines.append(f'    abstract: "{abstract}..."')
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Discover new learning papers from arXiv"
    )
    parser.add_argument(
        "--months",
        type=int,
        default=3,
        help="Search papers from the last N months (default: 3)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without creating anything"
    )
    parser.add_argument(
        "--create-pr", action="store_true", help="Create a GitHub PR with new papers"
    )
    args = parser.parse_args()

    yaml_path = Path(__file__).resolve().parent.parent.parent / "papers.yaml"
    by_id, titles_lower = load_existing_papers(yaml_path)

    print(f"Loaded {len(by_id)} existing papers from papers.yaml", flush=True)
    print(
        f"Searching arXiv for papers from the last {args.months} month(s)...",
        flush=True,
    )

    all_new = []
    for qi, query in enumerate(QUERIES):
        print(f"\nQuery {qi + 1}/{len(QUERIES)}...", flush=True)
        entries = search_arxiv(query, args.months)
        for entry in entries:
            arxiv_id_match = ARXIV_ID_PATTERN.search(entry.get("url", ""))
            arxiv_id = arxiv_id_match.group(1) if arxiv_id_match else None

            if arxiv_id and arxiv_id in by_id:
                continue

            title_lower = entry.get("title", "").lower().strip()
            if any(title_lower == t for t in titles_lower):
                continue

            if arxiv_id and any(e.get("url", "") == entry["url"] for e in all_new):
                continue

            all_new.append(entry)

        time.sleep(3)

    print(
        f"\nFound {len(all_new)} new papers ({len(by_id)} already in list)", flush=True
    )

    if not all_new:
        print("No new papers to add.", flush=True)
        return

    print("\n--- New Papers ---", flush=True)
    for entry in all_new:
        print(format_yaml_entry(entry), flush=True)
        print(flush=True)

    if args.dry_run:
        print("\nDry run complete — no files modified", flush=True)
        return

    if args.create_pr:
        branch_name = f"add-new-papers-{datetime.now().strftime('%Y%m%d')}"

        print(f"\nCreating branch '{branch_name}' and PR...", flush=True)

        try:
            subprocess.run(
                ["git", "checkout", "-b", branch_name], check=True, cwd=yaml_path.parent
            )
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}
            papers = data.get("papers", [])
            for entry in all_new:
                papers.append(
                    {
                        "title": entry.get("title", ""),
                        "date": entry.get("date", ""),
                        "url": entry.get("url", ""),
                        "category": "",
                        "subcategory": "",
                        "abstract": entry.get("abstract", ""),
                    }
                )
            data["papers"] = papers
            with open(yaml_path, "w") as f:
                yaml.dump(
                    data,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
            subprocess.run(
                ["git", "add", "papers.yaml"], check=True, cwd=yaml_path.parent
            )
            subprocess.run(
                [
                    "git",
                    "commit",
                    "-m",
                    f"Add {len(all_new)} new papers from arXiv discovery",
                ],
                check=True,
                cwd=yaml_path.parent,
            )
            subprocess.run(
                ["git", "push", "origin", branch_name], check=True, cwd=yaml_path.parent
            )
            subprocess.run(
                [
                    "gh",
                    "pr",
                    "create",
                    "--title",
                    f"Add {len(all_new)} new papers from arXiv discovery",
                    "--body",
                    f"Automatically discovered {len(all_new)} new papers.\n\n**Please review taxonomy assignments.**",
                ],
                check=True,
                cwd=yaml_path.parent,
            )
            print("PR created successfully!", flush=True)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to create PR: {e}", flush=True)
            sys.exit(1)
    else:
        print(
            "\nTo add these papers, re-run with --create-pr or manually add to papers.yaml",
            flush=True,
        )


if __name__ == "__main__":
    main()

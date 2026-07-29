#!/usr/bin/env python3
"""Generate README.md and docs/papers.json from papers.yaml."""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml


CATEGORY_ORDER = [
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

SUBCATEGORY_ORDER = [
    "theory",
    "algorithm",
    "architecture",
    "optimization",
    "scaling",
    "efficient",
    "robust",
    "application",
]

CATEGORY_DISPLAY = {
    "supervised": "Supervised Learning",
    "unsupervised": "Unsupervised Learning",
    "reinforcement": "Reinforcement Learning",
    "self-supervised": "Self-Supervised Learning",
    "meta-learning": "Meta-Learning",
    "continual": "Continual Learning",
    "transfer": "Transfer Learning",
    "multi-agent": "Multi-Agent Learning",
    "active": "Active Learning",
    "online": "Online Learning",
    "federated": "Federated Learning",
    "curriculum": "Curriculum Learning",
    "neurosymbolic": "Neurosymbolic Learning",
    "causal": "Causal Learning",
}

SUBCATEGORY_DISPLAY = {
    "theory": "Theory",
    "algorithm": "Algorithms",
    "architecture": "Architecture",
    "optimization": "Optimization",
    "scaling": "Scaling",
    "efficient": "Efficiency",
    "robust": "Robustness",
    "application": "Applications",
}


def github_anchor(text):
    return (
        re.sub(r"[^\w\s-]", "", text)
        .strip()
        .lower()
        .replace(" ", "-")
        .replace("--", "-")
    )


def load_papers(path):
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("papers", [])


def render_toc(papers):
    lines = []
    for cat in CATEGORY_ORDER:
        cat_display = CATEGORY_DISPLAY[cat]
        cat_anchor = github_anchor(cat_display)
        cat_papers = [p for p in papers if p["category"] == cat]
        count = len(cat_papers)
        lines.append(f"- [{cat_display}](#{cat_anchor}) ({count})")
        for sub in SUBCATEGORY_ORDER:
            sub_papers = [p for p in cat_papers if p["subcategory"] == sub]
            if not sub_papers:
                continue
            sub_display = SUBCATEGORY_DISPLAY[sub]
            sub_anchor = github_anchor(sub_display)
            lines.append(f"  - [{sub_display}](#{sub_anchor}) ({len(sub_papers)})")
    return lines


def render_paper_list(papers):
    lines = ["<!-- PAPER_LIST_START -->", ""]

    lines.append("### Table of Contents")
    lines.append("")
    for line in render_toc(papers):
        lines.append(line)
    lines.append("")

    for cat in CATEGORY_ORDER:
        cat_display = CATEGORY_DISPLAY[cat]
        cat_papers = [p for p in papers if p["category"] == cat]
        if not cat_papers:
            continue

        lines.append(f"### {cat_display}")
        lines.append("")

        for sub in SUBCATEGORY_ORDER:
            group = [p for p in cat_papers if p["subcategory"] == sub]
            if not group:
                continue

            sub_display = SUBCATEGORY_DISPLAY[sub]
            lines.append(f"#### {sub_display}")
            lines.append("")

            year_groups = defaultdict(list)
            for p in group:
                year = p["date"][:4]
                year_groups[year].append(p)

            for year in sorted(year_groups.keys(), reverse=True):
                lines.append(f"##### {year}")
                lines.append("")

                sorted_papers = sorted(
                    year_groups[year], key=lambda p: p["date"], reverse=True
                )
                for p in sorted_papers:
                    y = p["date"][:4]
                    title = p["title"]
                    url = p["url"]
                    venue = p.get("venue", "")
                    code_url = p.get("code_url", "")
                    project_url = p.get("project_url", "")

                    entry = f"- [{y}] **{title}**"
                    if venue:
                        entry += f" *{venue}*"
                    entry += f" [[paper]({url})]"
                    if code_url:
                        entry += f" [[code]({code_url})]"
                    if project_url:
                        entry += f" [[project]({project_url})]"
                    lines.append(entry)

                lines.append("")

            lines.append("[Back to top](#paper-list)")
            lines.append("")

    lines.append("<!-- PAPER_LIST_END -->")
    return "\n".join(lines)


def generate_readme(papers, readme_path, check_mode=False):
    readme_text = readme_path.read_text(encoding="utf-8")

    start_marker = "<!-- PAPER_LIST_START -->"
    end_marker = "<!-- PAPER_LIST_END -->"

    start_idx = readme_text.find(start_marker)
    end_idx = readme_text.find(end_marker)

    if start_idx == -1 or end_idx == -1:
        print(
            "Error: Could not find PAPER_LIST_START/END markers in README.md",
            file=sys.stderr,
        )
        sys.exit(1)

    before = readme_text[:start_idx]
    after = readme_text[end_idx + len(end_marker) :]

    generated_list = render_paper_list(papers)
    new_readme = before + generated_list + after

    if check_mode:
        if new_readme == readme_text:
            print("README.md is up-to-date.")
            sys.exit(0)
        else:
            print(
                "README.md is out-of-date. Run generate_readme.py without --check to update.",
                file=sys.stderr,
            )
            sys.exit(1)

    readme_path.write_text(new_readme, encoding="utf-8")
    print(f"Generated {readme_path}")


def generate_json(papers, json_path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps({"papers": papers}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Generated {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate README.md and papers.json from papers.yaml"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if README is up-to-date (exit 1 if not)",
    )
    parser.add_argument(
        "--skip-json", action="store_true", help="Skip generating papers.json"
    )
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    papers_yaml = base / "papers.yaml"
    readme_path = base / "README.md"
    json_path = base / "docs" / "papers.json"

    papers = load_papers(papers_yaml)

    generate_readme(papers, readme_path, check_mode=args.check)

    if not args.check and not args.skip_json:
        generate_json(papers, json_path)


if __name__ == "__main__":
    main()

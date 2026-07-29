#!/usr/bin/env python3
"""Export papers from papers.yaml to BibTeX format."""

import re
import sys
from pathlib import Path

import yaml


def sanitize_bibtex(text):
    text = text.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
    text = re.sub(r"([{}_])", r"\\\1", text)
    return text


def make_bibtex_key(paper, index):
    title = paper.get("title", f"paper{index}").lower()
    words = re.findall(r"[a-z]+", title)
    if len(words) >= 3:
        key = f"{words[0]}{words[1]}{words[2]}"
    elif words:
        key = "".join(words)
    else:
        key = f"paper{index}"
    date = paper.get("date", "")
    year = date[:4] if date else "0000"
    authors = paper.get("authors", [])
    if authors:
        first = re.sub(r"[^a-zA-Z]", "", authors[0].split()[-1].lower())
        key = f"{first}{year}{key[:20]}"
    else:
        key = f"{year}{key[:25]}"
    return key


def main():
    yaml_path = Path(__file__).resolve().parent.parent / "papers.yaml"
    if not yaml_path.exists():
        print(f"ERROR: {yaml_path} not found", flush=True)
        sys.exit(1)

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    papers = data.get("papers", [])
    if not papers:
        print("No papers to export.", flush=True)
        return

    output_path = Path(__file__).resolve().parent.parent / "paper" / "references.bib"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    for i, paper in enumerate(papers):
        key = make_bibtex_key(paper, i)
        title = sanitize_bibtex(paper.get("title", ""))
        url = paper.get("url", "")
        date = paper.get("date", "")
        year = date[:4] if date else "0000"
        authors = paper.get("authors", [])
        author_str = " and ".join(authors) if authors else "Unknown"
        venue = paper.get("venue", "")
        abstract = paper.get("abstract", "")

        entry_lines = [
            f"@article{{{key},",
            f"  title = {{{title}}},",
            f"  author = {{{author_str}}},",
            f"  year = {{{year}}},",
            f"  url = {{{url}}},",
        ]
        if venue:
            entry_lines.append(f"  journal = {{{sanitize_bibtex(venue)}}},")
        if abstract:
            abs_san = sanitize_bibtex(abstract[:500])
            entry_lines.append(f"  abstract = {{{abs_san}}},")
        entry_lines.append("}")
        entries.append("\n".join(entry_lines))

    bib_content = "\n\n".join(entries) + "\n"
    output_path.write_text(bib_content, encoding="utf-8")
    print(f"Exported {len(papers)} papers to {output_path}", flush=True)


if __name__ == "__main__":
    main()

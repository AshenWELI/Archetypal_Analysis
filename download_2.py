"""
download_aa_abstracts.py
------------------------
Collects abstracts of Archetypal Analysis papers automatically using:
 - arxiv (for open-access papers)
 - scholarly (for Google Scholar)
 - bibtexparser (for .bib reference lists)

Output: aa_abstracts.jsonl
"""

import os
import json
import time
import arxiv
from scholarly import scholarly, ProxyGenerator
import bibtexparser
from tqdm import tqdm


# ==============================
# 1. SETUP
# ==============================
# Setup scholarly proxy (to avoid blocking)
#pg = ProxyGenerator()
#pg.FreeProxies()
#scholarly.use_proxy(pg)

OUTPUT_FILE = "aa_abstracts.jsonl"
if os.path.exists(OUTPUT_FILE):
    print(f"Appending to existing file: {OUTPUT_FILE}")
else:
    print(f"Creating new file: {OUTPUT_FILE}")


# ==============================
# 2. READ TITLES FROM .BIB FILE
# ==============================
def read_bib_titles(bib_path="references.bib"):
    """Extract titles from a .bib file."""
    if not os.path.exists(bib_path):
        print("No references.bib found. You can manually add titles instead.")
        return []
    with open(bib_path, encoding="utf8") as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)
    titles = [entry.get("title", "").strip("{}") for entry in bib_database.entries if "title" in entry]
    print(f"Loaded {len(titles)} titles from {bib_path}")
    return titles


# ==============================
# 3. FETCH ABSTRACTS FROM ARXIV
# ==============================
def fetch_from_arxiv(query):
    """Try to fetch paper metadata from arXiv API."""
    search = arxiv.Search(query=query, max_results=1)
    for result in search.results():
        return {
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "abstract": result.summary.strip(),
            "published": str(result.published.date()),
            "source": "arxiv",
            "url": result.entry_id
        }
    return None


# ==============================
# 4. FETCH ABSTRACTS FROM SCHOLAR
# ==============================
def fetch_from_scholar(title):
    """Fallback: fetch paper info from Google Scholar."""
    try:
        search = scholarly.search_pubs(title)
        paper = next(search, None)
        if paper:
            full = scholarly.fill(paper)
            return {
                "title": full.get("bib", {}).get("title"),
                "authors": full.get("bib", {}).get("author"),
                "abstract": full.get("bib", {}).get("abstract"),
                "year": full.get("bib", {}).get("pub_year"),
                "url": full.get("eprint_url") or full.get("pub_url"),
                "source": "scholar"
            }
    except Exception as e:
        print(f"[SCHOLAR ERROR] {e}")
    return None


# ==============================
# 5. MAIN FUNCTION
# ==============================
def collect_abstracts(titles):
    """Main routine to collect abstracts for all given titles."""
    with open(OUTPUT_FILE, "a", encoding="utf8") as fout:
        for title in tqdm(titles, desc="Fetching abstracts"):
            data = None
            # 1. Try arXiv first
            data = fetch_from_arxiv(title)
            if not data:
                # 2. Try Google Scholar
                data = fetch_from_scholar(title)
            if data and data.get("abstract"):
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                print(f"[OK] {data['title']}")
            else:
                print(f"[MISS] {title}")
            time.sleep(2)  # polite delay


# ==============================
# 6. RUN SCRIPT
# ==============================
if __name__ == "__main__":
    # Option 1: Load from references.bib
    titles = read_bib_titles("references.bib")

    # Option 2 (manual): If no .bib file, define titles directly
    if not titles:
        titles = [
            "Archetypal analysis for machine learning and data mining",
            "Archetypal Analysis: An Algorithmic Perspective",
            "Deep archetypal analysis: Uncovering patterns in complex data",
            "A survey on Archetypal Analysis methods",
        ]

    collect_abstracts(titles)
    print("\n Finished! Results saved to", OUTPUT_FILE)

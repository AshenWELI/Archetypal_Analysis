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
import re
import arxiv
from scholarly import scholarly, ProxyGenerator
import bibtexparser
from tqdm import tqdm


# ==============================
# 1. SETUP
# ==============================
# Setup scholarly proxy (to avoid blocking)
# pg = ProxyGenerator()
# pg.FreeProxies()
# scholarly.use_proxy(pg)

OUTPUT_FILE = "aa_abstracts.jsonl"
if os.path.exists(OUTPUT_FILE):
    print(f"Appending to existing file: {OUTPUT_FILE}")
else:
    print(f"Creating new file: {OUTPUT_FILE}")


# ==============================
# 2. LATEX TO UNICODE CONVERSION
# ==============================

# Ref : <https://tex.stackexchange.com/questions/256836/getting-swedish-letters-to-work-in-latex>
LATEX_TO_UNICODE = {
    r'{\"o}': 'ö',
    r'{\o}': 'ø',
    r'{\"a}': 'ä',
    r'{\"u}': 'ü',
    r'{\"O}': 'Ö',
    r'{\"A}': 'Ä',
    r'{\"U}': 'Ü',
    r"{\\'e}": 'é',
    r"{\\'a}": 'á',
    r'{\\`e}': 'è',
    r'{\\^e}': 'ê',
    r'{\\~n}': 'ñ',
    r'{\\c{c}}': 'ç',
    r'\\&': '&',
    r'\\%': '%',
    r'\\$': '$',
}

def clean_latex_title(title):
    """
    Clean LaTeX formatting from BibTeX title.
    - Converts LaTeX special characters to Unicode
    - Removes curly braces
    - Normalizes whitespace
    """
    if not title:
        return ""
    
    # Remove curly braces used for case protection
    title = title.replace('{', '').replace('}', '')
    
    # Convert common LaTeX characters
    for latex, unicode_char in LATEX_TO_UNICODE.items():
        title = title.replace(latex, unicode_char)
    
    # Remove any remaining backslashes
    title = re.sub(r'\\[a-zA-Z]+', '', title)
    
    # Normalize whitespace
    title = ' '.join(title.split())
    
    return title.strip()


def extract_arxiv_id(entry):
    """
    Extract arXiv ID from BibTeX entry.
    Looks in: eprint, arxivId, archivePrefix fields, or in the journal field.
    """
    # Direct fields
    if 'eprint' in entry:
        return entry['eprint'].strip()
    if 'arxivId' in entry:
        return entry['arxivId'].strip()
    
    # Check journal field for arXiv preprint
    journal = entry.get('journal', '')
    if 'arxiv' in journal.lower():
        # Extract ID from "arXiv preprint arXiv:2301.13748"
        match = re.search(r'arXiv:(\d+\.\d+)', journal)
        if match:
            return match.group(1)
    
    return None


# ==============================
# 3. READ TITLES FROM .BIB FILE
# ==============================
def read_bib_entries(bib_path="references.bib"):
    """
    Extract paper information from a .bib file.
    Returns list of dicts with title, arxiv_id, authors, year.
    """
    if not os.path.exists(bib_path):
        print(f"No {bib_path} found. Manually add titles instead.")
        return []
    
    with open(bib_path, encoding="utf8") as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)
    
    papers = []
    for entry in bib_database.entries:
        if "title" not in entry:
            continue
        
        # Clean the title
        raw_title = entry.get("title", "")
        clean_title = clean_latex_title(raw_title)
        
        # Extract arXiv ID if available
        arxiv_id = extract_arxiv_id(entry)
        
        paper = {
            "title": clean_title,
            "raw_title": raw_title,
            "arxiv_id": arxiv_id,
            "authors": entry.get("author", "").split(" and "),
            "year": entry.get("year", ""),
            "journal": entry.get("journal", ""),
        }
        papers.append(paper)
    
    print(f"Loaded {len(papers)} papers from {bib_path}")
    return papers


# ==============================
# 4. FETCH ABSTRACTS FROM ARXIV
# ==============================
def fetch_from_arxiv_by_id(arxiv_id):
    """
    Fetch paper metadata from arXiv using the arXiv ID directly.
    This is much more reliable than searching by title.
    """
    try:
        search = arxiv.Search(id_list=[arxiv_id])
        client = arxiv.Client()
        result = next(client.results(search), None)
        
        if result is None:
            return None
        
        return {
            "title": result.title.strip(),
            "authors": [a.name for a in result.authors],
            "abstract": result.summary.strip(),
            "published": str(result.published.date()) if result.published else None,
            "source": "arxiv",
            "url": result.entry_id,
            "arxiv_id": arxiv_id
        }
    
    except Exception as e:
        print(f"[ARXIV ID ERROR] id='{arxiv_id}' → {e}")
        return None


def fetch_from_arxiv_by_title(title):
    """
    Try to fetch paper metadata from the arXiv API by searching the title.
    Returns None if no matching arXiv paper is found.
    """
    try:
        # Clean the title for search
        search_title = title.strip()
        
        # Limit search query length and remove special characters
        search_title = re.sub(r'[^\w\s]', ' ', search_title)
        search_title = ' '.join(search_title.split()[:15])  # First 15 words
        
        search = arxiv.Search(
            query=f'ti:"{search_title}"',  # Search in title field
            max_results=3,  # Get top 3 to find best match
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        client = arxiv.Client()
        
        # Find best matching result
        best_match = None
        best_score = 0
        
        for result in client.results(search):
            # Simple similarity check: count matching words
            result_words = set(result.title.lower().split())
            query_words = set(title.lower().split())
            match_score = len(result_words & query_words) / len(query_words | result_words)
            
            if match_score > best_score and match_score > 0.5:  # At least 50% match
                best_score = match_score
                best_match = result
        
        if best_match is None:
            return None
        
        return {
            "title": best_match.title.strip(),
            "authors": [a.name for a in best_match.authors],
            "abstract": best_match.summary.strip(),
            "published": str(best_match.published.date()) if best_match.published else None,
            "source": "arxiv",
            "url": best_match.entry_id,
            "match_score": best_score
        }
    
    except Exception as e:
        print(f"[ARXIV SEARCH ERROR] query='{title[:50]}...' → {e}")
        return None


# ==============================
# 5. FETCH ABSTRACTS FROM SCHOLAR
# ==============================
def fetch_from_scholar(title, authors=None):
    """Fallback: fetch paper info from Google Scholar."""
    try:
        # Construct search query
        search_query = title
        if authors and len(authors) > 0:
            # Add first author to improve search accuracy
            first_author = authors[0].split()[-1]  # Get last name
            search_query = f'{title} {first_author}'
        
        search = scholarly.search_pubs(search_query)
        paper = next(search, None)
        
        if paper:
            full = scholarly.fill(paper)
            bib = full.get("bib", {})
            
            # Check if title matches (at least 50% word overlap)
            result_title = bib.get("title", "")
            result_words = set(result_title.lower().split())
            query_words = set(title.lower().split())
            
            if len(result_words & query_words) / max(len(query_words), 1) < 0.3:
                print(f"[SCHOLAR] Title mismatch: '{result_title[:50]}' vs '{title[:50]}'")
                return None
            
            return {
                "title": result_title,
                "authors": bib.get("author", "").split(" and ") if isinstance(bib.get("author"), str) else bib.get("author", []),
                "abstract": bib.get("abstract", ""),
                "year": bib.get("pub_year"),
                "url": full.get("eprint_url") or full.get("pub_url"),
                "source": "scholar"
            }
    except StopIteration:
        print(f"[SCHOLAR] No results found")
    except Exception as e:
        print(f"[SCHOLAR ERROR] {e}")
    
    return None


# ==============================
# 6. MAIN FUNCTION
# ==============================
def collect_abstracts(papers):
    """Main routine to collect abstracts for all given papers."""
    success_count = 0
    fail_count = 0
    
    with open(OUTPUT_FILE, "a", encoding="utf8") as fout:
        for paper in tqdm(papers, desc="Fetching abstracts"):
            title = paper["title"]
            arxiv_id = paper.get("arxiv_id")
            authors = paper.get("authors", [])
            
            print(f"\n{'='*80}")
            print(f"Processing: {title}")
            if arxiv_id:
                print(f"  arXiv ID: {arxiv_id}")
            
            data = None
            
            # Strategy 1: If we have arXiv ID, use it directly (most reliable)
            if arxiv_id:
                print(f"  [1] Trying arXiv by ID: {arxiv_id}")
                data = fetch_from_arxiv_by_id(arxiv_id)
                if data and data.get("abstract"):
                    print(f"  ✓ Found on arXiv by ID")
            
            # Strategy 2: Search arXiv by title
            if not data:
                print(f"  [2] Trying arXiv by title search")
                data = fetch_from_arxiv_by_title(title)
                if data and data.get("abstract"):
                    print(f"  ✓ Found on arXiv by title (match: {data.get('match_score', 0):.2%})")
            
            # Strategy 3: Try Google Scholar
            if not data:
                print(f"  [3] Trying Google Scholar")
                data = fetch_from_scholar(title, authors)
                if data and data.get("abstract"):
                    print(f"  ✓ Found on Google Scholar")
            
            # Save if we got data
            if data and data.get("abstract"):
                # Add original metadata
                data["bib_title"] = paper.get("raw_title", title)
                data["bib_year"] = paper.get("year")
                
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                fout.flush()  # Ensure it's written immediately
                
                print(f"  ✓✓ SAVED: {data['title'][:60]}...")
                print(f"     Abstract length: {len(data['abstract'])} chars")
                success_count += 1
            else:
                print(f"  ✗✗ FAILED: Could not find abstract")
                fail_count += 1
            
            time.sleep(2)  # polite delay
    
    print(f"\n{'='*80}")
    print(f"SUMMARY:")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Total: {len(papers)}")
    print(f"{'='*80}")


# ==============================
# 7. RUN SCRIPT
# ==============================
if __name__ == "__main__":
    print("="*80)
    print("AA LITERATURE ABSTRACT DOWNLOADER")
    print("="*80)
    
    # Option 1: Load from references.bib
    papers = read_bib_entries("references.bib")
    
    if papers:
        print("\nLoaded papers:")
        for i, p in enumerate(papers, 1):
            print(f"  {i}. {p['title']}")
            if p.get('arxiv_id'):
                print(f"     arXiv: {p['arxiv_id']}")
    
    # Option 2 (manual): If no .bib file, define papers directly
    if not papers:
        print("\nNo .bib file found. Using manual paper list...")
        papers = [
            {
                "title": "Archetypal analysis for machine learning and data mining",
                "arxiv_id": None,
                "authors": ["Mørup", "Hansen"],
                "year": "2012"
            },
            {
                "title": "Archetypal Analysis: An Algorithmic Perspective",
                "arxiv_id": None,
                "authors": [],
                "year": ""
            },
        ]
    
    if not papers:
        print("\nNo papers to process!")
    else:
        collect_abstracts(papers)
        print(f"\n✓ Finished! Results saved to {OUTPUT_FILE}")
        
        # Show what was collected
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, 'r', encoding='utf8') as f:
                lines = f.readlines()
            print(f"\nTotal abstracts in file: {len(lines)}")
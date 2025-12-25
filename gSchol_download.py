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
from semanticscholar import SemanticScholar
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

# Configuration
USE_GOOGLE_SCHOLAR = True
USE_SEMANTIC_SCHOLAR = True
DELAY_BETWEEN_REQUESTS = 2
SCHOLAR_TIMEOUT = 20        # Timeout for Scholar queries (seconds) 
SCHOLAR_MAX_RETRIES = 1 

if os.path.exists(OUTPUT_FILE):
    print(f"Appending to existing file: {OUTPUT_FILE}")
else:
    print(f"Creating new file: {OUTPUT_FILE}")

print("\nConfiguration:")
print(f"  Google Scholar: {'ENABLED' if USE_GOOGLE_SCHOLAR else 'DISABLED (unreliable)'}")
print(f"  Semantic Scholar: {'ENABLED' if USE_SEMANTIC_SCHOLAR else 'DISABLED'}")
print(f"  Delay between requests: {DELAY_BETWEEN_REQUESTS}s")

# Try to import Semantic Scholar
SEMANTIC_SCHOLAR_AVAILABLE = False
if USE_SEMANTIC_SCHOLAR:
    try:
        from semanticscholar import SemanticScholar
        SEMANTIC_SCHOLAR_AVAILABLE = True
        print("  ✓ Semantic Scholar API ready")
    except ImportError:
        print("  ⚠️  Semantic Scholar not installed!")
        print("     Install with: pip install semanticscholar")
        USE_SEMANTIC_SCHOLAR = False


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
# 5. FETCH ABSTRACTS FROM SEMANTIC SCHOLAR (RELIABLE!)
# ==============================
def fetch_from_semantic_scholar(title, authors=None):
    """
    Fetch paper from Semantic Scholar API - more reliable than Google Scholar.
    
    Args:
        title: Paper title
        authors: List of authors (optional, for better matching)
    
    Returns:
        dict or None
    """
    if not SEMANTIC_SCHOLAR_AVAILABLE:
        print(f"  [SEMANTIC] Not available - install with: pip install semanticscholar")
        return None
    
    try:
        print(f"  [SEMANTIC] Searching for: '{title[:60]}...'")
        
        # Import here to avoid issues if not installed
        from semanticscholar import SemanticScholar
        
        sch = SemanticScholar(timeout=10)
        
        # Clean title for search
        clean_title = clean_latex_title(title)
        
        # Search with title
        results = sch.search_paper(clean_title, limit=5, fields=['title', 'abstract', 'authors', 'year', 'url', 'externalIds'])
        
        if not results:
            print(f"  [SEMANTIC] No results found")
            return None
        
        # Find best match
        best_match = None
        best_score = 0
        
        for paper in results:
            if not paper or not paper.title:
                continue
            
            # Calculate title similarity
            result_words = set(clean_latex_title(paper.title).lower().split())
            query_words = set(clean_title.lower().split())
            overlap = len(result_words & query_words)
            total = len(query_words)
            match_score = overlap / max(total, 1)
            
            # Also check author match if provided
            if authors and paper.authors:
                paper_authors = [a['name'].lower() for a in paper.authors]
                query_authors = [clean_latex_title(a).lower() for a in authors if a]
                
                author_match = any(
                    any(qa in pa for pa in paper_authors)
                    for qa in query_authors
                )
                if author_match:
                    match_score += 0.1  # Bonus for author match
            
            if match_score > best_score:
                best_score = match_score
                best_match = paper
        
        if best_match is None or best_score < 0.4:
            print(f"  [SEMANTIC] No good match found (best: {best_score:.1%})")
            return None
        
        print(f"  [SEMANTIC] Found: '{best_match.title[:50]}...' (match: {best_score:.1%})")
        
        # Check if abstract exists
        if not best_match.abstract:
            print(f"  [SEMANTIC] Paper found but no abstract available")
            return None
        
        # Extract authors
        author_list = []
        if best_match.authors:
            author_list = [a['name'] for a in best_match.authors if 'name' in a]
        
        # Get arXiv ID if available
        arxiv_id = None
        if best_match.externalIds and 'ArXiv' in best_match.externalIds:
            arxiv_id = best_match.externalIds['ArXiv']
        
        result = {
            "title": best_match.title,
            "authors": author_list,
            "abstract": best_match.abstract,
            "year": best_match.year,
            "url": best_match.url or f"https://www.semanticscholar.org/paper/{best_match.paperId}",
            "arxiv_id": arxiv_id,
            "source": "semantic_scholar",
            "match_score": best_score,
            "semantic_scholar_id": best_match.paperId
        }
        
        print(f"  [SEMANTIC] ✓ Successfully fetched ({len(best_match.abstract)} chars)")
        return result
        
    except ImportError:
        print(f"  [SEMANTIC ERROR] Module not installed! Install with: pip install semanticscholar")
        return None
    except Exception as e:
        print(f"  [SEMANTIC ERROR] {e}")
        return None


# ==============================
# 6. FETCH ABSTRACTS FROM GOOGLE SCHOLAR (WITH TIMEOUT AND RETRY)
# ==============================
import signal
from contextlib import contextmanager

# Define TimeoutException at module level
class TimeoutException(Exception):
    """Exception raised when operation times out."""
    pass

@contextmanager
def time_limit(seconds):
    """Context manager for timeout."""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    # Only works on Unix-like systems
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
    else:
        # Windows - no timeout
        yield


def fetch_from_scholar(title, authors=None, timeout=30, max_retries=2):
    """
    Fallback: fetch paper info from Google Scholar.
    
    Args:
        title: Paper title
        authors: List of authors
        timeout: Timeout in seconds (Unix only)
        max_retries: Number of retry attempts
    
    Returns:
        dict or None
    """
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"  [SCHOLAR] Retry attempt {attempt + 1}/{max_retries}")
                time.sleep(5 * attempt)  # Exponential backoff
            
            # Construct search query (limit to first 10 words to avoid issues)
            clean_title = clean_latex_title(title)
            search_words = clean_title.split()[:10]
            search_query = ' '.join(search_words)
            
            # Add first author's last name for better accuracy
            if authors and len(authors) > 0:
                first_author = authors[0].strip()
                if first_author:
                    last_name = first_author.split()[-1]
                    search_query = f'{search_query} {last_name}'
            
            print(f"  [SCHOLAR] Query: '{search_query[:60]}...'")
            
            # Try with timeout (Unix only)
            try:
                with time_limit(timeout):
                    # Search for papers
                    search = scholarly.search_pubs(search_query)
                    paper = next(search, None)
                    
                    if not paper:
                        print(f"  [SCHOLAR] No results found")
                        return None
                    
                    # Get basic info first (don't fill yet - it's slow)
                    bib = paper.get("bib", {})
                    result_title = bib.get("title", "")
                    
                    # Quick title match check before slow fill operation
                    result_words = set(clean_latex_title(result_title).lower().split())
                    query_words = set(clean_latex_title(title).lower().split())
                    overlap = len(result_words & query_words)
                    total = len(query_words)
                    match_ratio = overlap / max(total, 1)
                    
                    print(f"  [SCHOLAR] Found: '{result_title[:50]}...'")
                    print(f"  [SCHOLAR] Match score: {match_ratio:.1%} ({overlap}/{total} words)")
                    
                    if match_ratio < 0.4:  # Require 40% match
                        print(f"  [SCHOLAR] Match too low, skipping")
                        return None
                    
                    # Now fill the paper (slow operation)
                    print(f"  [SCHOLAR] Fetching full details...")
                    full = scholarly.fill(paper)
                    bib_full = full.get("bib", {})
                    
                    # Extract abstract
                    abstract = (bib_full.get("abstract") or 
                               bib_full.get("abstract_note") or 
                               bib.get("abstract") or "")
                    
                    if not abstract:
                        print(f"  [SCHOLAR] No abstract available")
                        return None
                    
                    # Extract authors
                    author_field = bib_full.get("author", "")
                    if isinstance(author_field, str):
                        authors_list = [a.strip() for a in author_field.split(" and ")]
                    elif isinstance(author_field, list):
                        authors_list = author_field
                    else:
                        authors_list = []
                    
                    result = {
                        "title": result_title,
                        "authors": authors_list,
                        "abstract": abstract.strip(),
                        "year": bib_full.get("pub_year") or bib.get("pub_year"),
                        "url": full.get("eprint_url") or full.get("pub_url") or "",
                        "source": "scholar",
                        "match_score": match_ratio
                    }
                    
                    print(f"  [SCHOLAR] ✓ Successfully fetched ({len(abstract)} chars)")
                    return result
                    
            except TimeoutException:
                print(f"  [SCHOLAR] Timeout after {timeout}s")
                if attempt < max_retries - 1:
                    continue
                return None
            
        except StopIteration:
            print(f"  [SCHOLAR] No results found")
            return None
            
        except Exception as e:
            error_msg = str(e)
            print(f"  [SCHOLAR ERROR] {error_msg[:100]}")
            
            # Check if it's a rate limit error
            if "429" in error_msg or "Too Many Requests" in error_msg or "captcha" in error_msg.lower():
                print(f"  [SCHOLAR] Rate limited! Waiting 30 seconds...")
                if attempt < max_retries - 1:
                    time.sleep(30)
                    continue
            
            # Other errors - don't retry
            return None
    
    print(f"  [SCHOLAR] Failed after {max_retries} attempts")
    return None


# ==============================
# 7. MAIN FUNCTION
# ==============================
def collect_abstracts(papers):
    """Main routine to collect abstracts for all given papers."""
    success_count = 0
    fail_count = 0
    
    with open(OUTPUT_FILE, "a", encoding="utf8") as fout:
        for paper in tqdm(papers, desc="Fetching abstracts"):
            title = paper["title"]
            arxiv_id = paper.get("arxiv_id")
            doi = paper.get("doi")
            authors = paper.get("authors", [])
            
            print(f"\n{'='*80}")
            print(f"Processing: {title}")
            if arxiv_id:
                print(f"  arXiv ID: {arxiv_id}")
            if doi:
                print(f"  DOI: {doi}")
            
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
            
            # Strategy 3: Try Semantic Scholar (more reliable than Google Scholar)
            if not data and USE_SEMANTIC_SCHOLAR:
                print(f"  [3] Trying Semantic Scholar")
                data = fetch_from_semantic_scholar(title, authors)
                if data and data.get("abstract"):
                    print(f"  ✓ Found on Semantic Scholar (match: {data.get('match_score', 0):.1%})")
            
            # Strategy 4: Try Google Scholar (if enabled and Semantic failed)
            if not data and USE_GOOGLE_SCHOLAR:
                print(f"  [4] Trying Google Scholar")
                data = fetch_from_scholar(title, authors, 
                                         timeout=SCHOLAR_TIMEOUT, 
                                         max_retries=SCHOLAR_MAX_RETRIES)
                if data and data.get("abstract"):
                    print(f"  ✓ Found on Google Scholar (match: {data.get('match_score', 0):.1%})")
            
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
            
            time.sleep(DELAY_BETWEEN_REQUESTS)  # polite delay
    
    print(f"\n{'='*80}")
    print(f"SUMMARY:")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Total: {len(papers)}")
    print(f"{'='*80}")


# ==============================
# 8. RUN SCRIPT
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
                "title":"Probabilistic archetypal analysis",
                "author":["Seth", "Sohan and Eugster", "Manuel JA"],
                "journal":"{Machine learning}",
                "volume":"102",
                "number":{1},
                "pages":"{85--113}",
                "year":"2016",
                "publisher":"Springer"
            },
            {
                 "title": "Evolutionary trade-offs, Pareto optimality, and the geometry of phenotype space",
                 "authors": ["Shoval", "Oren and Sheftel", "Hila and Shinar", "Guy and Hart", "Yuval and Ramote", "Omer and Mayo", "Avi and Dekel", "Erez and Kavanagh", "Kathryn and Alon", "Uri"],
                 "journal":"Science",
                 "volume":"336",
                 "number":"6085",
                 "year": "2012",
                 "publisher":"American Association for the Advancement of Science"
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
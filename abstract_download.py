"""
download_openreview_conference.py
----------------------------------
Downloads ALL accepted papers from a specific conference and year.
No filtering - gets complete conference proceedings.

Supports: ICLR, NeurIPS, ICML
Years: 2014-2024 (depending on conference)

Usage:
    python download_openreview_conference.py --conference ICLR --year 2019
    python download_openreview_conference.py --conference NeurIPS --year 2023
"""

import json
import os
import argparse
from datetime import datetime
import openreview
import pandas as pd
from tqdm import tqdm

# ============================================================================
# CONFIGURATION - CHANGE THESE FOR DIFFERENT CONFERENCES/YEARS
# ============================================================================

# Default settings (can be overridden by command line arguments)
DEFAULT_CONFERENCE = 'ICLR'
DEFAULT_YEAR = 2019

OUTPUT_DIR = "data\openreview_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Available conferences and their venue IDs
VENUE_IDS = {
    'ICLR': {
        2024: 'ICLR.cc/2024/Conference',
        2023: 'ICLR.cc/2023/Conference',
        2022: 'ICLR.cc/2022/Conference',
        2021: 'ICLR.cc/2021/Conference',
        2020: 'ICLR.cc/2020/Conference',
        2019: 'ICLR.cc/2019/Conference',
        2018: 'ICLR.cc/2018/Conference',
        2017: 'ICLR.cc/2017/conference',
        2016: 'ICLR.cc/2016/conference',
        2014: 'ICLR.cc/2014/conference',
    },
    'NeurIPS': {
        2023: 'NeurIPS.cc/2023/Conference',
        2022: 'NeurIPS.cc/2022/Conference',
        2021: 'NeurIPS.cc/2021/Conference',
        2020: 'NeurIPS.cc/2020/Conference',
        2019: 'NeurIPS.cc/2019/Conference',
        2018: 'NeurIPS.cc/2018/Conference',
        2017: 'NeurIPS.cc/2017/Conference',
    },
    'ICML': {
        2023: 'ICML.cc/2023/Conference',
        2022: 'ICML.cc/2022/Conference',
        2021: 'ICML.cc/2021/Conference',
        2020: 'ICML.cc/2020/Conference',
        2019: 'ICML.cc/2019/Conference',
        2018: 'ICML.cc/2018/Conference',
        2017: 'ICML.cc/2017/Conference',
    }
}


# ============================================================================
# Helper Functions
# ============================================================================

def get_available_years(conference):
    """Get list of available years for a conference."""
    if conference not in VENUE_IDS:
        return []
    return sorted(VENUE_IDS[conference].keys(), reverse=True)


def print_available_options():
    """Print all available conference/year combinations."""
    print("\n" + "="*80)
    print("AVAILABLE CONFERENCES AND YEARS")
    print("="*80)
    
    for conf in sorted(VENUE_IDS.keys()):
        years = get_available_years(conf)
        print(f"\n{conf}:")
        print(f"  Years: {', '.join(map(str, years))}")
        print(f"  Total: {len(years)} years available")
    
    print("\n" + "="*80)


# ============================================================================
# OpenReview Client Setup
# ============================================================================

def get_openreview_client():
    """Initialize OpenReview client (no authentication needed for public data)."""
    try:
        client = openreview.Client(baseurl='https://api.openreview.net')
        print("✓ OpenReview client initialized")
        return client
    except Exception as e:
        print(f"✗ Error initializing OpenReview client: {e}")
        print("  Make sure you have installed: pip install openreview-py")
        return None


# ============================================================================
# Paper Extraction
# ============================================================================

def extract_paper_info(note):
    """
    Extract relevant information from OpenReview note.
    Handles different field formats across years (dict, list, string, nested dict with 'value').
    
    Args:
        note: OpenReview note object
    
    Returns:
        dict: Paper information
    """
    try:
        content = note.content
        
        # Helper function to extract value from different formats
        def get_value(field_data, default=''):
            if field_data is None:
                return default
            # If it's a dict with 'value' key
            if isinstance(field_data, dict):
                return field_data.get('value', field_data.get('param', {}).get('value', default))
            # If it's already a string or list
            return field_data
        
        # Extract title
        title_data = content.get('title') or content.get('Title') or ''
        title = get_value(title_data)
        
        # Extract abstract
        abstract_data = content.get('abstract') or content.get('Abstract') or ''
        abstract = get_value(abstract_data)
        
        # Extract authors
        authors_data = content.get('authors') or content.get('Authors') or []
        authors = get_value(authors_data)
        
        # Handle different author formats
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(',') if a.strip()]
        elif not isinstance(authors, list):
            authors = []
        
        # Extract keywords
        keywords_data = content.get('keywords') or content.get('Keywords') or []
        keywords = get_value(keywords_data)
        
        # Handle different keyword formats
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(',') if k.strip()]
        elif not isinstance(keywords, list):
            keywords = []
        
        # Validate we have required fields
        if not title or not abstract:
            return None
        
        paper_info = {
            'id': note.id,
            'title': title if isinstance(title, str) else str(title),
            'abstract': abstract if isinstance(abstract, str) else str(abstract),
            'authors': authors,
            'keywords': keywords,
            'venue': get_value(content.get('venue', '')),
            'venueid': get_value(content.get('venueid', '')),
            'url': f'https://openreview.net/forum?id={note.id}',
            'pdf_url': f'https://openreview.net/pdf?id={note.id}',
        }
        
        return paper_info
    
    except Exception as e:
        print(f"  Warning: Error extracting paper {getattr(note, 'id', 'unknown')}: {e}")
        return None


# ============================================================================
# Download Function
# ============================================================================

def download_conference_papers(conference, year, client=None):
    """
    Download ALL accepted papers from a specific conference year.
    
    Args:
        conference: Conference name (e.g., 'ICLR', 'NeurIPS', 'ICML')
        year: Year (e.g., 2019, 2023)
        client: OpenReview client (optional, will create if None)
    
    Returns:
        list: List of paper dictionaries
    """
    # Validate inputs
    if conference not in VENUE_IDS:
        print(f"✗ Error: Conference '{conference}' not supported")
        print(f"  Available: {', '.join(VENUE_IDS.keys())}")
        return []
    
    if year not in VENUE_IDS[conference]:
        print(f"✗ Error: Year {year} not available for {conference}")
        print(f"  Available years: {', '.join(map(str, get_available_years(conference)))}")
        return []
    
    # Get venue ID
    venue_id = VENUE_IDS[conference][year]
    
    # Initialize client if needed
    if client is None:
        client = get_openreview_client()
        if not client:
            return []
    
    print(f"\n{'='*80}")
    print(f"DOWNLOADING {conference} {year}")
    print(f"{'='*80}")
    print(f"Venue ID: {venue_id}")
    
    papers = []
    
    try:
        # Get all accepted papers
        # Different years use different invitation patterns
        invitations_to_try = [
            f'{venue_id}/-/Blind_Submission',
            f'{venue_id}/-/Submission',
            f'{venue_id}/-/Paper',
        ]
        
        notes = []
        for invitation in invitations_to_try:
            try:
                print(f"Trying invitation: {invitation}")
                notes = client.get_all_notes(invitation=invitation)
                if notes:
                    print(f"✓ Found {len(notes)} submissions using {invitation}")
                    break
            except Exception as e:
                print(f"  Failed with {invitation}: {e}")
                continue
        
        if not notes:
            print("✗ Could not retrieve papers with any invitation pattern")
            return []
        
        print(f"\nProcessing {len(notes)} papers...")
        
        # Extract paper information
        for note in tqdm(notes, desc=f"Extracting {conference} {year}"):
            paper_info = extract_paper_info(note)
            
            if not paper_info:
                continue
            
            # Skip if no abstract
            if not paper_info['abstract'] or len(paper_info['abstract'].strip()) < 50:
                continue
            
            # Add metadata
            paper_info['conference'] = conference
            paper_info['year'] = year
            paper_info['download_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            papers.append(paper_info)
        
        print(f"✓ Successfully extracted {len(papers)} papers with abstracts")
        
        return papers
        
    except Exception as e:
        print(f"✗ Error downloading {conference} {year}: {e}")
        import traceback
        traceback.print_exc()
        return []


# ============================================================================
# Save Functions
# ============================================================================

def save_papers(papers, conference, year):
    """
    Save papers to JSONL and CSV files.
    
    Args:
        papers: List of paper dictionaries
        conference: Conference name
        year: Year
    
    Returns:
        dict: Paths to saved files
    """
    if not papers:
        print("No papers to save!")
        return {}
    
    # Create filenames
    base_name = f"{conference}_{year}_papers"
    jsonl_file = os.path.join(OUTPUT_DIR, f"{base_name}.jsonl")
    csv_file = os.path.join(OUTPUT_DIR, f"{base_name}.csv")
    
    # Save JSONL
    try:
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for paper in papers:
                f.write(json.dumps(paper, ensure_ascii=False) + '\n')
        print(f"\n✓ Saved JSONL: {jsonl_file}")
    except Exception as e:
        print(f"✗ Error saving JSONL: {e}")
        jsonl_file = None
    
    # Save CSV
    try:
        df = pd.DataFrame(papers)
        
        # Flatten lists for CSV
        for col in ['authors', 'keywords']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: '; '.join(x) if isinstance(x, list) else x)
        
        # Select relevant columns
        columns = ['conference', 'year', 'title', 'authors', 'abstract', 
                  'keywords', 'url']
        df = df[[c for c in columns if c in df.columns]]
        
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"✓ Saved CSV: {csv_file}")
    except Exception as e:
        print(f"✗ Error saving CSV: {e}")
        csv_file = None
    
    return {
        'jsonl': jsonl_file,
        'csv': csv_file
    }


def print_summary(papers, conference, year):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nConference: {conference} {year}")
    print(f"Total papers: {len(papers)}")
    
    if papers:
        # Average abstract length
        avg_abstract_len = sum(len(p['abstract']) for p in papers) / len(papers)
        print(f"Average abstract length: {avg_abstract_len:.0f} characters")
        
        # Papers with keywords
        with_keywords = sum(1 for p in papers if p.get('keywords'))
        print(f"Papers with keywords: {with_keywords} ({with_keywords/len(papers)*100:.1f}%)")
        
        # Sample titles
        print("\nSample paper titles (first 5):")
        for i, paper in enumerate(papers[:5], 1):
            print(f"  {i}. {paper['title'][:80]}...")
    
    print("="*80)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Download all accepted papers from OpenReview conference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_openreview_conference.py --conference ICLR --year 2019
  python download_openreview_conference.py --conference NeurIPS --year 2023
  python download_openreview_conference.py --conference ICML --year 2019
  python download_openreview_conference.py --list
        """
    )
    
    parser.add_argument('--conference', '-c', type=str, default=DEFAULT_CONFERENCE,
                       help=f'Conference name (ICLR, NeurIPS, ICML). Default: {DEFAULT_CONFERENCE}')
    parser.add_argument('--year', '-y', type=int, default=DEFAULT_YEAR,
                       help=f'Year. Default: {DEFAULT_YEAR}')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List all available conferences and years')
    
    args = parser.parse_args()
    
    # If --list flag, show available options and exit
    if args.list:
        print_available_options()
        return
    
    # Print header
    print("="*80)
    print("OPENREVIEW CONFERENCE PAPER DOWNLOADER")
    print("="*80)
    
    # Check if openreview is installed
    try:
        import openreview
    except ImportError:
        print("\n X OpenReview package not installed!")
        print("Install with: pip install openreview-py")
        return
    
    # Get conference and year
    conference = args.conference.upper()
    year = args.year
    
    print(f"\nRequested: {conference} {year}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Download papers
    client = get_openreview_client()
    if not client:
        return
    
    papers = download_conference_papers(conference, year, client)
    
    if not papers:
        print("\n No papers downloaded!")
        print("\nTroubleshooting:")
        print("  1. Check if conference/year combination is valid")
        print("  2. Run with --list to see available options")
        print("  3. Check your internet connection")
        return
    
    # Save papers
    files = save_papers(papers, conference, year)
    
    # Print summary
    print_summary(papers, conference, year)
    
    # Final message
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE!")
    print("="*80)
    print(f"\nDownloaded {len(papers)} papers from {conference} {year}")
    print(f"\nFiles saved in: {OUTPUT_DIR}/")
    if files.get('jsonl'):
        print(f"  - {os.path.basename(files['jsonl'])} (for processing)")
    if files.get('csv'):
        print(f"  - {os.path.basename(files['csv'])} (for viewing)")
  

if __name__ == "__main__":
    main()
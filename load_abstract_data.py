import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def load_aa_abstracts(jsonl_path, max_docs=None, min_df=2, max_df=0.95):
    texts = []
    titles = []

    with open(jsonl_path, "r", encoding="utf8") as f:
        for line in f:
            paper = json.loads(line)
            if "abstract" in paper and paper["abstract"]:
                texts.append(paper["abstract"])
                titles.append(paper.get("title", "unknown"))

                if max_docs and len(texts) >= max_docs:
                    break

    print(f"Loaded {len(texts)} abstracts from {jsonl_path}")

    vectorizer = TfidfVectorizer(stop_words='english',
                                 min_df=min_df,
                                 max_df=max_df)

    X = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()

    print("TF-IDF Matrix shape:", X.shape)
    print("Vocabulary size:", len(vocab))

    return X.toarray(), vocab, titles

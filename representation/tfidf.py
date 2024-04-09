#!/usr/bin/env python
import numpy as np


def tfidf(docs: list[str], smooth_idf: bool = True, normalize: bool = True) -> np.ndarray:
    """Performs TF-IDF vectorization of input documents."""

    unique_words = {word for doc in docs for word in doc.split()}
    word_to_idx = {word: idx for idx, word in enumerate(unique_words)}

    tf = np.zeros((len(docs), len(unique_words)))
    for doc_idx, doc in enumerate(docs):
        for word in doc.split():
            tf[doc_idx][word_to_idx[word]] += 1

    df = (tf != 0).sum(axis=0)
    idf = np.log(len(docs) / df) + 1 if not smooth_idf else np.log((len(docs) + 1) / (df + 1)) + 1
    tfidf = tf * idf

    if normalize:
        tfidf /= np.linalg.norm(tfidf, axis=1)[:, np.newaxis]

    return tfidf


# Eval {{{

if __name__ == "__main__":
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [
        "sun is shining",
        "sky is blue",
        "sun is shining and sky is blue",
    ]

    tfidf_sklearn = TfidfVectorizer().fit_transform(docs).toarray()

    assert np.allclose(np.sort(tfidf(docs)), np.sort(tfidf_sklearn))

# }}}

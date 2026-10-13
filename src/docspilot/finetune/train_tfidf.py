"""Classical baseline: TF-IDF + logistic regression, exported to JSON.

Every ML project should have this baseline *first*. If it is within a few points of the
fancy model, ship it: microseconds, no GPU, no runtime dependency, trivially auditable."""

from __future__ import annotations

import json
from pathlib import Path

from docspilot.finetune.dataset import Example
from docspilot.local_classifier import TfidfLinear, ngrams, tokenize


def train(train: list[Example], out_dir: Path, *, c: float = 8.0) -> TfidfLinear:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    # Use *our* tokenizer so training and the pure-python inference path agree exactly.
    vec = TfidfVectorizer(
        analyzer=lambda t: ngrams(tokenize(t), 2), sublinear_tf=True, norm="l2", min_df=1
    )
    x = vec.fit_transform([e.text for e in train])
    clf = LogisticRegression(C=c, max_iter=2000)
    clf.fit(x, [e.label for e in train])

    vocab = {str(k): int(v) for k, v in vec.vocabulary_.items()}
    coef = clf.coef_.tolist()
    intercept = clf.intercept_.tolist()
    if len(clf.classes_) == 2:  # binary: sklearn stores one row; expand to two
        coef = [[-v for v in coef[0]], coef[0]]
        intercept = [-intercept[0], intercept[0]]
    model = TfidfLinear(vocab, vec.idf_.tolist(), coef, intercept, [str(c) for c in clf.classes_])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.json").write_text(json.dumps(model.to_json()))
    (out_dir / "meta.json").write_text(json.dumps({"kind": "tfidf", "labels": model.labels}))
    return model

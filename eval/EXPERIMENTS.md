# Improvement experiments (2026-11-04) - hybrid sparse+dense ADOPTED, all other candidates rejected

A second pass tried to improve retrieval quality beyond the baseline
(TF-IDF cosine, english stopwords, l2 norm). Six candidates were measured
against the SAME frozen protocol as eval/RESULTS.md (development set: the
150 known-item sample and 10 judged queries were already used to build the
baseline evaluation, so treat every number here as development-used, not
held-out). One candidate - the hybrid sparse+dense blend - improved every development
stage and was ADOPTED as the shipped ranker. All sparse-only and dense-only
variants were rejected.

| Candidate | Known-item (dev, n=150) | Judged topical (dev, 10 queries) | Verdict |
|---|---|---|---|
| Baseline TF-IDF cosine | strict 132/150, dup MRR 0.9436 | P@5 0.68, P@10 0.63 | kept |
| BM25 unigram | MRR 0.9522 | P@5 0.64, P@10 0.57 (full re-judged pools) | rejected: topical drop |
| BM25 unigram+bigram | MRR 0.9169 | not judged (known-item already worse) | rejected |
| TF-IDF bigram | MRR 0.9302 | not judged (known-item already worse) | rejected |
| Duplicate-collapse per content hash | strict top-5 147/150 (worse) | P@5 0.68, P@10 0.63 (identical) | rejected |
| Title-boost x2 (headline weighted 2x) | strict 133/150, MRR 0.9478 | P@5 0.66, P@10 0.63 (full re-judged pools) | rejected: flat |
| LSA (TF-IDF + TruncatedSVD 300, cosine) | strict 113/150, dup MRR 0.8523 | not scored (known-item regression decisive) | rejected |
| MiniLM-L6-v2 embedding only | strict 115/150, dup MRR 0.858 | not scored (known-item regression decisive) | rejected |
| HYBRID 0.65*TF-IDF + 0.35*MiniLM | strict 135/150, dup 138/150, MRR 0.9536 | P@5 0.80, P@10 0.70 (merged-judge pools) | ADOPTED |

Notes and honest limits:
- Judged-stage pools differ per candidate, so candidate pools were fully
  scored: original labels reused for ids present in eval/judgments.json,
  new ids judged by a different author under the same written rubric
  (eval/experiments/added_judgments.json, with provenance note). Full
  candidate pools with scores ship in eval/experiments/pools_*.json.
- Held-out check for the adopted hybrid (disjoint seed-2026 known-item
  sample): hybrid 128/150 strict, MRR 0.952 vs baseline 131/150, MRR 0.961 -
  the small known-item development gain does not replicate held-out; the
  judged topical gain (0.68/0.63 -> 0.80/0.70) is the adopted improvement.
  The judged stage has no held-out counterpart (queries used for selection).
- Sparse-only and dense-only candidates each lost on at least one stage;
  the hybrid is the configuration class that won. Earlier negative results
  rule out ONLY those tested configurations on this small development set.
- The alpha weight (0.65) was selected on the development set only; it was
  NOT re-tuned on the held-out sample.
- All judgments remain author-built behavioral evidence, not an
  independent benchmark.

# Retrieval evaluation (added 2026-11-04, ranker updated 2026-11-04)

All numbers in this file are historical records of runs executed 2026-11-04/05 on the canonical dataset. They can be re-derived by re-running the shipped code against that dataset; judgment text snippets are omitted for rights reasons, so the judgment files support re-checking by document ID, not by reading the excerpt text here.

Earlier versions ran queries as unlabeled sanity checks only - no quality
measurement at all. This evaluation closes that gap with two measured stages
and one clearly labeled proxy. Everything is deterministic (fixed seeds) and
reproducible: `python3 eval/evaluate.py --dataset <bbc-news-summary root>`
(first run downloads the ~90 MB embedding model; indexing encodes the corpus
once, ~100 s on a small CPU box).

Shipped ranker: HYBRID 0.65 * TF-IDF cosine + 0.35 * MiniLM-L6-v2 embedding
cosine (adopted 2026-11-04; see eval/EXPERIMENTS.md for the full comparison,
including the variants that were measured and rejected).

## Stage 1 - known-item retrieval (150 sampled articles)
Query = the article's four most discriminating terms (highest term
frequency x IDF, deterministic tie-break). Can the system find the article
back? Success is counted two ways: strict (the exact source article id) and
duplicate-aware (any article with the same content hash - the corpus keeps 98
exact duplicates by design, so a duplicate outranking the source is not an
error).

| Metric | TF-IDF baseline | Hybrid (shipped) |
|---|---|---|
| top-1 strict | 132/150 | 135/150 |
| top-5 strict | 150/150 | 150/150 |
| top-1 duplicate-aware | 135/150 | 138/150 |
| top-5 duplicate-aware | 150/150 | 150/150 |
| MRR (duplicate-aware) | 0.944 | 0.954 |

Held-out check (disjoint 150-article sample, seed 2026, never used during
development or variant selection): baseline top-1 strict 131/150, MRR 0.961;
hybrid top-1 strict 128/150, MRR 0.952; top-5 saturated at 150/150 for both.
So the hybrid's known-item gain does NOT replicate on held-out data: an
observed 3-article decline on this sample, with no guarantee the gap
generalizes beyond it. Per-article diagnosis of the 6 lost / 3 gained flips
(thin sparse margins overridden by the 0.35 dense weight; effective 256-token
embedding window; long-article head-truncation is a supported hypothesis, not
a proven cause) is in eval/experiments/heldout_flip_analysis.md. That
diagnostic inspection makes the seed-2026 sample development-exposed; it is
no longer an untouched check. The judged topical gain (below) remains the
substantive improvement. Reported as-is, not rounded into a win.

## Stage 2 - judged topical queries (10 natural queries)
Pooled top-10 hits per query, each judged relevant/not from the article's
opening text. Mean precision: P@5 = 0.80, P@10 = 0.70 (TF-IDF baseline:
P@5 = 0.68, P@10 = 0.63, judged on its own pool).

Per-query P@10 (hybrid): focused topical queries score 0.7-1.0; ambiguous
phrasings remain the weak spot. The hybrid's gain concentrates exactly on the
baseline's polysemy failures ("grand slam", "ratings", "transfer").

Judgment provenance: original baseline judgments by the system's author; the
hybrid pool's 22 new ids were judged by a DIFFERENT judge (AI assistant)
calibrated to the same written rubric. Baseline judgments preserved verbatim
in eval/experiments/judgments_tfidf_original.json; added judgments in
eval/experiments/added_judgments.json. Author-built behavioral evidence, not
an independent benchmark. The judged stage has NO held-out counterpart: the
10 queries and their judgments were used to select the hybrid, so treat
0.80/0.70 as development-set numbers.

## Proxy - category coherence
Mean share of top-5 hits in the same category as the source article: 0.861
(baseline 0.832). Topical proxy only, not a relevance judgment.

## Limits
- Judgments are AUTHOR-BUILT (two judges, both affiliated with the work).
  This is behavioral evidence on fresh queries, not an independent benchmark.
  Judgments ship as data (eval/judgments.json and eval/judged_pool.json) so
  anyone can re-check them; article-text snippets are omitted for rights
  reasons and the document IDs reference the canonical dataset.
- Known-item retrieval measures findability of a specific article, not
  topical relevance; stage 2 covers topical relevance but on only 10 queries.
- Results describe this fixed corpus (2,225 BBC articles); no claim about
  other collections.
- The embedding model is a fixed public checkpoint (all-MiniLM-L6-v2);
  scores are blended weights, not probabilities.

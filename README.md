# BBC News Article Indexing and Retrieval

Inverted-index search over the BBC news corpus (2,225 articles, 5 categories)
with two selectable rankers, deterministic indexing, and a documented
held-out retrieval evaluation. The repo pairs a preserved original coursework
notebook (source-only) with a standalone, tested retrieval module built later.

## Contents

- `phase1_source_only.ipynb` - original DSAI201 Phase 1 coursework notebook,
  source-only (historical outputs stripped to avoid redistributing article
  excerpts). Kept unchanged as the starting point.
- `news_index.py` - standalone retrieval module: corpus download and
  verification, deterministic inverted index, two rankers, CLI.
- `test_news_index.py` / `tests.log` - 10 unit tests on synthetic fixtures.
- `eval/` - retrieval evaluation: `RESULTS.md`, `EXPERIMENTS.md`, and the
  judgment sets and pools for independent re-checking.
- `run_report.json`, `cli-query.log`, `verification.json`,
  `baseline_status.json` - records of the executed runs.

## Results

- Corpus: business 510 / entertainment 386 / politics 417 / sport 511 /
  tech 401. Inverted vocabulary 27,906 terms; TF-IDF matrix 2,225 x 27,609
  after English stopword removal. 2,127 unique content hashes: 98 exact-content
  duplicates are retained as distinct source IDs and reported, so search hits
  can repeat content (they are not independent evidence).
- Retrieval quality (details in `eval/RESULTS.md`): known-item top-5 150/150
  (MRR 0.954); judged topical queries P@5 0.80 / P@10 0.70. Both are
  development-set numbers with an honest held-out known-item check reported.
  The judgments are author-built (two judges, originals preserved) and shipped
  for re-checking - they are not an independent benchmark.

## Methods

- Correct module run order (the original notebook calls `preprocess` before
  its defining cell; running its cells 3-4 as-is fails with
  `NameError: preprocess not defined`). The original is retained unchanged.
- Article-only root discovery: the Kaggle download contains two duplicate
  trees (2,225 articles + 2,225 summaries each). Relative paths and SHA-256
  are compared for every article across roots; disagreement fails the run.
- Deterministic category/filename document IDs (bare basenames collide
  across topics), explicit per-file encoding (2,224 UTF-8, 1 cp1252), and
  deterministic posting order and tie-breaking.
- Two rankers: `search(rows)` defaults to a hybrid (0.65 TF-IDF cosine +
  0.35 MiniLM-L6-v2 embedding cosine; adopted 2026-11-04, see
  `eval/EXPERIMENTS.md`) intended for topical queries; `mode='tfidf'` keeps
  the sparse baseline for exact known-item lookup. Empty/OOV queries return
  no hits in both modes. First hybrid run downloads the ~90 MB MiniLM model
  from Hugging Face (cached afterwards).

## Data source and rights

- Dataset: `pariza/bbc-news-summary` on Kaggle (version 2, downloaded
  2026-11-04 via kagglehub). Underlying corpus: BBC News Summary, UCD
  (http://mlg.ucd.ie/datasets/bbc.html), which states non-commercial research
  use; the BBC owns the original article copyright.
- No article text, summaries, index postings, or trained index state are
  redistributed in this repo. The code downloads the data from its canonical
  source; generated local output may contain copyrighted terms and should not
  be redistributed without clearance.

## Reproduce

Python 3.10 verified. `requirements.lock.txt` is the exact tested environment
(conservative, not minimal). CPU-only; no GPU, Kaggle auth, or paid API.

    python -m venv .venv
    .venv/bin/pip install -r requirements.lock.txt
    .venv/bin/python test_news_index.py
    .venv/bin/python news_index.py --download --output local-output
    .venv/bin/python news_index.py --download --query "football world cup"

Or pass a local dataset root as the positional argument instead of
`--download`. The full pipeline was executed end-to-end twice with
byte-identical postings and run report (see `verification.json`).

## Attribution

- The Phase 1 coursework notebook is preserved as recovered; its team/solo
  context is not recorded, so no sole-authorship claim is made for it.
- The standalone module, tests, and evaluation harness are later portfolio
  work, built with AI-assistant help and presented as such - not as unaided
  coursework.

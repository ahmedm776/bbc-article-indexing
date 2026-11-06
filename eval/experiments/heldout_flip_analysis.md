# Held-out known-item flip analysis (added 2026-11-05, documentation only - ranker unchanged)

The held-out known-item check (disjoint seed-2026 sample, n=150) reported in
RESULTS.md showed hybrid 128/150 strict top-1 vs baseline 131/150: an observed
3-article decline on this sample, with no guarantee the gap generalizes beyond
it. This note records the per-article diagnosis. The shipped package previously
stored only aggregate held-out numbers; the per-article outcomes here were
regenerated with the exact frozen protocol (kagglehub pariza/bbc-news-summary
v2; locked env torch 2.14.1+cpu / sentence-transformers 6.2.0 / numpy 1.26.4 /
scikit-learn 1.7.2) and all four published aggregates reproduced exactly
(dev 132/135, held-out 131/128), so the regeneration is the same run.
Per-article outcomes: eval/experiments/heldout_known_item_per_article.json.

## Measured flips (held-out top-1 strict)

- Hybrid LOST top-1 on 6 articles: tech/150, tech/362, tech/379, politics/406,
  politics/293, sport/404.
- Hybrid GAINED top-1 on 3 articles: tech/182, business/365, business/024.
- No flipped pair is a same-content duplicate pair, so duplicate ordering is
  not a factor (duplicate-aware top-1: baseline 140, hybrid 137 - the same
  3-article gap).

## Shared mechanism across all 6 losses

In every loss, the source held the HIGHER sparse score but by a thin margin
(0.001-0.028), while the outranking article held higher dense similarity
(+0.005 to +0.140); the 0.35 dense weight overrode the thin sparse margin.
Two contributing factors:

1. Query construction: top-4 tf*idf terms produce near-tied sparse scores
   among top candidates (the synthetic-query bias already documented as W5),
   so on this stage the dense branch acts as a tie-breaker.
2. Effective embedding window: all-MiniLM-L6-v2 truncates at 256 word pieces
   (its max_seq_length). That token limit is the ACTUAL effective window;
   EMB_TEXT_CHARS=2000 is only an outer char cap, and any char count is an
   approximation of the 256-token window, not the mechanism itself.

All 6 lost sources are long articles (2,033-13,829 chars; 69 of the 150
held-out articles exceed 2,000 chars, median 1,934) and the winners were
mostly shorter. The hypothesis that head-only embedding hurts long articles
whose discriminating terms sit deeper in the text is SUPPORTED by this
correlation but NOT proven: all losses being long does not establish that
truncation caused them, and no controlled comparison was run.

## Status and discipline

- No ranking behavior changed; the frozen shipped ranker is unchanged.
- Inspecting these held-out outcomes for diagnosis makes the seed-2026 sample
  development-exposed; it is no longer an untouched check for future changes.
- No fix is proposed: changes targeting these 6 articles (length
  normalization, windowed encoding, reweighting) would be score optimization
  on an inspected sample, not defect repair. No defect was found: the ranker
  behaves as documented.

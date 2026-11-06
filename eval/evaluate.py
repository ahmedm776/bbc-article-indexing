"""Retrieval evaluation for the BBC article index. Reproduces eval/RESULTS.md.

Two measured stages plus one clearly-labeled proxy:
1. Known-item retrieval: 150 deterministically sampled articles; query = the
   article's top-4 highest-IDF terms. Success = source article (or an exact
   same-content duplicate, which the corpus intentionally retains) in top-k.
2. Judged topical queries: 10 natural queries, pooled top-10 hits, relevance
   judged from article opening text. Judgments are AUTHOR-BUILT (same person
   as the system) and ship as eval/judgments.json so anyone can re-check them
   against eval/judged_pool.json. This is behavioral evidence, not an
   independent benchmark.
3. Category coherence: fraction of top-5 hits sharing the source article's
   category. A topical proxy, NOT a relevance judgment.

Usage: python3 eval/evaluate.py --dataset /path/to/bbc-news-summary
Dataset: kagglehub handle pariza/bbc-news-summary (version 2). No Kaggle auth.
"""
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from news_index import article_manifest, Search, tokens

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    args = ap.parse_args()
    rows, _ = article_manifest(args.dataset)
    search = Search(rows)
    here = Path(__file__).resolve().parent

    from collections import Counter
    vocab = search.vectorizer.get_feature_names_out()
    idf = dict(zip(vocab, search.vectorizer.idf_))
    def make_query(row):
        # most discriminating terms: tf within this doc x global idf,
        # deterministic (-score, term) tie-break
        tf = Counter(tokens(row['text']))
        cand = {t: tf[t] * idf[t] for t in tf if len(t) > 3 and t in idf}
        return ' '.join(sorted(cand, key=lambda t: (-cand[t], t))[:4])

    rng = random.Random(42)
    sample = rng.sample(rows, 150)
    by_hash = {}
    for r in rows:
        by_hash.setdefault(r['sha256'], set()).add(r['id'])
    s1 = s5 = d1 = d5 = 0; rr = []; cat5 = []
    for row in sample:
        hits = search.query(make_query(row), k=10)
        ids = [h['id'] for h in hits]
        rs = ids.index(row['id']) + 1 if row['id'] in ids else None
        same = by_hash[row['sha256']]
        rd = next((i + 1 for i, d in enumerate(ids) if d in same), None)
        s1 += rs == 1; s5 += bool(rs and rs <= 5)
        d1 += rd == 1; d5 += bool(rd and rd <= 5)
        rr.append(1.0 / rd if rd else 0.0)
        cat5.append(sum(1 for h in hits[:5] if h['id'].split('/')[0] == row['category']) / 5)
    known = {'n': 150, 'top1_strict': s1, 'top5_strict': s5,
             'top1_duplicate_aware': d1, 'top5_duplicate_aware': d5,
             'MRR_duplicate_aware': round(sum(rr) / len(rr), 4),
             'category_coherence_p5_mean': round(sum(cat5) / len(cat5), 4)}

    # judged stage: verify the shipped pool still matches live ranking, then score
    pool = json.load(open(here / 'judged_pool.json'))
    judgments = json.load(open(here / 'judgments.json'))
    judgments = {k: v for k, v in judgments.items() if not k.startswith('_')}
    mismatches = []
    p5 = []; p10 = []
    for q, judged in judgments.items():
        live_ids = [h['id'] for h in search.query(q, k=10)]
        if live_ids != [h['id'] for h in pool[q]]:
            mismatches.append(q)
        vals = [judged.get(i, 0) for i in live_ids]
        p10.append(sum(vals) / len(vals))
        p5.append(sum(vals[:5]) / 5)
    judged_res = {'queries': len(judgments), 'pool_mismatch_queries': mismatches,
                  'P@5_mean': round(sum(p5) / len(p5), 3),
                  'P@10_mean': round(sum(p10) / len(p10), 3)}
    out = {'known_item': known, 'judged': judged_res}
    print(json.dumps(out, indent=2))
    return out

if __name__ == '__main__':
    main()

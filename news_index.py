"""AI-assisted post-coursework extension: article-only index and TF-IDF search.
Original Phase1 notebook is preserved unchanged. No retrieval-quality claim.
"""
import argparse,hashlib,json,re
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def read_text(path):
    b=Path(path).read_bytes()
    try:return b.decode('utf-8'),'utf-8'
    except UnicodeDecodeError:return b.decode('cp1252'),'cp1252'

def article_manifest(dataset):
    root=Path(dataset)
    candidates=sorted([root] if root.name=='News Articles' else [p for p in root.rglob('News Articles') if p.is_dir()])
    if not candidates:raise ValueError('No News Articles directory found; summaries are never substituted')
    manifests=[]
    for candidate in candidates:
        files=sorted(candidate.rglob('*.txt'))
        if not files:raise ValueError('Empty article root')
        manifests.append({str(p.relative_to(candidate)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    if any(m!=manifests[0] for m in manifests[1:]):raise ValueError('Multiple article roots disagree; explicit source resolution required')
    chosen=candidates[0];rows=[]
    for relative,digest in manifests[0].items():
        if len(Path(relative).parts)!=2:raise ValueError('Expected category/filename article path')
        text,enc=read_text(chosen/relative)
        if not text.strip():raise ValueError('Empty article: '+relative)
        rows.append({'id':relative,'category':Path(relative).parts[0],'sha256':digest,'encoding':enc,'text':text})
    return rows,{'article_roots':len(candidates),'duplicate_roots_verified_equal':len(candidates)>1,'chosen_root_relative':str(chosen.relative_to(root)) if chosen!=root else '.', 'all_txt_files':len(list(root.rglob('*.txt')))}

def tokens(text):return re.findall(r'[a-z]+',text.lower())

def build_index(rows):
    ids=[r['id'] for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate document IDs')
    postings=defaultdict(dict)
    for row in rows:
        for term,n in Counter(tokens(row['text'])).items():postings[term][row['id']]=n
    return {t:dict(sorted(p.items())) for t,p in sorted(postings.items())}

EMBEDDING_MODEL='all-MiniLM-L6-v2'
HYBRID_ALPHA=0.65
EMB_TEXT_CHARS=2000  # outer char cap only: the model's own max_seq_length (256 word pieces
# for all-MiniLM-L6-v2) is the actual effective window; chars beyond it are not encoded.
DENSE_MIN_SIM=0.20  # development-selected heuristic (see note in Search docstring): dense scores are
# uncalibrated and do NOT reliably separate nonsense from relevant OOV queries; this floor only
# rejects near-zero-signal inputs, it is not a relevance probability and not an abstention guarantee.

ENGLISH_STOP = None  # lazily imported sklearn stopword list (same set the vectorizer uses)

class PostingsTfidf:
    """Sparse TF-IDF cosine ranker served from the custom postings index
    (build_index output). Scoring replicates the sklearn TfidfVectorizer
    configuration used elsewhere in this project exactly:
    raw term frequency, smooth idf ln((1+N)/(1+df))+1, l2-normalized
    documents and queries, cosine similarity, English stopwords removed.
    test_postings_equivalence asserts score equality with the sklearn path,
    so swapping the serving path changes no ranking or metric."""
    def __init__(self, rows, postings):
        global ENGLISH_STOP
        if ENGLISH_STOP is None:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            ENGLISH_STOP = frozenset(ENGLISH_STOP_WORDS)
        self.ids = [r['id'] for r in rows]
        self.pos = {i: r['id'] for i, r in enumerate(rows)}
        self.idx = {r['id']: i for i, r in enumerate(rows)}
        self.n = len(rows)
        self.postings = {t: p for t, p in postings.items() if t not in ENGLISH_STOP}
        self.idf = {t: np.log((1 + self.n) / (1 + len(p))) + 1.0
                    for t, p in self.postings.items()}
        # per-document l2 norms of the tf-idf vector
        acc = defaultdict(float)
        for t, p in self.postings.items():
            w = self.idf[t]
            for d, tf in p.items():
                acc[d] += (tf * w) ** 2
        self.dnorm = {d: float(np.sqrt(v)) if v > 0 else 1.0 for d, v in acc.items()}

    def query_scores(self, text):
        """Cosine score of `text` against every document (numpy array)."""
        qtf = Counter(t for t in tokens(text) if t in self.postings)
        if not qtf:
            return None  # no in-vocabulary terms: sparse contract is "no signal"
        qn = float(np.sqrt(sum((n * self.idf[t]) ** 2 for t, n in qtf.items())))
        scores = np.zeros(self.n)
        for t, n in qtf.items():
            w = (n * self.idf[t]) / qn
            for d, tf in self.postings[t].items():
                scores[self.idx[d]] += w * (tf * self.idf[t]) / self.dnorm[d]
        return scores


class Search:
    """Retrieval with two selectable rankers.
    mode='hybrid' (default): 0.65 * TF-IDF cosine + 0.35 * MiniLM embedding
      cosine. Best judged topical relevance (dev P@5 0.80 vs 0.68). Intended
      for natural topical queries.
    mode='tfidf': the original sparse baseline. Slightly better strict
      known-item findability on held-out data (131 vs 128 top-1); no model
      download, faster startup. Intended for exact known-item lookup.
    Adopted 2026-11-04 after measured comparison on the frozen eval protocol
    (eval/EXPERIMENTS.md): sparse-only, dense-only, and other sparse variants
    each lost on at least one stage; the hybrid improved all measured stages.
    The embedding model downloads from Hugging Face on first use (~90 MB) and
    is cached under ~/.cache/huggingface. Scores are blended weights, NOT
    probabilities."""
    def __init__(self,rows,mode='hybrid'):
        if mode not in ('hybrid','tfidf'):raise ValueError("mode must be 'hybrid' or 'tfidf'")
        self.mode=mode
        self.rows=rows
        self.postings=build_index(rows)
        self.sparse=PostingsTfidf(rows,self.postings)
        self.vectorizer=TfidfVectorizer(tokenizer=tokens,token_pattern=None,lowercase=False,stop_words='english',norm='l2')
        self.matrix=self.vectorizer.fit_transform(r['text'] for r in rows)
        if self.mode=='hybrid':
            from sentence_transformers import SentenceTransformer
            self.embedder=SentenceTransformer(EMBEDDING_MODEL)
            self.embeddings=self.embedder.encode([r['text'][:EMB_TEXT_CHARS] for r in rows],batch_size=32,normalize_embeddings=True)
    def query(self,text,k=5):
        """Contract: empty/whitespace queries return [] in both modes. tfidf mode is the
        conservative no-hit mode: any fully-OOV query returns []. hybrid mode returns
        dense candidates for OOV queries, each flagged low_confidence=True with a warning;
        these are exploratory candidates, not confident answers (see calibration note)."""
        if k<1:raise ValueError('k must be positive')
        if not text.strip():return []  # empty/whitespace query contract: no hits, both modes
        sparse=self.sparse.query_scores(text)
        if self.mode=='tfidf':
            # Sparse contract preserved: no in-vocabulary terms -> no hits.
            if sparse is None:return []
            scores=sparse
        else:
            qe=self.embedder.encode([text],normalize_embeddings=True)
            dense=(self.embeddings@qe.T).ravel()
            if sparse is None:
                # Lexically OOV (synonyms, rare forms): return dense candidates as
                # EXPLORATORY search candidates only, flagged low_confidence in the output.
                # Real-corpus calibration (eval/experiments/oov_abstention_calibration.json,
                # development-used) shows dense scores overlap between nonsense and relevant
                # OOV queries, so DENSE_MIN_SIM is NOT a validated relevance guard and this
                # path is NOT reliable no-hit detection. Scores are not probabilities.
                # Floor applies to the RAW dense similarity (DENSE_MIN_SIM keeps its meaning);
                # the returned score is scaled by (1-HYBRID_ALPHA) so OOV candidates carry the
                # same dense weight as in the hybrid arm - no 1.0-vs-0.35 discontinuity.
                order=sorted(range(len(self.rows)),key=lambda i:(-dense[i],self.rows[i]['id']))
                return [{'id':self.rows[i]['id'],'score':float((1-HYBRID_ALPHA)*dense[i]),
                         'raw_dense_score':float(dense[i]),
                         'low_confidence':True,
                         'warning':'out-of-vocabulary query; exploratory dense candidate, uncalibrated score (raw floor '+str(DENSE_MIN_SIM)+' applied pre-scaling), not a validated relevance signal'}
                        for i in order[:k] if dense[i]>=DENSE_MIN_SIM]
            scores=HYBRID_ALPHA*sparse+(1-HYBRID_ALPHA)*dense
        order=sorted(range(len(self.rows)),key=lambda i:(-scores[i],self.rows[i]['id']))
        return [{'id':self.rows[i]['id'],'score':float(scores[i])} for i in order[:k] if scores[i]>0]

def run(dataset,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows,provenance=article_manifest(dataset);search=Search(rows);index=search.postings
    manifest=[{k:v for k,v in r.items() if k!='text'} for r in rows]
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    # Runtime-only local artifacts may contain copyrighted terms; do not redistribute.
    export_index={t:p for t,p in index.items() if t not in ENGLISH_STOP}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'postings.json').write_text(json.dumps(export_index))
    queries=['football world cup','bank interest rates','election parliament','mobile phone technology','film music awards','xyzzynonwordqq']
    checks={q:search.query(q,3) for q in queries}
    report={**provenance,'duplicate_content_policy':'Keep distinct article IDs, report exact content duplicates rather than silently delete. Duplicate content is not independent evidence and can repeat in search hits.','articles':len(rows),'unique_ids':len(set(r['id'] for r in rows)),'category_counts':dict(sorted(Counter(r['category'] for r in rows).items())),'encoding_counts':dict(Counter(r['encoding'] for r in rows)),'unique_content_hashes':len(set(r['sha256'] for r in rows)),'vocabulary_terms':len(index),'exported_postings_terms':len(export_index),'postings_export_note':'stopwords filtered at export to match engine vocabulary','tfidf_shape':list(search.matrix.shape),'manifest_sha256':digest,'query_sanity':checks,'evaluation':'Unlabeled development sanity checks only, no relevance judgments/held-out retrieval metric or accuracy claim. Queries not scored against categories.'}
    (out/'run_report.json').write_text(json.dumps(report,indent=2));return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',nargs='?');p.add_argument('--download',action='store_true');p.add_argument('--output',default='local-output');p.add_argument('--query');a=p.parse_args()
    if a.download:
        import kagglehub
        dataset=kagglehub.dataset_download('pariza/bbc-news-summary')
    elif a.dataset:dataset=a.dataset
    else:p.error('Give local dataset root or --download')
    report=run(dataset,a.output)
    if a.query:
        rows,_=article_manifest(dataset);print(json.dumps(Search(rows).query(a.query),indent=2))
    else:print(json.dumps(report,indent=2))

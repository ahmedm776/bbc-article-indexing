import tempfile,unittest
from pathlib import Path
from news_index import *
class Tests(unittest.TestCase):
 def root(self,d):
  root=Path(d)/'News Articles';(root/'a').mkdir(parents=True);(root/'b').mkdir()
  (root/'a/001.txt').write_text('alpha alpha beta');(root/'b/001.txt').write_text('gamma delta');return root
 def test_unique_category_ids(self):
  with tempfile.TemporaryDirectory() as d:
   self.root(d);rows,_=article_manifest(d);self.assertEqual([r['id'] for r in rows],['a/001.txt','b/001.txt'])
 def test_summary_exclusion(self):
  with tempfile.TemporaryDirectory() as d:
   self.root(d);p=Path(d)/'Summaries';p.mkdir();(p/'s.txt').write_text('forbidden_summary');rows,_=article_manifest(d);self.assertEqual(len(rows),2);self.assertFalse(any('forbidden' in r['text'] for r in rows))
 def test_disagreeing_roots(self):
  with tempfile.TemporaryDirectory() as d:
   self.root(d);other=Path(d)/'nested';other.mkdir();self.root(other);(other/'News Articles/a/001.txt').write_text('different')
   with self.assertRaises(ValueError):article_manifest(d)
 def test_equal_roots_not_doublecounted(self):
  with tempfile.TemporaryDirectory() as d:
   self.root(d);other=Path(d)/'nested';other.mkdir();self.root(other);rows,p=article_manifest(d);self.assertEqual(len(rows),2);self.assertTrue(p['duplicate_roots_verified_equal'])
 def test_empty(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):article_manifest(d)
 def test_frequencies(self):self.assertEqual(build_index([{'id':'a','text':'Alpha alpha beta'}])['alpha']['a'],2)
 def test_duplicate_id_rejected(self):
  with self.assertRaises(ValueError):build_index([{'id':'a','text':'one'},{'id':'a','text':'two'}])
 def test_search(self):
  s=Search([{'id':'a','text':'apple fruit sweet'},{'id':'b','text':'train engine rail'}]);self.assertEqual(s.query('fruit')[0]['id'],'a');self.assertEqual(s.query('xyzzqqq'),[])
 def test_determinism(self):
  with tempfile.TemporaryDirectory() as d:
   self.root(d);a,_=article_manifest(d);b,_=article_manifest(d);self.assertEqual(a,b);self.assertEqual(build_index(a),build_index(b))
 def test_tfidf_mode(self):
  s=Search([{'id':'a','text':'apple fruit sweet'},{'id':'b','text':'train engine rail'}],mode='tfidf');self.assertEqual(s.query('fruit')[0]['id'],'a');self.assertRaises(ValueError,Search,[],mode='bogus')
 def test_tie_id_order(self):
  s=Search([{'id':'z','text':'apple fruit'},{'id':'a','text':'apple fruit'}]);self.assertEqual([r['id'] for r in s.query('apple')],['a','z'])

FIX_DOCS=[{'id':'pol','text':'the senator won the election after a long campaign debate'},
 {'id':'spo','text':'the striker scored a late goal for united in the cup final'},
 {'id':'eco','text':'central bank raises interest rates to curb inflation pressures'},
 {'id':'tec','text':'new mobile phone chipset doubles battery life for smartphones'},
 {'id':'ent','text':'the film soundtrack won an award for best original music'}]

class TestPostingsEquivalence(unittest.TestCase):
 def test_scores_match_sklearn(self):
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.metrics.pairwise import cosine_similarity
  import numpy as np
  s=Search(FIX_DOCS,mode='tfidf')
  for q in ['election campaign','mobile battery','the bank','senator striker','united']:
   ref=cosine_similarity(s.vectorizer.transform([q]),s.matrix).ravel()
   got=s.sparse.query_scores(q)
   self.assertIsNotNone(got)
   np.testing.assert_allclose(got,ref,atol=1e-9,err_msg=q)
 def test_oov_returns_none(self):
  s=Search(FIX_DOCS,mode='tfidf')
  self.assertIsNone(s.sparse.query_scores('xyzzqqq'))
  self.assertEqual(s.query('xyzzqqq'),[])
 def test_stopword_only_query(self):
  s=Search(FIX_DOCS,mode='tfidf')
  self.assertEqual(s.query('the and of'),[])

class TestOovSemanticPath(unittest.TestCase):
 def test_empty_query_contract(self):
  for mode in ('tfidf','hybrid'):
   s=Search(FIX_DOCS,mode=mode)
   self.assertEqual(s.query(''),[]);self.assertEqual(s.query('   '),[])
 def test_gibberish_abstains(self):
  s=Search(FIX_DOCS,mode='hybrid')
  self.assertEqual(s.query('xyzzynonwordqq'),[])
  self.assertEqual(s.query('asdfqwer zzz'),[])
  # NOTE: fixture abstention is a small-corpus behavior check, NOT deterministic no-hit
  # proof; real-corpus reliability is documented in eval/experiments/oov_abstention_calibration.json
 def test_semantic_oov_answers(self):
  s=Search(FIX_DOCS,mode='hybrid')
  hits=s.query('voting')  # 'voting' absent from all docs; dense arm should answer
  self.assertTrue(hits);self.assertEqual(hits[0]['id'],'pol')
  self.assertGreaterEqual(hits[0]['raw_dense_score'],DENSE_MIN_SIM)  # floor on raw sim
  self.assertAlmostEqual(hits[0]['score'],(1-HYBRID_ALPHA)*hits[0]['raw_dense_score'],places=9)
  self.assertTrue(all(h.get('low_confidence') for h in hits))
  self.assertTrue(all('warning' in h for h in hits))
 def test_tfidf_mode_conservative_nohit(self):
  s=Search(FIX_DOCS,mode='tfidf')
  self.assertEqual(s.query('voting'),[])  # OOV: conservative no-hit in sparse mode
  self.assertEqual(s.query('xyzzynonwordqq'),[])

if __name__=='__main__':unittest.main()

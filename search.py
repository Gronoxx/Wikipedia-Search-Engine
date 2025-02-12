import re
import numpy as np
import time
import ssl
from collections import defaultdict
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')

class SearchEngine:
    def __init__(self, dataset, max_articles=1000):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.articles = []
        self.inverted_index = defaultdict(dict)
        self.title_token_sets = []
        self.title_magnitudes = []
        self.avg_doc_len = 0
        self.doc_lengths = []
        
        self._load_data(dataset, max_articles)
        self._precompute_statistics()

    def _preprocess_text(self, text):
        #For article content: remove stopwords + stem
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [self.stemmer.stem(t) for t in tokens if t not in self.stop_words]

    def _tokenize_title(self, text):
        #Original manual tokenization logic
        return text.lower().split()

    def _load_data(self, dataset, max_articles):
        for idx, example in enumerate(dataset['train']):
            if idx >= max_articles:
                break

            text_terms = self._preprocess_text(example['text'])
            doc_length = len(text_terms)
            text_str = ' '.join(text_terms)

            self.articles.append({
                'id': idx,
                'original_title': example['title'],
                'text': text_str,  # Preprocessed text for indexing
                'original_text': example['text'],  # Store original text for snippets
                'length': doc_length,
            })
            
            title_tokens = self._tokenize_title(example['title'])
            title_set = set(title_tokens)
            self.title_token_sets.append(title_set)
            self.title_magnitudes.append(np.sqrt(len(title_set)))
            
            #Inverted Index
            term_counts = defaultdict(int)
            for term in text_terms:
                term_counts[term] += 1

            for term, freq in term_counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term]['doc_freq'] = 0
                    self.inverted_index[term]['postings'] = []
                self.inverted_index[term]['doc_freq'] += 1
                self.inverted_index[term]['postings'].append({
                    'doc_id': idx,
                    'freq': freq
                })

            self.doc_lengths.append(doc_length)

        self.avg_doc_len = np.mean(self.doc_lengths)

    def _precompute_statistics(self):
        N = len(self.articles)
        for term in self.inverted_index:
            df = self.inverted_index[term]['doc_freq']
            self.inverted_index[term]['idf'] = np.log((N - df + 0.5) / (df + 0.5) + 1)

    def _bm25_score(self, query_terms, doc_id, k=1.2, b=0.75):
        score = 0.0
        doc_len = self.doc_lengths[doc_id]
        
        for term in query_terms:
            if term not in self.inverted_index:
                continue
                
            idf = self.inverted_index[term]['idf']
            postings = self.inverted_index[term]['postings']
            
            tf = next((p['freq'] for p in postings if p['doc_id'] == doc_id), 0)
            
            numerator = tf * (k + 1)
            denominator = tf + k * (1 - b + b * (doc_len / self.avg_doc_len))
            score += idf * (numerator / denominator)
            
        return score

    def search(self, query, bm25_weight=0.35, top_n=10):
        start_time = time.time()
        
        text_query_terms = self._preprocess_text(query)
        
        if not text_query_terms:
            return self._search_titles_only(query)
            
        candidate_docs = set()
        for term in text_query_terms:
            if term in self.inverted_index:
                candidate_docs.update(p['doc_id'] for p in self.inverted_index[term]['postings'])
                
        bm25_scores = {doc_id: self._bm25_score(text_query_terms, doc_id) 
                      for doc_id in candidate_docs}
        
        title_scores = self._calculate_title_scores(query)
        
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1
        max_title = max(title_scores.values()) if title_scores else 1
        
        combined_scores = []
        for doc_id in candidate_docs:
            bm25 = bm25_scores.get(doc_id, 0) / (max_bm25 or 1)
            title = title_scores.get(doc_id, 0) / (max_title or 1)
            combined = bm25_weight * bm25 + (1 - bm25_weight) * title
            combined_scores.append((doc_id, combined))
            
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in combined_scores[:top_n]:
            article = self.articles[doc_id]
            results.append({
                'title': article['original_title'],
                'score': score,
                'snippet': article['original_text'][:100] + '...',  # Use original text for snippet
                'id': doc_id,
            })
            
        print(f"Search completed in {time.time() - start_time:.4f}s")
        return results

    def _calculate_title_scores(self, query):
        query_tokens = self._tokenize_title(query)
        query_set = set(query_tokens)
        query_mag = np.sqrt(len(query_set)) if query_set else 0
        scores = {}
        
        for doc_id, (title_set, title_mag) in enumerate(zip(self.title_token_sets, self.title_magnitudes)):
            if not query_set or not title_set:
                similarity = 0.0
            else:
                intersection = len(query_set & title_set)
                dot_product = intersection
                mag_product = query_mag * title_mag
                similarity = dot_product / mag_product if mag_product > 0 else 0.0
            
            if similarity > 0:
                scores[doc_id] = similarity
                
        return scores

    def _search_titles_only(self, query):
        #stopword-only queries 
        scores = self._calculate_title_scores(query)
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{
            'title': self.articles[doc_id]['original_title'],
            'score': score,
            'snippet': self.articles[doc_id]['original_text'][:100] + '...',  # Use original text for snippet
            'id': doc_id,
        } for doc_id, score in sorted_docs[:10]]

if __name__ == '__main__':
    cache_dir = '/Users/gustavo/.cache/huggingface/datasets'
    dataset = load_dataset("wikipedia", "20220301.en", trust_remote_code=True, cache_dir=cache_dir)
    
    engine = SearchEngine(dataset, max_articles=100000)

    while True:
        query = input("\nEnter search query (q to quit): ").strip()
        if query.lower() == 'q':
            break
            
        start_time = time.time() 
        results = engine.search(query)
        elapsed_time = time.time() - start_time
        
        print(f"\nTop {len(results)} results (search time: {elapsed_time:.4f}s):")
        for result in results:
            print(f"[{result['score']:.3f}] {result['title']}")
            print(f"Snippet: {result['snippet']}")
from pyserini.index.lucene import IndexReader
import numpy as np
import math
from IPython.core.display import display, HTML
from tqdm import tqdm
import nltk
nltk.download('punkt')
from sklearn.utils import check_random_state

class PerturbDocument():
  # perturn docs in different ways:

  def __init__(self, num_samples=100, seed=10):
      self.num_samples = num_samples
      self.seed = seed


  def random_sampler(self, original_doc_text) -> np.array:
      """ Given a text original_doc, generate a list of perturbed doc, by randomly removing words in doc_h."""
      #original_doc_text = index_reader.doc(doc_id).raw()

      tokens = nltk.word_tokenize(original_doc_text)
      tokens_len = len(tokens)
      random_state = check_random_state(self.seed)

      #generate how many words to remove for each sample
      sample = random_state.randint(1, tokens_len - 10, self.num_samples - 1) #changed max num of words removed = tokens_len -10

      all_docs = [original_doc_text]
      for _, size in tqdm(enumerate(sample)):
          #random_state = check_random_state(None) #changed seed to None for random sampling
          #generate the idx of tokens to remove for each sample
          remove_pos = random_state.choice(range(tokens_len), size, replace=False)
          doc_adv = ' '.join([tok for i, tok in enumerate(tokens) if i not in remove_pos])
          all_docs.append(doc_adv)

      all_docs = np.array(all_docs)
      return all_docs

  def random_sampler_using_doc_id(self, doc_id, index_reader) -> np.array:
      """ Given a text original_doc, generate a list of perturbed doc, by randomly removing words in doc_h."""
      original_doc_text = index_reader.doc(doc_id).raw()
      return self.random_sampler(original_doc_text)


  def masking_sampler(self, doc_text, chunk_size=10, chunk_visible_prob=0.25):
      """ Given a text doc_text, generate a list of perturbed doc, segment a document D into D/k chunks (input chunk_size k), make chunk i visible with prob p (input p)."""
      #doc_text = index_reader.doc(doc_id).raw()

      tokens = nltk.word_tokenize(doc_text)
      tokens_len = len(tokens)
      print("num of tokens: ",tokens_len)

      num_chunks = (int) (tokens_len/chunk_size)
      if (tokens_len % chunk_size != 0):
        num_chunks += 1
      print("num of chunks: ",num_chunks)

      mask_prob_generated = np.random.rand(self.num_samples, num_chunks+1)
      #print(mask_prob_generated)

      all_docs = [doc_text]
      for sample_id in range(self.num_samples):
          doc_generated = []
          for i in range(num_chunks):
            if(mask_prob_generated[sample_id][i] > chunk_visible_prob ):
              doc_generated.append(" ".join(tokens[i*chunk_size : (i+1)*chunk_size]))

          #print(doc_generated)
          all_docs.append(" ".join(doc_generated))

      all_docs = np.array(all_docs)
      return all_docs

  def masking_sampler_using_doc_id(self, doc_id, index_reader, chunk_size=10, chunk_visible_prob=0.25):
      """ Given a text doc_text, generate a list of perturbed doc, segment a document D into D/k chunks (input chunk_size k), make chunk i visible with prob p (input p)."""
      doc_text = index_reader.doc(doc_id).raw()
      return self.masking_sampler(doc_text)


  def tfidf_sampler(self, doc_id, index_reader, num_terms_ratio = 0.2):
    #doc_id = 'LA010490-0078'
    #self_num_samples=2
    # Initialize from a pre-built index:
    #index_reader = IndexReader.from_prebuilt_index('robust04')
    #index_reader = IndexReader(index_path)

    doc_raw = index_reader.doc(doc_id).raw()
    #display(HTML('' + doc_raw + ''))
    #print("\ndoc:\n", index_reader.doc(doc_id).raw())
    tf = index_reader.get_document_vector(doc_id)
    if(tf is None):
      #Handle appropriately
      print("doc vector not stored")

    all_terms = list(tf.keys())
    #print(len(all_terms))
    all_docs = [doc_raw]
    for sample_id in range(self.num_samples):
        doc_generated = []
        tf_selected = np.random.choice( all_terms, math.floor(len(all_terms) * num_terms_ratio), replace=False)
        for terms in tf_selected:
          #print(terms, tf[terms])
          doc_generated += [terms]*tf[terms]

        #print(tf_selected)
        #print(" ".join(doc_generated))
        all_docs.append(" ".join(doc_generated))

    all_docs = np.array(all_docs)
    return all_docs


  def score_samples_with_reranker(self, query, sample_docs, reranker, batch_size=10):
      """
      sample_docs: np array cpntaining the original doc and the perturbed docs
      reranker: reranking model used to score the docs (e.g from beir )
      """
      input_pairs = [[query, doc] for doc in sample_docs]
      rank_scores = reranker.predict(input_pairs, batch_size)

      return rank_scores
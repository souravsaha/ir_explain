import pandas as pd
import os
import utils
from utils.pairwise_utils import *
from explainers.pairwise.explain_more import ExplainMore
from explainers.base_explainer import BaseExplainer

class Axiom:
    def compare(self, query, document1, document2):
        raise NotImplementedError("Subclasses should implement this!")

    def __add__(self, other):
        return CombinedAxiom(self, other, '+')

    def __sub__(self, other):
        return CombinedAxiom(self, other, '-')

    def __mul__(self, coeff):
        return ScaledAxiom(self, coeff)

    def __rmul__(self, coeff):
        return self.__mul__(coeff)

    def __and__(self, other):
        return CombinedAxiom(self, other, '&')

    def __mod__(self, other):
        return MajorityAxiom(self, other)


class CombinedAxiom(Axiom):
    def __init__(self, axiom1, axiom2, operation):
        self.axiom1 = axiom1
        self.axiom2 = axiom2
        self.operation = operation

    def compare(self, query, document1, document2):
        score1 = self.axiom1.compare(query, document1, document2)
        score2 = self.axiom2.compare(query, document1, document2)

        if self.operation == '+':
            return score1 + score2
        elif self.operation == '-':
            return score1 - score2
        elif self.operation == '&':
            # Return 1 if both are 1, else 0
            return 1 if score1 == 1 and score2 == 1 else 0

class ScaledAxiom(Axiom):
    def __init__(self, axiom, coeff):
        self.axiom = axiom
        self.coeff = coeff

    def compare(self, query, document1, document2):
        return self.coeff * self.axiom.compare(query, document1, document2)

class MajorityAxiom(Axiom):
    def __init__(self, axiom1, axiom2):
        self.axiom1 = axiom1
        self.axiom2 = axiom2

    def compare(self, query, document1, document2):
        score1 = self.axiom1.compare(query, document1, document2)
        score2 = self.axiom2.compare(query, document1, document2)

        # Return the result agreed by the majority of axioms
        if score1 == 1 and score2 == 1:
            return 1
        elif score1 == -1 and score2 == -1:
            return -1
        else:
            return 0




class PairwiseAxiomaticExplainer(BaseExplainer):

  def __init__(self, query, doc1, doc2, index_path):
        self.query = query
        self.doc1 = doc1
        self.doc2 = doc2
        self.index_path = index_path

  def explain(self, query, doc1, doc2, axiom_classes):
        results = {'Query': query, 'Document 1': doc1[:25] + '...', 'Document 2': doc2[:25] + '...'}

        for axiom in axiom_classes:
            combined_score = axiom.compare(query, doc1, doc2)

            if combined_score > 0:
                result = 1
            elif combined_score < 0:
                result = -1
            else:
                result = 0

            results[axiom.__class__.__name__] = result

        df = pd.DataFrame([results])
        return df

  def explain_details(self, query, doc1, doc2, axiom_name):
        self.index_path
        axiom_class = getattr(ExplainMore, axiom_name, None)
        if axiom_class:
            explanation_df = axiom_class.explain(query, doc1, doc2, self.index_path)
            return explanation_df
        else:
            return "Axiom not found in explain_more class"
    

  def _get_axiom_class(self, axiom_name):
        axiom_classes_mapping = {
            "TFC1": self.TFC1(),
            "TFC3": self.TFC3(self.index_path),
            "TDC": self.TDC(self.index_path),
            "M_TDC": self.M_TDC(self.index_path),
            "PROX1": self.PROX1(),
            "PROX2": self.PROX2(),
            "PROX3": self.PROX3(),
            "PROX4": self.PROX4(),
            "PROX5": self.PROX5(),
            "LNC1": self.LNC1(),
            "LNC2": self.LNC2(),
            "TF_LNC": self.TF_LNC(),
            "LB1": self.LB1(),
            "STMC1": self.STMC1(),
            "STMC2": self.STMC2(),
            "AND": self.AND(),
            "REG": self.REG(),
            "DIV": self.DIV()
        }
        return axiom_classes_mapping.get(axiom_name)

  class TFC1(Axiom):

    def compare(self,query, document1, document2):

        if abs(len(document1) - len(document2)) >= 0.1 * max(len(document1),len(document2)):
          return 0

        def term_frequency(term, document):
          return document.split().count(term)

        query_terms = query.split()

        doc1_tf = sum(term_frequency(term, document1) for term in query_terms)
        doc2_tf = sum(term_frequency(term, document2) for term in query_terms)

        if doc1_tf > doc2_tf:
            return 1
        elif doc1_tf == doc2_tf:
            return 0
        else:
            return -1

  class TFC3(Axiom):

        def __init__(self, index_path):
            self.term_discrimination_values = self.calculate_term_discrimination_values(index_path)

        def calculate_term_discrimination_values(self, index_path):
            term_doc_freq = {}
            total_docs = 0

            for filename in os.listdir(index_path):
                if filename.endswith('.txt'):
                    total_docs += 1
                    with open(os.path.join(index_path, filename), 'r') as file:
                        document = file.read()
                        terms = set(document.split())
                        for term in terms:
                            if term in term_doc_freq:
                                term_doc_freq[term] += 1
                            else:
                                term_doc_freq[term] = 1

            term_discrimination_values = {term: 1.0 / freq for term, freq in term_doc_freq.items()}
            return term_discrimination_values

        def compare(self, query, document1, document2):
            query_terms = query.split()
            query_term_set = set(query_terms)

            td_value = 1.0
            for term in query_terms:
                td_value = self.term_discrimination_values.get(term, 1.0)

            doc1_words = document1.split()
            doc2_words = document2.split()

            if len(doc1_words) != len(doc2_words):
                return 0

            c_q1_D1 = doc1_words.count(query_terms[0])
            c_q2_D1 = doc1_words.count(query_terms[1])
            c_q1_D2 = doc2_words.count(query_terms[0])
            c_q2_D2 = doc2_words.count(query_terms[1])

            if not (c_q1_D1 == c_q1_D2 + c_q2_D2 and c_q2_D1 == 0 and c_q1_D2 == 0 and c_q2_D2 == 0):
                return 0

            S_Q_D1 = td_value * (c_q1_D1 + c_q2_D1)
            S_Q_D2 = td_value * (c_q1_D2 + c_q2_D2)

            if S_Q_D1 < S_Q_D2:
                return -1
            elif S_Q_D1 > S_Q_D2:
                return 1
            else:
                return 0


  class TDC(Axiom):

    def __init__(self, index_path):
        self.term_discrimination_values = self.calculate_term_discrimination_values(index_path)

    def calculate_term_discrimination_values(self, index_path):
        term_doc_freq = {}
        total_docs = 0

        for filename in os.listdir(index_path):
            if filename.endswith('.txt'):
                total_docs += 1
                with open(os.path.join(index_path, filename), 'r') as file:
                    document = file.read()
                    terms = set(document.split())
                    for term in terms:
                        if term in term_doc_freq:
                            term_doc_freq[term] += 1
                        else:
                            term_doc_freq[term] = 1

        term_discrimination_values = {term: 1.0 / freq for term, freq in term_doc_freq.items()}
        return term_discrimination_values

    def compare(self, query, document1, document2):
        query_terms = query.split()
        if len(query_terms) != 2:
            return 0
        
        q1, q2 = query_terms
        
        document1_words = set(document1.split())
        document2_words = set(document2.split())

        if q1 in document1_words and q2 not in document1_words and q2 in document2_words and q1 not in document2_words:
            td_q1 = self.term_discrimination_values.get(q1, 1.0)
            td_q2 = self.term_discrimination_values.get(q2, 1.0)
            
            if td_q1 > td_q2:
                return 1
            elif td_q2 > td_q1:
                return -1
            else:
                return 0
        else:
            return 0
            
  class M_TDC(Axiom):

    def __init__(self, index_path):
        self.term_discrimination_values = self.calculate_term_discrimination_values(index_path)

    def calculate_term_discrimination_values(self, index_path):
        term_doc_freq = {}
        total_docs = 0

        for filename in os.listdir(index_path):
            if filename.endswith('.txt'):
                total_docs += 1
                with open(os.path.join(index_path, filename), 'r') as file:
                    document = file.read()
                    terms = set(document.split())
                    for term in terms:
                        if term in term_doc_freq:
                            term_doc_freq[term] += 1
                        else:
                            term_doc_freq[term] = 1

        term_discrimination_values = {term: 1.0 / freq for term, freq in term_doc_freq.items()}
        return term_discrimination_values

    def compare(self, query, document1, document2):
        query_terms = query.split()
        if len(query_terms) != 2:
            return 0
        
        q1, q2 = query_terms
        
        c_w1_d1 = document1.split().count(q1)
        c_w2_d1 = document1.split().count(q2)
        c_w1_d2 = document2.split().count(q1)
        c_w2_d2 = document2.split().count(q2)

        if c_w1_d1 == c_w2_d2 and c_w2_d1 == c_w1_d2:
            td_q1 = self.term_discrimination_values.get(q1, 1.0)
            td_q2 = self.term_discrimination_values.get(q2, 1.0)

            if td_q1 >= td_q2 and c_w1_d1 >= c_w1_d2:
                return 1
            elif td_q1 < td_q2 and c_w1_d1 < c_w1_d2:
                return -1
            else:
                return 0
        else:
            return 0
            
  class PROX1(Axiom):

    def compare(self,query, document1, document2):
      query_words = query.split()

      if not all(word in document1 for word in query_words) or not all(word in document2 for word in query_words):
          return 0

      words_doc1 = document1.split()
      words_doc1 = [word.replace('.', '') for word in words_doc1]
      words_doc2 = document2.split()
      words_doc2 = [word.replace('.', '') for word in words_doc2]

      common_word_pairs = [(word1, word2) for word1 in query_words if word1 in words_doc1 and word1 in words_doc2
                          for word2 in query_words if word2 in words_doc1 and word2 in words_doc2 and word1 != word2]

      words_between_pairs_doc1 = {}
      words_between_pairs_doc2 = {}

      for word1, word2 in common_word_pairs:
          indices_query_doc1 = [i for i, word in enumerate(words_doc1) if word == word1 or word == word2]
          indices_query_doc2 = [i for i, word in enumerate(words_doc2) if word == word1 or word == word2]

          if len(indices_query_doc1) == 2:
              start, end = min(indices_query_doc1), max(indices_query_doc1)
              words_between_pairs_doc1[(word1, word2)] = abs(end - start) - 1

          if len(indices_query_doc2) == 2:
              start, end = min(indices_query_doc2), max(indices_query_doc2)
              words_between_pairs_doc2[(word1, word2)] = abs(end - start) - 1

      sum_words_between_doc1 = sum(words_between_pairs_doc1.values())
      sum_words_between_doc2 = sum(words_between_pairs_doc2.values())

      total_possible_pairs = len(query_words) * (len(query_words) - 1) // 2

      ratio_doc1 = sum_words_between_doc1 / total_possible_pairs if total_possible_pairs > 0 else 0
      ratio_doc2 = sum_words_between_doc2 / total_possible_pairs if total_possible_pairs > 0 else 0

      if ratio_doc1 < ratio_doc2:
          return 1
      elif ratio_doc1 > ratio_doc2:
          return -1
      else:
          return 0

  class PROX2(Axiom):

    def compare(self,query, document1, document2):
      query_words = query.split()

      if not all(word in document1 for word in query_words) or not all(word in document2 for word in query_words):
          return 0

      words_doc1 = document1.split()
      words_doc2 = document2.split()
      first_positions_doc1 = [words_doc1.index(word) if word in words_doc1 else None for word in query_words]
      first_positions_doc2 = [words_doc2.index(word) if word in words_doc2 else None for word in query_words]

      sum_first_positions_doc1 = sum(position for position in first_positions_doc1 if position is not None)
      sum_first_positions_doc2 = sum(position for position in first_positions_doc2 if position is not None)

      if sum_first_positions_doc1 < sum_first_positions_doc2:
        return 1
      elif sum_first_positions_doc1 > sum_first_positions_doc2:
        return -1
      else:
        return 0

  class PROX3(Axiom):

    def compare(self,query, document1, document2):
      if query in document1 and query in document2:
          first_position_doc1 = document1.find(query)
          first_position_doc2 = document2.find(query)

          if first_position_doc1 < first_position_doc2:
              return 1
          elif first_position_doc1 > first_position_doc2:
              return -1
          else:
              return 0
      elif query in document1:
          return 1
      elif query in document2:
          return -1
      else:
          return 0

  class PROX4(Axiom):

    def compare(self, query, doc1, doc2):
        query_terms = set(query.split())

        def smallest_span(document):
            words = document.split()
            term_positions = {term: [] for term in query_terms}

            for idx, word in enumerate(words):
                if word in query_terms:
                    term_positions[word].append(idx)

            min_span_length = float('inf')
            min_span_non_query_count = float('inf')
            min_span = []

            for term in term_positions:
                for start_pos in term_positions[term]:
                    end_pos = start_pos
                    valid = True
                    for other_term in term_positions:
                        if other_term != term:
                            if term_positions[other_term]:
                                closest_pos = min(term_positions[other_term], key=lambda x: abs(x - start_pos))
                                end_pos = max(end_pos, closest_pos)
                            else:
                                valid = False
                                break
                    
                    if valid and end_pos - start_pos + 1 < min_span_length:
                        min_span = words[start_pos:end_pos + 1]
                        min_span_length = len(min_span)
                        min_span_non_query_count = sum(1 for word in min_span if word not in query_terms)

            return min_span_non_query_count

        def calculate_gap(document):
            min_span_non_query_count = smallest_span(document)
            words = document.split()
            gap_frequency = words.count(str(min_span_non_query_count))

            return (min_span_non_query_count, gap_frequency)

        gap1 = calculate_gap(doc1)
        gap2 = calculate_gap(doc2)

        if gap1 < gap2:
            return 1
        elif gap1 > gap2:
            return -1
        else:
            return 0



  class PROX5(Axiom):

    def compare(self,query, doc1, doc2):

      query_terms = query.split()

      def find_positions(term, document):
          positions = []
          words = document.split()
          for idx, word in enumerate(words):
              if word == term:
                  positions.append(idx)
          return positions

      def smallest_span_around(term_positions, all_positions, num_terms):
          min_span = float('inf')
          for pos in term_positions:
              spans = []
              for i in range(num_terms):
                  term_pos = all_positions[i]
                  if term_pos:
                      distances = [abs(pos - p) for p in term_pos]
                      spans.append(min(distances))
              if len(spans) == num_terms:
                  min_span = min(min_span, max(spans) - min(spans) + 1)
          return min_span

      def average_smallest_span(document):
          all_positions = [find_positions(term, document) for term in query_terms]
          total_span = 0
          count = 0

          for i, term_positions in enumerate(all_positions):
              if term_positions:
                  span = smallest_span_around(term_positions, all_positions, len(query_terms))
                  if span < float('inf'):
                      total_span += span
                      count += 1

          return total_span / count if count > 0 else float('inf')

      span1 = average_smallest_span(doc1)
      span2 = average_smallest_span(doc2)

      if span1 < span2:
          return 1
      elif span1 > span2:
          return -1
      else:
          return 0


  class LNC1(Axiom):

    def compare(self,query, doc1, doc2):

      query_words = query.split()

      count_query_terms_doc1 = sum(1 for word in query_words if word in doc1)
      count_query_terms_doc2 = sum(1 for word in query_words if word in doc2)

      max_allowed_difference = 0.1 * min(count_query_terms_doc1, count_query_terms_doc2)
      if abs(count_query_terms_doc1 - count_query_terms_doc2) > max_allowed_difference:
          return 0
      else:
          if len(doc1) == len(doc2):
              return 0
          elif len(doc1) < len(doc2):
              return 1
          else:
              return -1

  class LNC2(Axiom):

    def compare(self,query, doc1, doc2):
      original_doc, copied_doc = (doc1, doc2) if len(doc1) <= len(doc2) else (doc2, doc1)
      original_words = set(original_doc.split())
      copied_words = set(copied_doc.split())

      jaccard_coefficient = len(original_words.intersection(copied_words)) / len(original_words)

      if jaccard_coefficient >= 0.8:
          shared_terms = original_words.intersection(copied_words)
          min_frequency = min(original_doc.split().count(term) for term in shared_terms)
          max_frequency = max(original_doc.split().count(term) for term in shared_terms)
          m = max(1, min_frequency / max_frequency)
          return 1 if len(original_doc) <= len(copied_doc) else -1
      else:
          return 0

  class TF_LNC(Axiom):

      def compare(self,query, document1, document2):

          query_words = set(query.split())
          document1_words = set(document1.split())
          document2_words = set(document2.split())

          doc1_tf = query_words.intersection(document1_words)
          doc2_tf = query_words.intersection(document2_words)

          words1 = document1.split()
          words2 = document2.split()

          filtered_words1 = [word for word in words1 if word not in doc1_tf]
          filtered_words2 = [word for word in words2 if word not in doc2_tf]

          new_doc1 = ' '.join(filtered_words1)
          new_doc2 = ' '.join(filtered_words2)

          max_len = max(len(new_doc1), len(new_doc2))
          tolerance = 0.1 * max_len

          if abs(len(new_doc1) - len(new_doc2)) > tolerance:
              return 0
          else:
              if doc1_tf > doc2_tf:
                return -1
              if doc1_tf < doc2_tf:
                return 1
              if doc1_tf == doc1_tf:
                return 0

  class LB1(Axiom):

    def compare(self,query, document1, document2):
      query_terms = set(query.lower().split())
      doc1_terms = set(document1.lower().split())
      doc2_terms = set(document2.lower().split())

      unique_to_doc1 = [term for term in query_terms if term in doc1_terms and term not in doc2_terms]
      unique_to_doc2 = [term for term in query_terms if term in doc2_terms and term not in doc1_terms]

      if unique_to_doc1 == unique_to_doc2:
        return 0
      if unique_to_doc1 > unique_to_doc2:
        return 1
      return -1

  class STMC1(Axiom):

    
    def compare(self,query, document1, document2):
      similarity_doc1, similarity_doc2 = wordnet_similarity(query, document1, document2)

      if similarity_doc1 > similarity_doc2:
          return 1
      elif similarity_doc1 < similarity_doc2:
          return -1
      else:
          return 0
  
  class STMC2(Axiom):

    def compare(self, query, document1, document2):
        query_terms = set(query.split())
        
        def calculate_term_frequencies(document):
            words = document.split()
            term_freq = {}
            for word in words:
                if word in term_freq:
                    term_freq[word] += 1
                else:
                    term_freq[word] = 1
            return term_freq
        
        def find_max_similar_term(query_terms, document):
            doc_words = set(document.split())
            non_query_terms = doc_words - query_terms
            max_sim = 0
            max_term = None
            max_query_term = None
            for query_term in query_terms:
                for word in doc_words:
                    sim = w_sim2(query_term, word)
                    if sim > max_sim:
                        max_sim = sim
                        max_term = word
                        max_query_term = query_term
            return max_term, max_query_term, max_sim
        
        term_freq_d1 = calculate_term_frequencies(document1)
        term_freq_d2 = calculate_term_frequencies(document2)

        max_term_d1, max_query_term_d1, max_sim_d1 = find_max_similar_term(query_terms, document1)
        max_term_d2, max_query_term_d2, max_sim_d2 = find_max_similar_term(query_terms, document2)

        if max_sim_d1 > max_sim_d2:
            max_term = max_term_d1
            max_query_term = max_query_term_d1
        else:
            max_term = max_term_d2
            max_query_term = max_query_term_d2

        len_d1 = len(document1.split())
        len_d2 = len(document2.split())

        tf_t_d1 = term_freq_d1.get(max_term, 0)
        tf_t_d2 = term_freq_d2.get(max_term, 0)
        tf_t0_d1 = term_freq_d1.get(max_query_term, 0)

        ratio = len_d2 / len_d1
        tf_ratio = tf_t_d2 / (tf_t0_d1 if tf_t0_d1 != 0 else 1)

        if 0.18 <= ratio <= 0.22 and tf_ratio >= 0.2:
            return 1
        else:
            return -1


  class AND(Axiom):

    def compare(self,query, document1, document2):
      query_terms = set(query.lower().split())
      doc1_terms = set(document1.lower().split())
      doc2_terms = set(document2.lower().split())

      if query_terms.issubset(doc1_terms):
          return 1
      elif query_terms.issubset(doc2_terms):
          return -1
      else:
          return 0

  class REG(Axiom):

    def compare(self,query, document1, document2):
      query_terms = query.lower().split()

      most_similar_term = get_most_similar_term(query_terms)

      if not most_similar_term:
          return 0
      all_texts = [query] + [document1, document2]

      vectorizer = CountVectorizer()
      term_frequency_matrix = vectorizer.fit_transform(all_texts)

      most_similar_term_index = vectorizer.vocabulary_[most_similar_term]
      doc1_term_frequency = term_frequency_matrix[-2, most_similar_term_index]
      doc2_term_frequency = term_frequency_matrix[-1, most_similar_term_index]

      if doc1_term_frequency > doc2_term_frequency:
          return 1
      elif doc2_term_frequency > doc1_term_frequency:
          return -1
      else:
          return 0

  class DIV(Axiom):

    def compare(self,query, document1, document2):
      query_terms = set(query.lower().split())
      doc1_terms = set(document1.lower().split())
      doc2_terms = set(document2.lower().split())

      jaccard_coefficient_doc1 = len(query_terms.intersection(doc1_terms)) / len(query_terms.union(doc1_terms))
      jaccard_coefficient_doc2 = len(query_terms.intersection(doc2_terms)) / len(query_terms.union(doc2_terms))

      if jaccard_coefficient_doc1 < jaccard_coefficient_doc2:
          return 1
      else:
          return -1
      return 0
      
TFC1 = PairwiseAxiomaticExplainer.TFC1
TFC3 = PairwiseAxiomaticExplainer.TFC3
TDC = PairwiseAxiomaticExplainer.TDC
M_TDC = PairwiseAxiomaticExplainer.M_TDC
PROX1 = PairwiseAxiomaticExplainer.PROX1
PROX2 = PairwiseAxiomaticExplainer.PROX2
PROX3 = PairwiseAxiomaticExplainer.PROX3
PROX4 = PairwiseAxiomaticExplainer.PROX4
PROX5 = PairwiseAxiomaticExplainer.PROX5
LNC1 = PairwiseAxiomaticExplainer.LNC1
LNC2 = PairwiseAxiomaticExplainer.LNC2
TF_LNC = PairwiseAxiomaticExplainer.TF_LNC
LB1 = PairwiseAxiomaticExplainer.LB1
STMC1 = PairwiseAxiomaticExplainer.STMC1
STMC2 = PairwiseAxiomaticExplainer.STMC2
AND = PairwiseAxiomaticExplainer.AND
REG = PairwiseAxiomaticExplainer.REG
DIV = PairwiseAxiomaticExplainer.DIV

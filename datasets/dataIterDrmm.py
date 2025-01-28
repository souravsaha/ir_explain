from pathlib import Path
from typing import Iterable, Tuple, Callable, List, ClassVar, Dict, Any
import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab
from datasets.dataIterBase import PairwiseTrainDatasetBase, PointwiseTrainDatasetBase, ValTestDatasetBase
import datasets.glove
import numpy as np


Input = Tuple[torch.LongTensor, torch.LongTensor]
Batch = Tuple[torch.LongTensor, torch.IntTensor, torch.LongTensor]
PointwiseTrainInput = Tuple[Input, int]
PointwiseTrainBatch = Tuple[Batch, torch.FloatTensor]
PairwiseTrainInput = Tuple[Input, Input]
PairwiseTrainBatch = Tuple[Batch, Batch]
ValTestInput = Tuple[int, int, Input, int]
ValTestBatch = Tuple[torch.IntTensor, torch.IntTensor, Batch, torch.IntTensor]


def matching_histogram_mapping(query_tvs, doc_tvs, num_bins):
    def compute_similarity(A_vec, B_vec):
        dot = np.dot(A_vec, B_vec)
        normA = np.linalg.norm(A_vec)
        normB = np.linalg.norm(B_vec)
        sim = dot / (normA * normB)
        return sim

    def histogram_mapping(similarities, num_bins):
        count = [0.0] * num_bins
        for sim in similarities:
            bin_idx = int((sim + 1.0) / 2.0 * (num_bins - 1))
            count[bin_idx] += 1
        return count

    histograms = [histogram_mapping([compute_similarity(query_tv, doc_tv)
                                     for doc_tv in doc_tvs], num_bins) for query_tv in query_tvs]
    return histograms


def term2vector(vocab: Vocab, input_ids: List[int]) -> List[Any]:
    vectors = [vocab.id2vector(idx) for idx in input_ids]
    return vectors


def nh(values):
    s = np.sum(values)
    if (s > 0):
        return [v/s for v in values]
    return values


def lnh(values):
    return [np.log10(v+1) for v in values]


def _get_single_input(query: str, doc: str, vocab: Vocab, num_bins: int = 30) -> Input:
    """Return a (query, document) pair for BERT, making sure the strings are not empty
    Args:
        query (str): The query
        doc (str): The document
    Returns:
        Input: Non-empty query and document
    """
    # empty queries or documents might cause problems later on
    if len(query.strip()) == 0:
        query = '(empty)'
    if len(doc.strip()) == 0:
        doc = '(empty)'

    query_index = vocab.sent2index(query)
    doc_index = vocab.sent2index(doc)
    query_vector = term2vector(vocab, query_index)
    doc_vector = term2vector(vocab, doc_index)
    histograms = matching_histogram_mapping(query_vector, doc_vector, num_bins)
    histograms = [lnh(hist) for hist in histograms]
    return torch.LongTensor(query_index),  torch.FloatTensor(histograms)


def pad_histograms(histograms, length):
    padded = []
    _, dim = histograms[0].shape

    for hist in histograms:
        len_q, _ = hist.shape
        if len_q < length:
        # if hist.shape[0] < length:
            new_hist = torch.cat((hist, torch.tensor([[0.0]*dim]*(length-len_q))), 0)
            padded.append(new_hist)
        else:
            padded.append(hist)
    return padded


def _collate_simple(inputs: Iterable[Input], pad_id: int) -> Batch:
    """Tokenize and collate a number of single inputs, preprocessing and padding.
       Args:
           inputs (Iterable[Input]): The inputs
           tokenizer : pyserini default
       Returns:
           Batch: Input IDs, attention masks
       """
    queries, histograms = zip(*inputs)
    # query_len = [len(x) for x in queries]
    pad_queries = pad_sequence(queries, batch_first=True, padding_value=pad_id)
    max_len = pad_queries.shape[-1]
    paded_histograms = torch.stack(pad_histograms(histograms, max_len))
    return pad_queries , paded_histograms



class PointwiseTrainDataset(PointwiseTrainDatasetBase):
    def __init__(self, data_file: Path, train_file: Path, vocab_dir: Path, max_length: int=510):
        super().__init__(data_file, train_file)


class PairwiseTrainDataset(PairwiseTrainDatasetBase):
    """Dataset for pairwise training.
    Args:
        data_file (Path): Data file containing queries and documents
        train_file (Path): Trainingset file
        bert_type (str): Type for the tokenizer
    """
    def __init__(self, data_file: Path, train_file: Path, vocab: Vocab, num_bins: int):
        super().__init__(data_file, train_file)
        # self.vocab = glove.GloveTokenizer(data_file, vocab_dir, max_length)
        self.vocab = vocab
        self.pad_id = self.vocab.pad_id
        self.num_bins = num_bins

    def get_single_input(self, query: str, doc: str) -> Input:
        """Create a single model input from a query and a document.
        Args:
            query (str): The query
            doc (str): The document
        Returns:
            Input: The model input
        """
        return _get_single_input(query, doc, self.vocab, self.num_bins)

    def collate_fn(self, inputs: Iterable[PairwiseTrainInput]) -> PairwiseTrainBatch:
        """Collate a number of pairwise inputs.
        Args:
            inputs (Iterable[PairwiseTrainInput]): The inputs
        Returns:
            PairwiseTrainBatch: A batch of pairwise inputs
        """
        pos_inputs, neg_inputs = zip(*inputs)
        return _collate_simple(pos_inputs, self.pad_id), _collate_simple(neg_inputs, self.pad_id)


class ValTestDataset(ValTestDatasetBase):
    """Dataset for glove validation/testing.
    Args:
        data_file (Path): Data file containing queries and documents
        val_test_file (Path): Validationset/testset file
        bert_type (str): Type for the tokenizer
    """
    def __init__(self, data_file: Path, val_test_file: Path, vocab: Vocab, num_bins: int):
        super().__init__(data_file, val_test_file)
        self.vocab = vocab
        self.pad_id = self.vocab.pad_id
        self.num_bins = num_bins

    def get_single_input(self, query: str, doc: str) -> Input:
        """Create a single model input from a query and a document.
        Args:
            query (str): The query
            doc (str): The document
        Returns:
            Input: The model input
        """
        return _get_single_input(query, doc, self.vocab, self.num_bins)

    def collate_fn(self, val_test_inputs: Iterable[ValTestInput]) -> ValTestBatch:
        """Collate a number of validation/testing inputs.
        Args:
            val_test_inputs (Iterable[BertValInput]): The inputs
        Returns:
            ValTestBatch: A batch of validation inputs
        """
        q_ids, doc_ids, inputs, labels = zip(*val_test_inputs)
        return torch.IntTensor(q_ids), \
               torch.IntTensor(doc_ids), \
               _collate_simple(inputs, self.pad_id), \
               torch.IntTensor(labels)

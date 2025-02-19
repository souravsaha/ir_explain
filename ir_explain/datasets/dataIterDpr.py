from pathlib import Path
from typing import Any, Iterable, Tuple, List
import torch
from torch._C import device
from torch.utils.data import Dataset
from transformers import DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
from ir_explain.datasets.dataIterBase import PairwiseTrainDatasetBase, PointwiseTrainDatasetBase, ValTestDatasetBase, InferenceDatasetBase


Input = Tuple[str, str]
Batch = Tuple[torch.LongTensor, torch.LongTensor]   # [question insdex, context index]
PointwiseTrainInput = Tuple[Input, int]
PointwiseTrainBatch = Tuple[Batch, torch.FloatTensor]
PairwiseTrainInput = Tuple[Input, Input]
PairwiseTrainBatch = Tuple[Batch, Batch]
ValTestInput = Tuple[int, int, Input, int]
ValTestBatch = Tuple[torch.IntTensor, torch.IntTensor, Batch, torch.IntTensor]


def _get_single_input(query: str, doc: str) -> Input:
    """Return a (query, document) pair for DPR, making sure the strings are not empty
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
    return query, doc


def _collate_dpr(inputs: Iterable[Input], q_tokenizer: DPRQuestionEncoderTokenizer, c_tokenizer:DPRContextEncoderTokenizer, device=torch.device('cpu')) -> Batch:
    """Tokenize and collate a number of single BERT inputs, adding special tokens and padding.
    Args:
        inputs (Iterable[Input]): The inputs
        tokenizer (BertTokenizer): Tokenizer
    Returns:
        Batch: Input IDs, attention masks, token type IDs
    """
    queries, docs = zip(*inputs)
    #inputs = tokenizer(queries, docs, padding=True, truncation=True)
    q_inputs = q_tokenizer(queries, padding=True)
    c_inputs = c_tokenizer(docs, padding=True, truncation=True)
    return torch.LongTensor(q_inputs['input_ids']).to(device), \
            torch.LongTensor(c_inputs['input_ids']).to(device)
           #torch.LongTensor(inputs['attention_mask']), \
           #torch.LongTensor(inputs['token_type_ids'])

# @NotImplemented 
class PointwiseTrainDataset(PointwiseTrainDatasetBase):
    def __init__(self, data_file: Path, train_file: Path, bert_type: str):
        super().__init__(data_file, train_file)


class PairwiseTrainDataset(PairwiseTrainDatasetBase):
    """Dataset for pairwise training.
    Args:
        data_file (Path): Data file containing queries and documents
        train_file (Path): Trainingset file
        bert_type (str): Type for the tokenizer
    """
    def __init__(self, data_file: Path, train_file: Path, dpr_question: str, dpr_context: str, cache_dir: Path):
        super().__init__(data_file, train_file)
        self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(dpr_question, cache_dir=cache_dir)
        self.c_tokenizer = DPRContextEncoderTokenizer.from_pretrained(dpr_context, cache_dir=cache_dir)
    def get_single_input(self, query: str, doc: str) -> Input:
        """Create a single model input from a query and a document.
        Args:
            query (str): The query
            doc (str): The document
        Returns:
            Input: The model input
        """
        return _get_single_input(query, doc)

    def collate_fn(self, inputs: Iterable[PairwiseTrainInput]) -> PairwiseTrainBatch:
        """Collate a number of pairwise inputs.
        Args:
            inputs (Iterable[PairwiseTrainInput]): The inputs
        Returns:
            PairwiseTrainBatch: A batch of pairwise inputs
        """
        pos_inputs, neg_inputs = zip(*inputs)
        return _collate_dpr(pos_inputs, self.q_tokenizer, self.c_tokenizer), _collate_dpr(neg_inputs, self.q_tokenizer, self.c_tokenizer)


class ValTestDataset(ValTestDatasetBase):
    """Dataset for BERT validation/testing.
    Args:
        data_file (Path): Data file containing queries and documents
        val_test_file (Path): Validationset/testset file
        bert_type (str): Type for the tokenizer
    """
    def __init__(self, data_file: Path, val_test_file: Path, dpr_question: str, dpr_context: str, cache_dir: Path):
        super().__init__(data_file, val_test_file)
        self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(dpr_question, cache_dir=cache_dir)
        self.c_tokenizer = DPRContextEncoderTokenizer.from_pretrained(dpr_context, cache_dir=cache_dir)

    def get_single_input(self, query: str, doc: str) -> Input:
        """Create a single model input from a query and a document.
        Args:
            query (str): The query
            doc (str): The document
        Returns:
            Input: The model input
        """
        return _get_single_input(query, doc)

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
               _collate_dpr(inputs, self.q_tokenizer, self.c_tokenizer), \
               torch.IntTensor(labels)



class InferenceDataset(InferenceDatasetBase):

    def __init__(self, documents_file: Path, top_file: Path, dpr_question: str, dpr_context: str, cache_dir: Path, DATA: str='clueweb09', device=torch.device('cpu')):
        super().__init__(documents_file, top_file, DATA=DATA)
        q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(dpr_question, cache_dir=cache_dir)
        c_tokenizer = DPRContextEncoderTokenizer.from_pretrained(dpr_context, cache_dir=cache_dir)
        self.tokenizer = [q_tokenizer, c_tokenizer]
        self.device = device

    # def __init_q_docs__(self, q_id: str, query: str):
        # pass

    def get_single_input(self, query: str, doc: str) -> Batch:
        query, doc = _get_single_input(query, doc)
        inputs = [[query, doc]]
        return _collate_dpr(inputs, self.tokenizer[0], self.tokenizer[1], device=self.device)
        


class CustomDataset(Dataset):
    def __init__(self, query: str, docs: List[str], tokenizer, device=torch.device('cpu')):
        self.query = query
        self.docs = docs
        self.tokenizer = tokenizer  # no need to load repeatedly 
        self.device = device

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        doc = self.docs[idx]
        query, doc = _get_single_input(self.query, doc)
        return query, doc

    def collate_fn(self, inputs):
        """Collate a number of pairwise inputs.
        Args:
            inputs (Iterable[PairwiseTrainInput]): The inputs
        Returns:
            PairwiseTrainBatch: A batch of pairwise inputs
        """

        return _collate_dpr(inputs, self.tokenizer[0], self.tokenizer[1], device=self.device)

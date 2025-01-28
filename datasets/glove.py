import torch
import torchtext
from torchtext.vocab import Vocab
from typing import Iterable, Tuple, Callable, List, ClassVar, Dict
from pathlib import Path
from tqdm import tqdm
import h5py
import pickle
import nltk
import os
from itertools import chain
from collections import Counter
from pyserini.analysis import Analyzer, get_lucene_analyzer
bert_special_token = {'unk_token': {'[UNK]': 100},
                      'sep_token': {'[SEP]': 102},
                      'pad_token': {'[PAD]': 0},
                      'cls_token': {'[CLS]': 101},
                      'mask_token': {'[MASK]': 103}
                      }


class GloveTokenizer(object):
    """Init Bert-like tokenizer with pretrained glove embedding.
        Use Bert special tokens and function names.
        args:
            data_file: address of 'query, doc' file
            vocab_dir: vocab pickle
    """
    def __init__(self, data_file: Path, vocab_dir: Path, max_length: int = 510):
        self.data_file = data_file
        self.vocab_dir = vocab_dir
        self.max_length = max_length
        #self._init_analyze()
        self._init_glove()
        self.pad_id = self.vocab.stoi['<pad>']
        self.unk_id = self.vocab.stoi['<unk>']

    def _init_analyze(self, stop: str='english', stem: bool=False, stemmer:str='krovetz') -> None:
        self.analyzer = Analyzer(get_lucene_analyzer(stopwords=stop, stemming=stem, stemmer=stemmer))

    def _tokenize(self, sentence: str) -> List[str]:
        #return self.analyzer.analyze(sentence)
        return nltk.word_tokenize(sentence)

    def _index(self, tokens: List[str]) -> List[int]:
        return [self.vocab.stoi[tok] for tok in tokens]

    def sent2index(self, sentence: str) -> List[int]:
        tokens = self._tokenize(sentence)
        return self._index(tokens)

    def id2vector(self, idx: int):
        return self.vocab.vectors[idx].detach().numpy()

    def _index_pair_batch(self, queries: List[str], docs: List[str]) -> Dict[str, List[List[int]]]:
        """Tokenize and collate a number of single inputs, indexing and padding.
            similar as BertTokenizer.
            Args:
                inputs (Iterable[Input]): The inputs
                # tokenizer: _simple_tokenize
                # Vocab for indexing: customized, e.g, glove
            Returns:
                dict: input_ids, attention_mask, token_type_ids
        """
        input_ids, token_type_ids, attention_mask = [], [], []
        indexes_que = [self._index(self._tokenize(que)) for que in queries]
        indexes_doc = [self._index(self._tokenize(doc)) for doc in docs]
        max_length = max([len(q) + len(d) for (q, d) in zip(indexes_que, indexes_doc)])
        if max_length > self.max_length:
            max_length = self.max_length

        for index_que, index_doc in zip(indexes_que, indexes_doc):
            # index_que, index_doc = self._index(self._tokenize(query)), self._index(self._tokenize(doc))
            len_que, len_doc = len(index_que), len(index_doc)
            pad_length = max(0, (max_length - (len_que + len_doc)))
            if not pad_length:
                # input too long, need to truncate doc
                len_doc = max_length - len_que
                index_doc = index_doc[:len_doc]

            index = index_que + index_doc + [0] * pad_length
            token_type = [0] * len_que + [1] * len_doc + [0] * pad_length
            attention_ = [1] * (len_que + len_doc) + [0] * pad_length

        input_ids.append(index)
        token_type_ids.append(token_type)
        attention_mask.append(attention_)
        return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}

    def _init_glove(self, max_size: int=400000, name: str='glove.840B.300d', cache_dir: Path=Path('/home/lyu/robustness/pretrained')) -> None:
        if os.path.isfile(self.vocab_dir):
            print(f'loading glove vocabulary...')
            with open(self.vocab_dir, 'rb')as f:
                vocab = pickle.load(f)
        else:
            print(f'reading {self.data_file}...')
            with h5py.File(self.data_file, 'r') as fp:
                num_items = len(fp['queries']) + len(fp['docs'])
                ct = Counter()
                for s in tqdm(chain(fp['queries'], fp['docs']), total=num_items):
                    ct.update(nltk.word_tokenize(s))
                vocab = Vocab(ct, max_size, vectors=name, vectors_cache=cache_dir, unk_init=torch.nn.init.normal_)

            print(f'writing {self.vocab_dir}...')
            with open(self.vocab_dir, 'wb') as fp:
                pickle.dump(vocab, fp)
        print(f'vocab size: {len(vocab)}')
        self.vocab = vocab



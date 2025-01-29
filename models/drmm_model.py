import torch
from torchtext.vocab import Vocab
from argparse import ArgumentParser
from typing import Any, Dict, Union
from models.base_model import BaseRanker
from datasets.dataIterDrmm import PointwiseTrainDataset, PairwiseTrainDataset, ValTestDataset, Batch, PointwiseTrainBatch, PairwiseTrainBatch

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch.nn.init as weight_init
from torch.autograd import Variable

import sys
sys.path.append("/home/lyu/ExpRank/")
sys.path.append("/home/lyu/ExpRank/Datasets/")
import datasets.glove
torch.manual_seed(222)

use_cuda = torch.cuda.device_count() > 0

if use_cuda:
    torch.cuda.manual_seed(222)


class DRMM(nn.Module):

    def __init__(self, dim_term_gating, num_bins: int = 30):
        super(DRMM, self).__init__()

        # feedfoward matching network
        self.z1 = nn.Linear(num_bins, 5)
        self.z2 = nn.Linear(5, 1)
        weight_init.xavier_normal(self.z1.weight, gain=weight_init.calculate_gain('tanh'))
        weight_init.xavier_normal(self.z2.weight, gain=weight_init.calculate_gain('tanh'))
        # term gating network

        self.g = nn.Linear(dim_term_gating, 1, bias=False)
        weight_init.xavier_normal(self.g.weight, gain=weight_init.calculate_gain('linear'))

    def forward(self,  histograms, queries_tvs):
        assert(len(queries_tvs.shape)==3)  # B, L, H
        assert(len(histograms.shape)==3)
        out_ffn = self.z1(histograms)
        out_ffn = F.tanh(out_ffn)
        out_ffn = self.z2(out_ffn)
        out_ffn = F.tanh(out_ffn)
    
        out_tgn = F.softmax(self.g(queries_tvs))

        assert(out_tgn.shape == out_ffn.shape)
        matching_score = torch.sum(out_ffn * out_tgn, dim=1)

        return matching_score


class HingeLoss(torch.nn.Module):
    """
        Hinge Loss
          max(0, 1-x+y)
    """

    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, x, y):
        output = 1 - x + y
        return output.clamp(min=0).mean()


class DRMMRanker(BaseRanker):
    """QA-DRMM for passage ranking using GloVe embeddings.
    Args:
        hparams (Dict[str, Any]): All model hyperparameters
        vocab (Vocab): Vocabulary
        rr_k (int, optional): Compute MRR@k. Defaults to 10.
        num_workers (int, optional): Number of DataLoader workers. Defaults to 16.
        training_mode (str, optional): Training mode, 'pointwise' or 'pairwise'. Defaults to 'pairwise'.
    """
    def __init__(self, hparams: Dict[str, Any], rr_k: int = 10, num_workers: int = 16, training_mode: str = 'pairwise'):
        self.vocab = glove.GloveTokenizer(hparams['data_file'], hparams['vocab_dir'])
        if training_mode == 'pointwise':
            train_ds = PointwiseTrainDataset(hparams['data_file'], hparams['train_file_pointwise'], self.vocab)
        else:
            assert training_mode == 'pairwise'
            train_ds = PairwiseTrainDataset(hparams['data_file'], hparams['train_file_pairwise'], self.vocab, hparams['num_bins'])
        val_ds = ValTestDataset(hparams['data_file'], hparams['val_file'], self.vocab, hparams['num_bins'])
        test_ds = ValTestDataset(hparams['data_file'], hparams['test_file'], self.vocab, hparams['num_bins'])
        uses_ddp = 'ddp' in hparams['distributed_backend']
        super().__init__(hparams, train_ds, val_ds, test_ds, hparams['loss_margin'], hparams['batch_size'], rr_k, num_workers, uses_ddp)

        pad_id = self.vocab.pad_id
        self.emb_dim = self.vocab.vocab.vectors.shape[1]
        self.embedding = torch.nn.Embedding.from_pretrained(self.vocab.vocab.vectors, freeze=False, padding_idx=pad_id)
        self.hist_bin_size = hparams['num_bins']
        self.drmm = DRMM(self.emb_dim)

    def forward(self, batch: Batch) -> torch.Tensor:
        """Return the similarities for all query and document pairs.
        Args:
            batch (Batch): The input batch
        Returns:
            torch.Tensor: The similarities
        """
        queries,  histograms = batch
        queries = self.embedding(queries)
        return self.drmm(Variable(histograms), queries)

    def configure_optimizers(self) -> torch.optim.Adam:
        """Create an Adam optimizer.
        Returns:
            torch.optim.Adam: The optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])

    def training_step(self, batch: Union[PointwiseTrainBatch, PairwiseTrainBatch], batch_idx: int) -> torch.Tensor:
        """Train a single batch. In pairwise training mode, we use the similarity directly without sigmoid.
        Args:
            batch (Union[PointwiseTrainBatch, PairwiseTrainBatch]): A training batch, depending on the mode
            batch_idx (int): Batch index
        Returns:
            torch.Tensor: Training loss
        """
        print('Batch size: ', len(batch))
        if self.training_mode == 'pointwise':
            inputs, labels = batch
            loss = self.bce(self(inputs).flatten(), labels.flatten())
        else:
            pos_inputs, neg_inputs = batch
            pos_outputs = self(pos_inputs)
            neg_outputs = self(neg_inputs)
            # loss = self.criterion(pos_outputs, neg_outputs)
            loss = torch.mean(torch.clamp(self.loss_margin - pos_outputs + neg_outputs, min=0))
        self.log('train_loss', loss)
        return loss

    @staticmethod
    def add_model_specific_args(ap: ArgumentParser):
        """Add model-specific arguments to the parser.
        Args:
            ap (ArgumentParser): The parser
        """
        # ap.add_argument('--hidden_dim', type=int, default=256, help='The hidden dimensions throughout the model')
        # ap.add_argument('--dropout', type=float, default=0.5, help='Dropout percentage')
        ap.add_argument('--lr', type=float, default=0.001, help='Learning rate')
        ap.add_argument('--loss_margin', type=float, default=0.2, help='Margin for pairwise loss')
        ap.add_argument('--batch_size', type=int, default=32, help='Batch size')

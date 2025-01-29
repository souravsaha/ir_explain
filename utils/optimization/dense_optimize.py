""" Solve the rank pair coverage problem by NN models """
from pathlib import Path
from pytorch_lightning import seed_everything
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from typing import Dict, List, Tuple
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import pickle
import os

#random_seed = 123
#seed_everything(random_seed)

class DenseModel(pl.LightningModule):
    def __init__(self, exp_num: int, terms_num: int):
        super(DenseModel, self).__init__()
        self.linears = nn.ParameterList()
        for i in range(exp_num):
            #self.linears.append(Parameter(torch.Tensor(terms_num)))
            self.linears.append(Parameter(torch.randn(terms_num).uniform_(-1, 1)))
        self.classify = nn.Sigmoid()
        

    def forward(self, Input, get_label=True):
        Batch, M, N = Input.shape
        outs = []
        for i in range(M):
            out = torch.tanh(self.linears[i] * Input[:, i, :])
            outs.append(out.unsqueeze(1))   # Batch, 1, N
        
        outs = torch.cat(outs, dim=1)    # Batch, M, N
        outs, _ = torch.max(outs, dim=1) # Batch, N
        if get_label:
            outs = self.classify(torch.sum(outs, dim=1))    # Batch, 
        return outs

    def training_step(self, batch, batch_idx):
        X, Y = batch
        Y_hat = self(X)
        if (torch.isnan(Y_hat)).any() or (torch.isinf(Y_hat)).any():
            nan_idx = torch.where(torch.isnan(Y_hat))[0]

            print(X[nan_idx], '\n---------------------\n', Y_hat[nan_idx])
        loss = F.binary_cross_entropy(Y_hat, Y)
        self.log('train_loss', loss.data)
        return {'loss': loss, 'preds': Y_hat, 'targets': Y}

    #def training_step_end(self, outs):
        #acc = FM.accuracy(torch.round(outs['preds']).long(), outs['targets'].long())
        #print(acc)
        #self.log("train/acc_step", acc)

    def training_epoch_end(self, outs) -> None:
        ACC = []
        for out_B in outs:
            preds = out_B['preds']
            targets = out_B['targets']
            acc = (torch.round(preds) == targets).float().sum()/len(preds)
            ACC.append(acc.data)
        #print(f'Train acc: {sum(ACC)/len(ACC)}')
        self.log("train_acc", acc)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if (torch.isnan(x)).any() or (torch.isinf(x)).any():
            print('Input: ', x)
            print('Weights: ', self.linears)
        if (torch.isnan(y_hat)).any() or (torch.isinf(y_hat)).any():
            print('input: ',x, '\n-------------------------------\n', 'output: ', y_hat)
            print('weights: ', self.linears)
            print('nan in weights: ',torch.isnan(self.linears[0]).any(), torch.isnan(self.linears[1]).any())
        # print(y_hat.dtype, y.dtype)
        with open('/home/lyu/ExpRank/debug.pkl','wb')as f:
            pickle.dump([x.detach().cpu().numpy(), y.detach().cpu().numpy(), y_hat.detach().cpu().numpy(), [w.detach().cpu().numpy() for w in self.linears]], f)
        
        loss = F.binary_cross_entropy(y_hat, y)
        return {'val_loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        return optimizer


class DenseOptimize:
    def __init__(self, hparams: Dict):
        self.margs = {'exp_num': hparams['exp_num'], 'terms_num': hparams['terms_num']}
        self.model = DenseModel(**self.margs).double()
        self.outdir = hparams['outdir']

    def _train(self, data: List):
        train_size = int(len(data) * 0.8)
        valid_size = len(data) - train_size
        train, valid = random_split(data, [train_size, valid_size], generator=torch.Generator().manual_seed(42))

        early_stopping = EarlyStopping(monitor='train_acc', mode='max', stopping_threshold=0.95,  patience=100, verbose=True)
        model_checkpoint = ModelCheckpoint(monitor='train_acc', mode='max', verbose=True)
        trainer = pl.Trainer(gpus=1, max_epochs=1000, default_root_dir = self.outdir, callbacks=[LearningRateMonitor(), early_stopping, model_checkpoint], deterministic=True)
        trainer.fit(self.model, DataLoader(data, batch_size=32, shuffle=True, num_workers=8), DataLoader(valid, batch_size=32, num_workers=8))

    def _load_weights(self):
        self.Weights = []
        checkpoints = self.outdir / 'lightning_logs' / 'version_0' / 'checkpoints'
        model_dir = list(checkpoints.glob('epoch*'))[0]  # The first saved model
        self.model = DenseModel.load_from_checkpoint(model_dir, **self.margs).eval()
        for i in range(len(self.model.linears)):
            W = self.model.linears[i].detach().cpu().clone()
            self.Weights.append(W)

    def _repeat_forward(self, pair: np.array) -> Tuple[np.array, np.array]:
        W = self.Weights.copy()
        assert pair.shape[0]==len(W)
        dots, non_negs = [], []
        indexes, values = [],[]
        for i in range(len(W)):
            dot = torch.tanh(W[i] * pair[i, :])
            dots.append(dot.data)
            non_neg = torch.where(dot>0)[0]
            non_negs.append(non_neg)
        dots = torch.stack(dots)
        out, position = torch.max(dots, 0)
        for i in range(len(non_negs)):
            picked_i = torch.where(position == i)[0]
            picked_index = np.intersect1d(non_negs[i].cpu().numpy(), picked_i.cpu().numpy())
            indexes.append(picked_index)
            values.append(dots[i, :][picked_index].cpu().numpy())
        return indexes, values 

    def _term_ids_all(self, pairs:List[np.array]):
        expansions = {}
        exp_num = self.margs['exp_num']
        for i in range(len(pairs)):
            pair = pairs[i][0]
            pred = self.model(torch.tensor([pair])).detach().cpu().round().item()
            if pred < 1:      # punish the wrong prediction.
                sign = -1
            else:
                sign = 1
            indexs, scores = self._repeat_forward(pair)
            for e in range(exp_num):
                Index = indexs[e]
                Score = scores[e]
                for j, s in zip(Index, Score):
                    if j in expansions:
                        expansions[j][e] += s*sign
                    else:
                        expansions[j] = [0] * exp_num
                        expansions[j][e] = s*sign

        sort_expansions = sorted(expansions.items(), key=lambda kv: sum(kv[1]), reverse=True)
        return sort_expansions

#@deprecated
def Train(weights: List[List[List[float]]], out_dir: Path) -> None:  
    """Train model"""
    matrixs = np.stack(weights).astype(np.double)
    exp_num, term_num, pairs = matrixs.shape
    positive = [(matrixs[:, :, i], torch.tensor(1).double()) for i in range(pairs)]
    negative = [(-matrixs[:, :, i], torch.tensor(0).double()) for i in range(pairs)] 
    args = {'exp_num': exp_num, 'terms_num': term_num, 'outdir': out_dir}
    optimizer = DenseOptimize(args)
    optimizer._train(positive+negative)


def Optimize(candidates_tokens: List[str], EXP_model: List[str], fold_dir: Path, q_id: str, max_k: int=10) -> List[str]:
    """Generate expansion terms from trianed model"""   
    matrixs, optimizer = read_matrixs(EXP_model, fold_dir, q_id)
    positive_pairs = [(matrixs[:, :, i], torch.tensor(1).double()) for i in range(matrixs.shape[2]) if (matrixs[:, :, i]!=0).any()]
    # normalize saliency
    positive_pairs = [(normalize(m, 1), label) for m, label in positive_pairs]
    negative_pairs = [(-inp, torch.tensor(0).double()) for (inp, label) in positive_pairs] 
    
    print(f"Pairs after 0-filtering: {len(positive_pairs)}")
    if not os.path.exists(optimizer.outdir):
        optimizer._train(positive_pairs + negative_pairs)
    optimizer._load_weights()

    sorted_expands = optimizer._term_ids_all(positive_pairs)
    top_expands = [k for k, v in sorted_expands[:max_k]]
    terms = [candidates_tokens[k] for k in top_expands]
    return terms

def rescale_matrix(EXP_model: List[str], fold_dir: Path, q_id: str):
    ''' Only return the matrixs combined with multi-explainers.'''
    matrixs, optimizer = read_matrixs(EXP_model, fold_dir, q_id)
    optimizer._load_weights()
    positive_pairs = [(matrixs[:, :, i], torch.tensor(1).double()) for i in range(matrixs.shape[2])]
    # normalize saliency scores across terms
    positive_pairs = [(normalize(m, 1), label) for m, label in positive_pairs]
    data_iter = DataLoader(positive_pairs, batch_size=32, shuffle=True, num_workers=8)
    Weights_Rescale = np.empty((0, matrixs.shape[1]))
    for B, _ in data_iter:
        weights_rescale = optimizer.model(B, get_label=False).detach().cpu().numpy()   # batch, term
        Weights_Rescale = np.append(Weights_Rescale, weights_rescale, axis=0)
    return Weights_Rescale.transpose(1, 0).tolist()


def read_matrixs(EXP_model: List[str], fold_dir: Path, q_id: str):
    matrixs = []
    for M in EXP_model:
        M_dir = fold_dir / f"{q_id}_matrix_{M}_False.pkl"
        with open(M_dir, 'rb')as f:
            matrix = pickle.load(f)
        matrixs.append(matrix)
    matrixs = np.stack(matrixs).astype(np.double)
    exp_num, term_num, doc_pair_num = matrixs.shape
    out_dir = fold_dir / f'Dense_{q_id}_norm'
    args = {'exp_num': exp_num, 'terms_num': term_num, 'outdir': out_dir}
    optimizer = DenseOptimize(args)
    return matrixs, optimizer


def normalize(m: np.array, i: int):
    if (m[i, :]!=0).any():
        m[i, :] = m[i, :]/np.abs(m[i, :]).sum()
    return m

def main():
    dataset = 'clueweb09'
    fold = 'fold_1' 
    q_id = 80
    Rerank_model = 'bert'
    folder = Path(f'/home/lyu/ExpRank/Exp_results/{dataset}/{fold}/matrix_{Rerank_model}')
    EXP_model = ['language_model', 'saliency']
    matrixs = []
    for M in EXP_model:
        M_dir = folder / f"{q_id}_matrix_{M}_False.pkl"
        with open(M_dir, 'rb')as f:
            matrix = pickle.load(f)
        matrixs.append(matrix)
    out_dir = folder / f'Dense_{q_id}'
    # candidates_file = folder / f'{q_id}_candidates.json'
    Train(matrixs, out_dir)
    

if __name__ == '__main__':
    main()

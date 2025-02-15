from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import abc
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning import LightningModule

import sys
sys.path.append("/a/administrator/codebase/neural-ir/RankingExplanation_bkp/Datasets/")
sys.path.append("/a/administrator/codebase/neural-ir/RankingExplanation_bkp/utilities/")
from utils.metrics import average_precision, reciprocal_rank
#from sampler import DistributedQuerySampler
from datasets.dataIterBase import PointwiseTrainDatasetBase, PairwiseTrainDatasetBase, ValTestDatasetBase

# input batches vary for each model, hence we use Any here
InputBatch = Any
PointwiseTrainBatch = Tuple[InputBatch, torch.IntTensor]
PairwiseTrainBatch = Tuple[InputBatch, InputBatch]
ValTestBatch = Tuple[torch.IntTensor, torch.IntTensor, InputBatch, torch.IntTensor]


class BaseRanker(LightningModule, abc.ABC):
    """
    Abstract base class for re-rankers. Implements average precision and reciprocal rank validation.
    
    This class needs to be extended and (at least) the following methods must be implemented:
    
    - forward
    - configure_optimizers
    
    Since this class uses custom sampling in DDP mode, the `Trainer` object must be initialized using
    `replace_sampler_ddp=False` and the argument `uses_ddp=True` must be set when DDP is active.
    
    Args:
        hparams (Dict[str, Any]): All model hyperparameters
        train_ds (Union[PointwiseTrainDatasetBase, PairwiseTrainDatasetBase]): The training dataset
        val_ds (Optional[ValTestDatasetBase]): The validation dataset
        test_ds (Optional[ValTestDatasetBase]): The testing dataset
        loss_margin (float, optional): Margin used in pairwise loss
        batch_size (int): The batch size
        rr_k (int, optional): Compute RR@K. Defaults to 10.
        num_workers (int, optional): Number of DataLoader workers. Defaults to 16.
        uses_ddp (bool, optional): Whether DDP is used. Defaults to False.
    """

    def __init__(self, hparams: Dict[str, Any],
                 train_ds: Union[PointwiseTrainDatasetBase, PairwiseTrainDatasetBase],
                 val_ds: Optional[ValTestDatasetBase], test_ds: Optional[ValTestDatasetBase],
                 loss_margin: Optional[float],
                 batch_size: int, rr_k: int = 10,
                 num_workers: int = 16, uses_ddp: bool = False):
        super().__init__()
        #self.hparams = hparams        # pytorch_lighnting version updates.
        self.hparams.update(hparams)
        self.save_hyperparameters(hparams)

        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.loss_margin = loss_margin
        self.batch_size = batch_size
        self.rr_k = rr_k
        self.num_workers = num_workers
        self.uses_ddp = uses_ddp
        if issubclass(train_ds.__class__, PointwiseTrainDatasetBase):
            self.training_mode = 'pointwise'
            self.bce = torch.nn.BCEWithLogitsLoss()
        elif issubclass(train_ds.__class__, PairwiseTrainDatasetBase):
            self.training_mode = 'pairwise'
        else:
            self.training_mode = None

    def train_dataloader(self) -> DataLoader:
        """Return a trainset DataLoader. If the trainset object has a function named `collate_fn`,
        it is used. If the model is trained in DDP mode, the standard `DistributedSampler` is used.
        Returns:
            DataLoader: The DataLoader
        """
        if self.uses_ddp:
            sampler = DistributedSampler(self.train_ds, shuffle=True)
            shuffle = None
        else:
            sampler = None
            shuffle = True

        return DataLoader(self.train_ds, batch_size=self.batch_size, sampler=sampler, shuffle=shuffle,
                          num_workers=self.num_workers, collate_fn=getattr(self.train_ds, 'collate_fn', None))

    def val_dataloader(self) -> Optional[DataLoader]:
        """Return a validationset DataLoader if the validationset exists. If the validationset object has a function
        named `collate_fn`, it is used. If the model is validated in DDP mode, `DistributedQuerySampler` is used
        for ranking metrics to work on a query level.
        Returns:
            Optional[DataLoader]: The DataLoader, or None if there is no validation dataset
        """
        if self.val_ds is None:
            return None

        if self.uses_ddp:
            #sampler = DistributedQuerySampler(self.val_ds)
            sample = None
        else:
            sampler = None

        return DataLoader(self.val_ds, batch_size=self.batch_size, sampler=sampler, shuffle=False,
                          num_workers=self.num_workers, collate_fn=getattr(self.val_ds, 'collate_fn', None))

    def test_dataloader(self) -> Optional[DataLoader]:
        """Return a testset DataLoader if the testset exists. If the testset object has a function
        named `collate_fn`, it is used. If the model is tested in DDP mode, the standard `DistributedSampler` is used.
        Returns:
            Optional[DataLoader]: The DataLoader, or None if there is no testing dataset
        """
        if self.test_ds is None:
            return None

        if self.uses_ddp:
            sampler = DistributedSampler(self.test_ds, shuffle=False)
        else:
            sampler = None

        return DataLoader(self.test_ds, batch_size=self.batch_size, sampler=sampler, shuffle=False,
                          num_workers=self.num_workers, collate_fn=getattr(self.test_ds, 'collate_fn', None))

    def training_step(self, batch: Union[PointwiseTrainBatch, PairwiseTrainBatch], batch_idx: int) -> torch.Tensor:
        """Train a single batch.
        Args:
            batch (Union[PointwiseTrainBatch, PairwiseTrainBatch]): A training batch, depending on the mode
            batch_idx (int): Batch index
        Returns:
            torch.Tensor: Training loss
        """
        if self.training_mode == 'pointwise':
            inputs, labels = batch
            loss = self.bce(self(inputs).flatten(), labels.flatten())
        elif self.training_mode == 'pairwise':
            pos_inputs, neg_inputs = batch
            pos_outputs = torch.sigmoid(self(pos_inputs))
            neg_outputs = torch.sigmoid(self(neg_inputs))
            loss = torch.mean(torch.clamp(self.loss_margin - pos_outputs + neg_outputs, min=0))
        else:
            print(self.training_mode)
            raise RuntimeError(
                'Unsupported training dataset (should subclass PointwiseTrainDatasetBase or PairwiseTrainDatasetBase)')
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: ValTestBatch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Process a single validation batch.
        Args:
            batch (ValTestBatch): Query IDs, document IDs, inputs and labels
            batch_idx (int): Batch index

        Returns:
            Dict[str, torch.Tensor]: Query IDs, predictions and labels
        """
        q_ids, _, inputs, labels = batch
        return {'q_ids': q_ids, 'predictions': self(inputs), 'labels': labels}

    def test_step(self, batch: ValTestBatch, batch_idx: int):
        """Process a single test batch. The resulting query IDs, predictions and labels are written to files.
        In DDP mode one file for each device is created. The files are created in the `save_dir` of the logger.
        Args:
            batch (ValTestBatch): Query IDs, document IDs, inputs and labels
            batch_idx (int): Batch index
        """
        q_ids, doc_ids, inputs, labels = batch
        out_dict = {
            'q_id': [self.test_ds.get_original_query_id(q_id.cpu()) for q_id in q_ids],
            'doc_id': [self.test_ds.get_original_document_id(doc_id.cpu()) for doc_id in doc_ids],
            'prediction': self(inputs),
            'label': labels
        }
        save_dir = Path(self.logger.save_dir)
        self.write_prediction_dict(out_dict, str(save_dir / 'test_outputs.pt'))

    def validation_epoch_end(self, val_results: Iterable[Dict[str, torch.Tensor]]):
        """Accumulate all validation batches and compute MAP and MRR@k. The results are approximate in DDP mode.
        Args:
            val_results (Iterable[Dict[str, torch.Tensor]]): Query IDs, predictions and labels
        """
        temp = defaultdict(lambda: ([], []))
        for r in val_results:
            # print(r)
            # for a, b, c in zip(r['q_ids'], r['predictions'], r['labels']):
                # print(a,b,c)
            for q_id, (prediction,), label in zip(r['q_ids'], r['predictions'], r['labels']):
                # q_id is a tensor with one element, we convert it to an int to use it as dict key
                q_id = int(q_id.cpu())
                temp[q_id][0].append(prediction)
                temp[q_id][1].append(label)
        aps, rrs = [], []
        for predictions, labels in temp.values():
            predictions = torch.stack(predictions)
            labels = torch.stack(labels)
            aps.append(average_precision(predictions, labels))
            rrs.append(reciprocal_rank(predictions, labels, self.rr_k))
        self.log('val_map', torch.mean(torch.stack(aps)), sync_dist=self.uses_ddp, sync_dist_op='mean')
        self.log('val_mrr', torch.mean(torch.stack(rrs)), sync_dist=self.uses_ddp, sync_dist_op='mean')

from transformers import DPRQuestionEncoder, DPRContextEncoder
from transformers import AdamW, get_constant_schedule_with_warmup
from ir_explain.models.base_model import BaseRanker
from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple
import sys
sys.path.append('/a/administrator/codebase/neural-ir/RankingExplanation_bkp/Datasets/')
sys.path.append('/a/administrator/codebase/neural-ir/RankingExplanation_bkp/models/')
from ir_explain.datasets.dataIterDpr import PointwiseTrainDataset, PairwiseTrainDataset, ValTestDataset, Batch
import torch



class DprRanker(BaseRanker):
    """Vanilla two-tower ranker.
    Args:
        hparams (Dict[str, Any]): All model hyperparameters
    """
    def __init__(self, hparams: Dict[str, Any]):
        train_ds = None
        if hparams.get('training_mode') == 'pointwise':
            train_ds = PointwiseTrainDataset(hparams['data_file'], hparams['train_file_pointwise'], hparams['question_model'],hparams['context_model'], hparams['dpr_cache'])
        elif hparams.get('training_mode') == 'pairwise':
            train_ds = PairwiseTrainDataset(hparams['data_file'], hparams['train_file_pairwise'], hparams['question_model'], hparams['context_model'], hparams['dpr_cache'])
        val_ds = None
        if hparams.get('val_file') is not None:
            val_ds = ValTestDataset(hparams['data_file'], hparams['val_file'], hparams['question_model'], hparams['context_model'], hparams['dpr_cache'])
        test_ds = None
        if hparams.get('test_file') is not None:
            test_ds = ValTestDataset(hparams['data_file'], hparams['test_file'], hparams['question_model'], hparams['context_model'], hparams['dpr_cache'])

        rr_k = hparams.get('rr_k', 10)
        num_workers = hparams.get('num_workers')
        uses_ddp = 'ddp' in hparams.get('accelerator', '')
        super().__init__(hparams, train_ds, val_ds, test_ds, hparams['loss_margin'], hparams['batch_size'], rr_k, num_workers, uses_ddp)

        #self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(hparams['question_model'], cache_dir=hparams['dpr_cache'])
        self.question_encoder = DPRQuestionEncoder.from_pretrained(hparams['question_model'], cache_dir=hparams['dpr_cache'])
        #self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(hparams['context_model'], cache_dir=hparams['dpr_cache'])
        self.context_encoder = DPRContextEncoder.from_pretrained(hparams['context_model'], cache_dir=hparams['dpr_cache'])
        
        #print(hparams.training_mode)
        #print('training mode: ', self.training_mode)
        for p in self.question_encoder.parameters() :
            p.requires_grad = not hparams['freeze_bert']
        for p in self.context_encoder.parameters():
            p.requires_grad = not hparams['freeze_bert']

    def forward(self, batch: Batch) -> torch.Tensor:
        """Compute the relevance scores for a batch.
        Args:
            batch (Batch): tokenized and indexed DPR inputs
        Returns:
            torch.Tensor: The output scores, shape (batch_size, 1)
        """
        q_vec = self.question_encoder(batch[0]).pooler_output
        c_vec = self.context_encoder(batch[1]).pooler_output
        score = torch.cosine_similarity(q_vec, c_vec)
        return score

    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        """Create an AdamW optimizer using constant schedule with warmup.
        Returns:
            Tuple[List[Any], List[Any]]: The optimizer and scheduler
        """
        params_with_grad = filter(lambda p: p.requires_grad, self.parameters())
        opt = AdamW(params_with_grad, lr=self.hparams['lr'])
        sched = get_constant_schedule_with_warmup(opt, self.hparams['warmup_steps'])
        return [opt], [{'scheduler': sched, 'interval': 'step'}]

    @staticmethod
    def add_model_specific_args(ap: ArgumentParser):
        """Add model-specific arguments to the parser.
        Args:
            ap (ArgumentParser): The parser
        """
        ap.add_argument('--question_model', default='facebook/dpr-question_encoder-multiset-base', help='DPR question model')
        ap.add_argument('--context_model', default='facebook/dpr-ctx_encoder-multiset-base', help='DPR context model')
        ap.add_argument('--dpr_cache', default='/home/lyu/pretrained/dpr/', help='DPR cache dir')
        ap.add_argument('--model_dim', type=int, default=768, help='DPR output dimension')
        ap.add_argument('--dropout', type=float, default=0.1, help='Dropout percentage')
        ap.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
        ap.add_argument('--loss_margin', type=float, default=0.2, help='Margin for pairwise loss')
        ap.add_argument('--batch_size', type=int, default=32, help='Batch size')
        ap.add_argument('--warmup_steps', type=int, default=1000, help='Number of warmup steps')
        ap.add_argument('--freeze_bert', action='store_true', help='Do not update any weights of BERT (only train the classification layer)')
        ap.add_argument('--training_mode', choices=['pointwise', 'pairwise'], default='pairwise', help='Training mode')
        ap.add_argument('--rr_k', type=int, default=10, help='Compute MRR@k (validation)')
        ap.add_argument('--num_workers', type=int, default=16, help='Number of DataLoader workers')

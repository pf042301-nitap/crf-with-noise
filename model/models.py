# file name: models.py
import os
import pickle
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedTokenizerBase

# Import CRF layer
try:
    from crf import CRFLayer
except ImportError:
    CRFLayer = None


def js_div(P, Q):
    # Add epsilon to avoid numerical issues
    eps = 1e-10
    P = P + eps
    Q = Q + eps
    
    # Normalize to sum to 1
    P = P / P.sum(dim=-1, keepdim=True)
    Q = Q / Q.sum(dim=-1, keepdim=True)
    
    M = 0.5 * (P + Q)
    
    # Compute KL divergence with log_softmax for numerical stability
    kl_pm = F.kl_div(F.log_softmax(P, dim=-1), M, reduction='batchmean')
    kl_qm = F.kl_div(F.log_softmax(Q, dim=-1), M, reduction='batchmean')
    
    return 0.5 * (kl_pm + kl_qm)


class BlackBoxPredictor(nn.Module):
    def __init__(self, num_labels:int, model:str, freeze_encoder:bool, use_crf:bool = False):
        super().__init__()
        self.num_labels = num_labels
        self.use_crf = use_crf
        self.encoder = AutoModel.from_pretrained(model)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Token predictor for emission scores
        self.token_predictor = nn.Sequential(
            nn.Dropout(self.encoder.config.hidden_dropout_prob),
            nn.Linear(self.encoder.config.hidden_size, 2 if use_crf else 1)  # 2 tags for CRF, 1 for attention
        )
        
        # CRF layer for structured prediction
        if use_crf and CRFLayer is not None:
            self.crf = CRFLayer(num_tags=2, batch_first=True)
        elif use_crf:
            raise ImportError("CRFLayer not available. Make sure crf.py is in your path.")
        
        # Classification head
        self.predictor = nn.Sequential(
            nn.Dropout(self.encoder.config.hidden_dropout_prob),
            nn.Linear(self.encoder.config.hidden_size, self.num_labels)
        )

    def forward(self, input_ids, token_type_ids, attention_mask, return_crf_scores=False):
        # get contextualized embeddings from a transformer-based encoder
        outputs = self.encoder(
            input_ids = input_ids,
            token_type_ids = token_type_ids,
            attention_mask = attention_mask
        )
        
        # get hidden states without the CLS token
        hidden_states_no_cls = outputs[0][:, 1:, :] # (batch_size, seq_len-1, hidden_size)
        
        if self.use_crf:
            # use token classification head to generate emission scores for CRF
            emission_scores = self.token_predictor(hidden_states_no_cls) # (batch_size, seq_len-1, 2)
            
            # create mask for non-padding tokens
            mask = (input_ids[:, 1:] != 0) & (attention_mask[:, 1:] == 1)
            
            # decode using CRF
            crf_tags = self.crf.decode(emission_scores, mask)  # (batch_size, seq_len-1)
            
            # convert to attention probabilities (using softmax on emission scores for rationale class)
            token_probs = F.softmax(emission_scores, dim=-1)[:, :, 1:2]  # probability of tag=1 (rationale)
            token_att = token_probs
            
            # convert CRF tags to mask format
            crf_tags_tensor = crf_tags.float().unsqueeze(-1)  # (batch_size, seq_len-1, 1)
            
            # Get context vector using token attention (not CRF tags)
            ctx_vec = torch.bmm(hidden_states_no_cls.transpose(1, 2), token_att).squeeze() # (batch_size, hidden_size)
            # return predicted labels, per token probabilities P(z|x)
            att_pred = F.softmax(self.predictor(ctx_vec), -1)
            
            if return_crf_scores:
                return att_pred, token_att, emission_scores, crf_tags_tensor
            return att_pred, token_att, crf_tags_tensor
        else:
            # use token classification head to generate token logits
            token_logits = self.token_predictor(hidden_states_no_cls).squeeze() # (batch_size, seq_len-1, 1)
            # use softmax to get attention over tokens
            token_att = F.softmax(token_logits, -1).unsqueeze(-1) # (batch_size, seq_len-1, 1)
        
            # generate context vector
            ctx_vec = torch.bmm(hidden_states_no_cls.transpose(1, 2), token_att).squeeze() # (batch_size, hidden_size)
            # return predicted labels, per token probabilities P(z|x)
            att_pred = F.softmax(self.predictor(ctx_vec), -1)
            
            if return_crf_scores:
                return att_pred, token_att, None, None
            return att_pred, token_att, None

    # def get_loss(self, att_pred, hard_pred, labels, proximity, crf_tags=None, emission_scores=None, mask=None):
    #     classification_loss = F.cross_entropy(att_pred, labels)
    #     proximity_loss = proximity * js_div(att_pred, hard_pred)
        
    #     total_loss = classification_loss + proximity_loss
        
    #     # Add CRF loss if using CRF
    #     if self.use_crf and crf_tags is not None and emission_scores is not None and mask is not None:
    #         # Ensure crf_tags has the right shape for loss computation
    #         crf_tags_for_loss = crf_tags.squeeze(-1).long()  # (batch_size, seq_len-1)
    #         crf_loss = self.crf(emission_scores, crf_tags_for_loss, mask)
    #         total_loss = total_loss + 0.1 * crf_loss  # Weighted CRF loss
        
    #     return total_loss
    
    def get_loss(self,
    att_pred,
    hard_pred,
    labels,
    proximity,
    emission_scores=None,
    token_att=None,
    mask=None,
    lambda_sparse=0.01,
    lambda_entropy=0.05
    ):
        """
        Weakly-supervised rationale learning loss
        """

        # (1) Task / label prediction loss
        task_loss = F.cross_entropy(att_pred, labels)

        # (2) Sufficiency / proximity loss
        suff_loss = proximity * js_div(att_pred, hard_pred)

        total_loss = task_loss + suff_loss

        # (3) Sparsity loss (encourage short rationales)
        if token_att is not None:
            sparse_loss = token_att.mean()
            total_loss += lambda_sparse * sparse_loss

        # (4) CRF entropy loss (NO tag supervision)
        if self.use_crf and emission_scores is not None and mask is not None:
            probs = F.softmax(emission_scores, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            entropy = (entropy * mask).sum() / mask.sum()
            total_loss += lambda_entropy * entropy

        return total_loss

    
    def get_classification_prediction(self, hidden_states_cls):
        """Get classification prediction from CLS token."""
        return F.softmax(self.predictor(hidden_states_cls), -1)


class RationalePredictor(nn.Module):
    def __init__(self, num_labels:int, model:str, freeze_encoder:bool):
        super().__init__()
        self.num_labels = num_labels
        self.encoder = AutoModel.from_pretrained(model)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.predictor = nn.Sequential(
            nn.Dropout(self.encoder.config.hidden_dropout_prob),
            nn.Linear(self.encoder.config.hidden_size, self.num_labels)
        )

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.encoder(
            input_ids = input_ids,
            token_type_ids = token_type_ids,
            attention_mask = attention_mask
        )
        return F.softmax(self.predictor(outputs[1]), -1)

    def get_loss(self, att_pred, hard_pred, labels, proximity):
        classification_loss = F.cross_entropy(hard_pred, labels)
        proximity_loss = proximity * js_div(att_pred, hard_pred)
        return classification_loss + proximity_loss


@dataclass
class RationaleExtractor:
    tokenizer: PreTrainedTokenizerBase
    device: str

    def extract_from_mask(self, batch, hard_mask, max_length=512):
        # Add CLS tokens
        hard_mask = torch.cat((torch.ones_like(hard_mask[:, 1]).unsqueeze(-1), hard_mask), 1).squeeze() # (batch_size, seq_len - CLS, 1) -> (batch_size, seq_len)
        # Mask PAD tokens
        hard_mask.masked_fill_(batch.reviews_tokenized.input_ids == self.tokenizer.pad_token_id, False)
        # Unmask SEP tokens - because we are taking rationale batches as is
        hard_mask.masked_fill_(batch.reviews_tokenized.input_ids == self.tokenizer.sep_token_id, True)
        # Get max seq length to pad to
        max_len = min(int(hard_mask.sum(dim = 1).max().item()), max_length)
        # Extract rationales and pad
        rationale_ids = []
        attention_mask = []
        for ids, mask in zip(batch.reviews_tokenized.input_ids, hard_mask):
            rationale_ids.append(ids.masked_select(mask).unsqueeze(0))
            attention_mask.append(torch.ones_like(rationale_ids[-1], device = self.device))
            
            # Truncate if necessary
            if rationale_ids[-1].shape[1] > max_len:
                rationale_ids[-1] = rationale_ids[-1][:, :max_len]
                attention_mask[-1] = attention_mask[-1][:, :max_len]
            
            # pad if necessary
            if rationale_ids[-1].shape[1] < max_len:
                pad = torch.full(
                    size = (1, max_len - rationale_ids[-1].shape[1]),
                    fill_value = self.tokenizer.pad_token_id,
                    device = self.device
                )
                rationale_ids[-1] = torch.cat((rationale_ids[-1], pad), 1)
                attention_mask[-1] = torch.cat((attention_mask[-1], torch.zeros_like(pad)), 1)

        # to tensors
        rationale_ids = torch.cat(rationale_ids, 0).to(self.device)
        attention_mask = torch.cat(attention_mask, 0).to(self.device)
        token_type_ids = torch.zeros_like(rationale_ids).to(self.device)
        return {'input_ids': rationale_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}


    def __call__(self, batch, hard_mask):
        tokenized_rationales = self.extract_from_mask(batch, hard_mask)
        tokenized_remainders = self.extract_from_mask(batch, ~hard_mask)
        replacement_ratio = 0
        return tokenized_rationales, tokenized_remainders, replacement_ratio


@dataclass
class BaseNoisyRationaleExtractor(RationaleExtractor):
    data_path: str
    seed: Optional[int] = None

    def __post_init__(self):
        self.load_scored_vocab()
        self.rng = np.random.default_rng(self.seed)

    def load_scored_vocab(self):
        with open(os.path.join(self.data_path, "word_statistics", "scored_vocab.pkl"), "rb") as f:
            self.vocab, self.scores = pickle.load(f)

    @abstractmethod
    def extract_rationale(self, review, indices, replacement_mask):
        pass

    def extract_remainder(self, review, remainder_indices):
        return [review[i] for i in remainder_indices]

    def batch_tokenize(self, text, max_length=512):
        return self.tokenizer(text = text, is_split_into_words = True, padding = True, 
                             truncation = True, max_length = max_length, return_tensors = 'pt').to(self.device)

    def extract_from_mask_with_replacement(self, batch, hard_mask):
        # Mask PAD tokens
        hard_mask = hard_mask.squeeze() # (batch_size, seq_len - CLS, 1) -> (batch_size, seq_len - CLS)
        hard_mask.masked_fill_(batch.reviews_tokenized.input_ids[:,1:] == self.tokenizer.pad_token_id, False) # (batch_size, seq_len - CLS, 1)
        # Mask SEP tokens - because we are retokenizing rationale batches
        hard_mask.masked_fill_(batch.reviews_tokenized.input_ids[:,1:] == self.tokenizer.sep_token_id, False) # (batch_size, seq_len - CLS, 1)
        # Extract rationales and pad
        rationales = []
        replacement_ratio_sum = 0
       
        for i, (review, replacement_probs, mask) in enumerate(zip(batch.reviews, batch.replacement_probs, hard_mask)):
          
            # Masked select
            indices = [x for x, is_masked in zip(batch.reviews_tokenized.word_ids(i), mask) if x is not None and is_masked]
            # Unique Consecutive
            indices = [x for i, x in enumerate(indices) if i == 0 or x != indices[i - 1]]
            # If no valid indices (selected tokens do not map to words such as
            # selecting only CLS/SEP/PAD tokens), select the first word
            if len(indices) == 0:
                indices = [0]
            # Sample replacement mask
            replacement_mask = self.rng.binomial(n = 1, p = replacement_probs[indices])
            # Extract rationale
            rationale, replacement_ratio = self.extract_rationale(
                review = review,
                indices = indices,
                replacement_mask = replacement_mask
            )
            replacement_ratio_sum += replacement_ratio
            rationales.append(rationale)

        average_replacement_ratio = replacement_ratio_sum / len(batch.reviews)

        tokenized_rationales = self.batch_tokenize(rationales, max_length=512)
        return tokenized_rationales, average_replacement_ratio

    def __call__(self, batch, hard_mask):
        tokenized_rationales, replacement_ratio = self.extract_from_mask_with_replacement(batch, hard_mask)
        tokenized_remainder = self.extract_from_mask(batch, ~hard_mask)
        return tokenized_rationales, tokenized_remainder, replacement_ratio


@dataclass
class RandomNoisyRationaleExtractor(BaseNoisyRationaleExtractor):
    def __post_init__(self):
        super().__post_init__()
        self.vocab = np.array(self.vocab)

    def extract_rationale(self, review, indices, replacement_mask):
        rationale = np.array(review, dtype = self.vocab.dtype)[indices]
        rationale[replacement_mask == 1] = self.rng.choice(self.vocab, size = replacement_mask.sum(), p = self.scores)
        return rationale.tolist(), replacement_mask.sum()/replacement_mask.shape[0]



@dataclass
class RationaleExtractorFactory:
    tokenizer: PreTrainedTokenizerBase
    device: str
    data_path: Optional[str] = None
    seed: Optional[int] = None

    def create_extractor(self, inject_noise):
        if inject_noise:
            return RandomNoisyRationaleExtractor(
                tokenizer = self.tokenizer,
                device = self.device,
                data_path = self.data_path,
                seed = self.seed
            )
        return RationaleExtractor(
            tokenizer = self.tokenizer,
            device = self.device
        )


@dataclass
class TopkSelector:
    sparsity: float
    max_length: int
    pad_token_id: int
    device: str

    # gets empty mask
    def get_mask(self, token_att):
        return torch.zeros_like(token_att, dtype=torch.bool, requires_grad=False, device=self.device)

    # gets k = how many tokens to select
    def get_k(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_lens = torch.sum(input_ids != self.pad_token_id, dim = 1)
        seq_lens[seq_lens > self.max_length - 1] = self.max_length - 1 # - CLS token
        k = torch.round(seq_lens * self.sparsity).type(torch.long)
        k[k < 1] = 1 # select at least 1 token
        return k

    def __call__(self, token_att: torch.Tensor, input_ids: torch.Tensor, crf_tags: Optional[torch.Tensor] = None) -> torch.Tensor:
        hard_mask = self.get_mask(token_att)
        k = self.get_k(input_ids)
        # batch-first
        for i in range(token_att.shape[0]):
            atts = token_att[i, :, :].squeeze().detach()
            _, indices = atts.topk(k = k[i], dim = -1)
            hard_mask[i, indices, :] = True
        return hard_mask


@dataclass
class TopkContiguousSpanSelector(TopkSelector):
    def __call__(self, token_att: torch.Tensor, input_ids: torch.Tensor, crf_tags: Optional[torch.Tensor] = None) -> torch.Tensor:
        hard_mask = self.get_mask(token_att)
        k = self.get_k(input_ids)
        # batch-first
        for i in range(token_att.shape[0]):
            atts = token_att[i, :, :].unsqueeze(0).swapdims(2, 1)
            filter = torch.ones((1, 1, k[i]), device = self.device)
            start = F.conv1d(atts, filter).argmax()
            hard_mask[i, start:start + k[i], :] = True
        return hard_mask


@dataclass
class CRFSelector:
    sparsity: float
    max_length: int
    pad_token_id: int
    device: str

    def __call__(self, token_att: torch.Tensor, input_ids: torch.Tensor, crf_tags: Optional[torch.Tensor] = None) -> torch.Tensor:
        if crf_tags is not None:
            # Use CRF tags directly as hard mask (binary predictions)
            return crf_tags.bool()
        else:
            # Fallback to top-k selection based on attention
            hard_mask = torch.zeros_like(token_att, dtype=torch.bool, device=self.device)
            batch_size, seq_len_minus_one, _ = token_att.shape
            
            # Calculate sequence lengths (excluding CLS and PAD)
            seq_lens = torch.sum(input_ids[:, 1:] != self.pad_token_id, dim=1)
            seq_lens = torch.clamp(seq_lens, max=self.max_length - 1)
            
            # Calculate k based on sparsity
            k = torch.round(seq_lens * self.sparsity).type(torch.long)
            k = torch.clamp(k, min=1)
            
            # Select top-k tokens
            for i in range(batch_size):
                atts = token_att[i, :, :].squeeze().detach()
                valid_length = min(seq_lens[i].item(), seq_len_minus_one)
                if valid_length > 0:
                    atts_valid = atts[:valid_length]
                    _, indices = atts_valid.topk(k=min(k[i].item(), valid_length), dim=-1)
                    hard_mask[i, indices, :] = True
            
            return hard_mask


@dataclass
class SelectorFactory:
    sparsity: float
    max_length: int
    pad_token_id: int
    device: str

    def create_selector(self, selection_method, use_crf=False):
        if use_crf:
            return CRFSelector(
                sparsity = self.sparsity,
                max_length = self.max_length,
                pad_token_id = self.pad_token_id,
                device = self.device
            )
        elif selection_method == 'words':
            return TopkSelector(
                sparsity = self.sparsity,
                max_length = self.max_length,
                pad_token_id = self.pad_token_id,
                device = self.device
            )
        elif selection_method == 'span':
            return TopkContiguousSpanSelector(
                sparsity = self.sparsity,
                max_length = self.max_length,
                pad_token_id = self.pad_token_id,
                device = self.device
            )
        raise ValueError(f'Unknown selection method {selection_method}')
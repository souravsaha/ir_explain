from typing import List, Tuple
import torch
import torch.nn as nn
from captum.attr import LayerDeepLift
special_tokens = ['[UNK]','[SEP]', '[PAD]',  '[CLS]', '[MASK]']


class WrapperBert(nn.Module):
    # a wrapper model for bert reranker
    def __init__(self, model):
        super(WrapperBert, self).__init__()
        self.model = model
    def forward(self, input, atten, token_type):
        return self.model((input, atten, token_type))

# todo: drmm and dpr wrappers
# can't attribute to tokens for drmm, while the input is histogram.
class WrapperDrmm(nn.Module):
    pass  

class WrapperDpr(nn.Module):
    def __init__(self, model):
        super(WrapperDpr, self).__init__()
        self.model = model
    def forward(self, input_doc, input_q):
        return self.model((input_q, input_doc))


def construct_ref_input(orig_input: torch.tensor) -> torch.tensor:
    ref_input= orig_input.clone()
    select = torch.zeros_like(ref_input)
    for i in [101 ,102,100 ,103]:   # keep special_token
        special = ref_input == i
        select = torch.logical_or(select, special)
    return ref_input * select


def summerize_attributes(attributions: torch.tensor) -> torch.tensor:
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

def token_to_word_importance(tokens, importance):
    words, scores = [], []
    for i in range(len(tokens)):
        token = tokens[i]
        if token not in special_tokens:
            if token.startswith('##'):
                token_offset = words[-1]+token[2:]    # append this subtoken to the last subtoken.
                words[-1] = token_offset
                score_offset = max(scores[-1], importance[i])   # pick the maximum subtoken importance for a word
                scores[-1] = score_offset
            else:
                words.append(token)
                scores.append(importance[i])
        else:
            if token == '[PAD]':   # end of the text
                break   
            else:
                continue
    # sum up scores for the same words, frequent words might have higher overall scores
    words_importance = {}
    for w in list(set(words)):
        w_importance = [s for t, s in zip(words, scores) if t==w]
        words_importance[w] = sum(w_importance)/len(w_importance)  
    words_importance_sorted = sorted(words_importance.items(), key=lambda kv: kv[1], reverse=True)
    return words_importance_sorted

def attributes_dpr(model, tokenizer, query: str, doc: str, abs: bool=True, device=torch.device('cpu')) -> List[Tuple[str, float]]:
    input_q = tokenizer[0](query, return_tensors='pt')['input_ids'].to(device)
    input_d = tokenizer[1](doc, truncation=True, return_tensors='pt')['input_ids'].to(device)
    input_ref = construct_ref_input(input_d)

    wrapmodel = WrapperDpr(model).to(device).eval()
    wrapmodel.zero_grad()
    LDF = LayerDeepLift(wrapmodel, wrapmodel.model.context_encoder.ctx_encoder.bert_model.embeddings)
    importance = LDF.attribute(inputs=input_d, baselines=input_ref, additional_forward_args=(input_q)).detach()
    if abs: 
        importance = torch.abs(importance)
    importance = summerize_attributes(importance).cpu().numpy()
    tokens = tokenizer[1].convert_ids_to_tokens(input_d[0])
    word_importance = token_to_word_importance(tokens, importance)
    return word_importance

def attributes_bert(model, tokenizer, query: str, doc: str, abs: bool=True, device=torch.device('cpu')) -> List[Tuple[str, float]]:
    inputs = tokenizer([query], [doc], truncation=True)
    inputs_model = (torch.LongTensor(inputs['input_ids']).to(device), torch.LongTensor(inputs['attention_mask']).to(device), torch.LongTensor(inputs['token_type_ids']).to(device))
    inputs_ref = construct_ref_input(inputs_model[0])

    wrapmodel = WrapperBert(model).to(device).eval()
    wrapmodel.zero_grad()
    LDF = LayerDeepLift(wrapmodel, wrapmodel.model.bert.embeddings)
    importance = LDF.attribute(inputs=inputs_model[0],baselines=inputs_ref, additional_forward_args=(inputs_model[1], inputs_model[2])).detach()
    if abs: 
        importance = torch.abs(importance)
    importance = summerize_attributes(importance).cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    word_importance = token_to_word_importance(tokens, importance)
    return word_importance



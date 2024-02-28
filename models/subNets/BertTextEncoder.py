import os
import sys
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
__all__ = ['BertTextEncoder']
class BertTextEncoder(nn.Module):
    def __init__(self, language='en', use_finetune=False):
        super(BertTextEncoder, self).__init__()
        assert language in ['en', 'cn']
        tokenizer_class = BertTokenizer
        model_class = BertModel
        if language == 'en':
            self.tokenizer = tokenizer_class.from_pretrained('bert-base-uncased', do_lower_case=True)
            self.model = model_class.from_pretrained('bert-base-uncased')
        elif language == 'cn':
            self.tokenizer = tokenizer_class.from_pretrained('bert-base-uncased')
            self.model = model_class.from_pretrained('bert-base-uncased')
        self.use_finetune = use_finetune
    def get_tokenizer(self):
        return self.tokenizer
    def from_text(self, text):
        input_ids = self.get_id(text)
        with torch.no_grad(): last_hidden_states = self.model(input_ids)[0]  
        return last_hidden_states.squeeze()
    def forward(self, text):
        input_ids, input_mask, segment_ids = text[:,0,:].long(), text[:,1,:].float(), text[:,2,:].long()
        if self.use_finetune: last_hidden_states = self.model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)[0]  
        else: with torch.no_grad(): last_hidden_states = self.model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)[0]  
        return last_hidden_states
if __name__ == "__main__":
    bert_normal = BertTextEncoder()
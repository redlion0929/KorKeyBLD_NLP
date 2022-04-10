from transformers import BertModel, BertTokenizer, Trainer, TrainingArguments
import torch
from torch import nn


class keyBLD(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('klue/bert-base')
        self.linear = nn.Linear(768, 1)
        self.drop = nn.Dropout(0.1)

    #input으로는 tokenize된 것들이 들어옴
    def forward(self, input_ids, attention_mask, token_type_ids):

        input_ids = input_ids.view([-1, input_ids.shape[-1]])
        attention_mask = attention_mask.view([-1, input_ids.shape[-1]])
        token_type_ids = token_type_ids.view([-1, input_ids.shape[-1]])

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_states = outputs.last_hidden_state
        output_tensor = torch.stack([last_hidden_states[i][0] for i in range(len(last_hidden_states))])
        output = self.drop(output_tensor)
        output = self.linear(output)
        output = output.reshape(-1)
        return output
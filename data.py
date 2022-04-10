from torch.utils.data import Dataset
import torch

class keybld_dataset(Dataset):
    def __init__(self, dataset, tokenizer, seq_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        #data 토크나이징 하는 부분 -> 나온 input_ids를 inputs에 넣어놓음
        tokenized_input = self.tokenizer(data['input'], max_length=self.seq_length, truncation=True, padding='max_length', return_tensors='pt')

        inputs = {'input_ids': tokenized_input['input_ids'].long(),
                  'token_type_ids': tokenized_input['token_type_ids'].long(),
                  'attention_mask': tokenized_input['attention_mask'].float()}
        label = torch.LongTensor(data['label'])

        return inputs, label
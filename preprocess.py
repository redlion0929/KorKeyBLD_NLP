from datasets import Dataset, load_from_disk
from transformers import BertModel, BertTokenizer, Trainer, TrainingArguments
import torch
from math import log
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('---after get_data---')

class preprocess():
    def __init__(self, dataset):
        self.dataset = dataset
        self.passage = self.dataset['passage']
        self.query = self.dataset['question']
        self.label = self.dataset['label']
        self.seq_length = 128

    def tfidf(self, w, block, paragraph_set):
        # idf
        N = len(paragraph_set)
        df = 0
        for p in paragraph_set:
            df += w in p
        # tf * idf
        return (log(block.count(w) + 1)) * log(N + 1 / (df + 1))

    # 거의 sentence segmentation, 63토큰보다 클 경우 truncate
    def block_segmentation(self, paragraph_set):
        blocks = []
        from_idx = 0
        for idx, char in enumerate(paragraph_set):
            if char == '.' or char == '!':
                block = paragraph_set[from_idx:idx + 1]
                tokenized_block = tokenizer(block)['input_ids']
                l = len(tokenized_block)
                k = 1
                if l > 65:
                    while l > 65:
                        block = tokenizer.decode(tokenized_block[k: k + 63])
                        l = l - 63
                        k = k + 63
                        blocks.append(block)
                    blocks.append(tokenizer.decode(tokenized_block[k: len(tokenized_block)]))
                else:
                    blocks.append(block)
                from_idx = idx + 1
        return blocks

    def block_ranking(self, query, blocks, paragraph):
        try:
            scores = []
            for block in blocks:
                score = 0
                for word in query.split(' '):
                    score += self.tfidf(word, block, paragraph)
                scores.append(score)
            scores = torch.Tensor(scores)
            idx = torch.topk(scores, k=len(blocks))[1]
        except:
            idx = []
        return idx

    def make_input_sentence(self, query, blocks, rank_idx, max_seq_length):
        idx_selected = []
        last_block = ''
        max_seq_length = max_seq_length - len(tokenizer(query)['input_ids']) - 1
        for i in rank_idx:
            tokenized_block = tokenizer(blocks[i], return_tensors = 'pt').to(device)
            max_seq_length = max_seq_length - (len(tokenized_block['input_ids']) - 2)
            if max_seq_length < 0:
                last_block = blocks[i]
                break
            else:
                idx_selected.append(i)

        block_selected = [blocks[i] for i in sorted(idx_selected)]

        return query + '[SEP]' + (' '.join(block_selected) + last_block)

    def make_x(self):
        inputs = []
        for idx, passage in tqdm(enumerate(self.passage)):
            input = []
            query = self.query[idx]
            for i, k in enumerate(passage):
                blocks = self.block_segmentation(k)
                rank_idx = self.block_ranking(query, blocks, self.passage[idx])
                input_sentence = self.make_input_sentence(query, blocks, rank_idx, self.seq_length)
                input.append(input_sentence)
            inputs.append(input)
        return inputs

    def make_dataset(self):

        train_dataset = Dataset.from_dict({'input': self.make_x(),
                                            'label': self.label})

        return train_dataset

train_dataset = load_from_disk('korquad_train')
eval_dataset = load_from_disk('korquad_eval')


train_process = preprocess(train_dataset)
train_dataset = train_process.make_dataset()

eval_process = preprocess(eval_dataset)
eval_dataset = eval_process.make_dataset()

train_dataset.save_to_disk('korquad_train_128')
eval_dataset.save_to_disk('korquad_eval_128')
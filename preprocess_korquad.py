import transformers
from datasets import Dataset
from transformers import BertModel, BertTokenizerFast, Trainer, TrainingArguments
import torch
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import random
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

datasets = load_dataset('squad_kor_v1')

train_dataset = datasets['train']
eval_dataset = datasets['validation']
tokenizer = BertTokenizerFast.from_pretrained('klue/bert-base')

class preprocess_dataset():
    def __init__(self, dataset, tokenizer):
        self.contexts = []
        self.questions = []
        self.dataset = dataset
        self.tokenizer = tokenizer

        last_context = ''
        for idx, data in enumerate(self.dataset):
            if data['context'] != last_context:
                self.contexts.append(data['context'])
                last_context = data['context']
                self.questions.append([data['question']])
            else:
                self.questions[-1].append(data['question'])

        #bm25에 contexts 전달
        tokenized_corpus = [self.tokenizer(doc)['input_ids'] for doc in self.contexts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    #using bm25, return top_doc_index
    def get_negative_passage_bm25(self, question_row, question_column):
        tokenized_question = self.tokenizer(self.questions[question_row][question_column])['input_ids']
        top_doc = self.bm25.get_top_n(tokenized_question, self.contexts, n=32)
        top_doc_idx = [self.contexts.index(doc) for doc in top_doc]
        if question_row in top_doc_idx:
            top_doc_idx.remove(question_row)
            return top_doc_idx
        else:
            return -1

    def make_dataset(self):
        question_data = []
        passage_data = []
        label = []
        for row, questions in tqdm(enumerate(self.questions)):
            for col, question in enumerate(questions):
                top_idx = self.get_negative_passage_bm25(row, col)
                if top_idx == -1:
                    continue

                question_data.append(question)
                passage_data.append([])
                passage_data[-1].append(self.contexts[row])
                for i in top_idx:
                    passage_data[-1].append(self.contexts[i])

                #섞고
                random.shuffle(passage_data[-1])

                #찾아서 label에 넣기
                l = [0]*32
                l[passage_data[-1].index(self.contexts[row])] = 1
                label.append(l)


        #suffle
        index_list = list(range(len(question_data)))
        random.shuffle(index_list)

        questions = [question_data[idx] for idx in index_list]
        passages = [passage_data[idx] for idx in index_list]
        labels = [label[idx] for idx in index_list]


        return Dataset.from_pandas(pd.DataFrame({'question':questions,
                                                'passage':passages,
                                                'label':labels}))


train_dataset_process = preprocess_dataset(train_dataset, tokenizer)
eval_dataset_process = preprocess_dataset(eval_dataset, tokenizer)

train_dataset = train_dataset_process.make_dataset()
eval_dataset = eval_dataset_process.make_dataset()

train_dataset.save_to_disk('korquad_train')
eval_dataset.save_to_disk('korquad_eval')

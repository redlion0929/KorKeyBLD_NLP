import transformers
from transformers import BertModel, BertTokenizerFast, Trainer, TrainingArguments
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk, load_dataset, Dataset
from torch.utils.data import DataLoader
from pytorchltr.loss import PairwiseHingeLoss
import argparse
from attrdict import AttrDict
import os
import json
from model import keyBLD
from transformers import AdamW, get_linear_schedule_with_warmup
from data import keybld_dataset


class train_keyBLD(object):
    def __init__(self, args):
        self.args = args

        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:" + str(self.args.gpu_ids[0]) if torch.cuda.is_available() else "cpu")
        self.best_model = None
        self.best_val_loss = float('inf')

        self.build_model()
        self.build_dataloader()
        self.setup_training()


    def build_dataloader(self):
        self.train_dataset = keybld_dataset(Dataset.from_dict(load_from_disk(self.args.train_data_dir)[:10016]), self.tokenizer, self.args.seq_length)

        self.eval_dataset = keybld_dataset(Dataset.from_dict(load_from_disk(self.args.validation_data_dir)[:2016]), self.tokenizer, self.args.seq_length)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.cpu_workers,
            shuffle=True,
            drop_last=True
        )

        self.test_dataset = keybld_dataset(Dataset.from_dict(load_from_disk(self.args.validation_data_dir)[2017:4017]),
                                           self.tokenizer, self.args.seq_length)

        print(
            """
            # -------------------------------------------------------------------------
            #   BUILD DATALOADER DONE
            # -------------------------------------------------------------------------
            """
        )

    def build_model(self):
        self.tokenizer = BertTokenizerFast.from_pretrained('klue/bert-base')
        self.model = keyBLD().to(self.device)

        print(
            """
            # -------------------------------------------------------------------------
            #   BUILD MODEL DONE
            # -------------------------------------------------------------------------
            """
        )

    def setup_training(self):

        # =============================================================================
        #   optimizer / scheduler
        # =============================================================================

        self.iterations = len(self.train_dataset) // self.args.virtual_batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        self.scheduler = get_linear_schedule_with_warmup(
            #여기 건드려보기
            self.optimizer, num_warmup_steps=0.1*self.iterations,
            num_training_steps=self.iterations * self.args.num_epochs
        )

        self.loss_fn = PairwiseHingeLoss()

        print(
            """
            # -------------------------------------------------------------------------
            #   SETUP TRAINING DONE
            # -------------------------------------------------------------------------
            """
        )

    def train_one_epoch(self):
        self.model.train()
        cnt = 0
        for i, batch in tqdm(enumerate(self.train_dataloader)):

            input, label = batch

            input_ids = input["input_ids"]
            input_ids = input_ids.to(self.device)

            attention_mask = input["attention_mask"]
            attention_mask = attention_mask.to(self.device)

            token_type_ids = input["token_type_ids"]
            token_type_ids = token_type_ids.to(self.device)

            output = self.model(input_ids, attention_mask, token_type_ids)
            label = label.to(self.device)
            n = torch.Tensor([label.size(1)])
            n = n.to(self.device)

            loss = self.loss_fn(output.unsqueeze(0), label, n)
            loss.backward()
            if (cnt == self.args.virtual_batch_size) or (i ==(len(self.train_dataloader) -1)):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                cnt = 0

    def evaluate(self):
        print('--------Evaluate Start----------')
        self.model.eval()
        total_loss = 0.
        correct = 0
        correct_1=0
        with torch.no_grad():
            for batch in self.eval_dataset:
                input, label = batch

                # outputs의 길이는 40이다.
                input_ids = input["input_ids"]
                input_ids = input_ids.to(self.device)

                attention_mask = input["attention_mask"]
                attention_mask = attention_mask.to(self.device)

                token_type_ids = input["token_type_ids"]
                token_type_ids = token_type_ids.to(self.device)

                output = self.model(input_ids, attention_mask, token_type_ids)
                label = label.to(self.device)
                n = torch.Tensor([label.size(0)])
                n = n.to(self.device)

                loss = self.loss_fn(output.unsqueeze(0), label.unsqueeze(0), n)
                total_loss +=loss
                #정확도 계산
                values, indices = torch.topk(output, k=3, dim=-1)
                for idx in indices:
                    if label[idx] == 1:
                        correct += 1

                if label[indices[0]] == 1:
                    correct_1 += 1

        return total_loss / len(self.eval_dataset), correct /len(self.eval_dataset), correct_1/len(self.eval_dataset)

    def test(self):
        print('--------Test Start----------')
        self.model.eval()
        total_loss = 0.
        correct = 0
        correct_1=0
        with torch.no_grad():
            for batch in self.test_dataset:
                input, label = batch

                # outputs의 길이는 40이다.
                input_ids = input["input_ids"]
                input_ids = input_ids.to(self.device)

                attention_mask = input["attention_mask"]
                attention_mask = attention_mask.to(self.device)

                token_type_ids = input["token_type_ids"]
                token_type_ids = token_type_ids.to(self.device)

                output = self.model(input_ids, attention_mask, token_type_ids)
                label = label.to(self.device)
                n = torch.Tensor([label.size(0)])
                n = n.to(self.device)

                loss = self.loss_fn(output.unsqueeze(0), label.unsqueeze(0), n)
                total_loss +=loss
                #정확도 계산
                values, indices = torch.topk(output, k=3, dim=-1)
                for idx in indices:
                    if label[idx] == 1:
                        correct += 1

                if label[indices[0]] == 1:
                    correct_1 += 1

        return total_loss / len(self.test_dataset), correct /len(self.test_dataset), correct_1/len(self.test_dataset)

    def train(self):
        cnt = 0
        for e in tqdm(range(self.args.num_epochs)):
            print('Epoch:', e)
            if cnt>=5:
                print('early stopping at Epoch'+str(e))
                break
            self.train_one_epoch()
            val_loss, val_acc, val1_acc = self.evaluate()
            print('val_loss', val_loss)
            print('val_acc', val_acc)
            print('val_acc_about_top1', val1_acc)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model = self.model
                cnt = 0
            else:
                cnt+=1
        torch.save(self.best_model.state_dict(), self.args.save_dirpath + self.args.load_pthpath + 'korquad' + '.pth')

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="keyBLD-korean (PyTorch)")
    arg_parser.add_argument("--config_dir", dest="config_dir", type=str, default="config", help="Config Directory")
    arg_parser.add_argument("--config_file", dest="config_file", type=str, default="keybld",
                            help="Config json file")
    arg_parser.add_argument("--mode", dest="mode", type=str, default="test")
    parsed_args = arg_parser.parse_args()

    # Read from config file and make args
    with open(os.path.join(parsed_args.config_dir, "{}.json".format(parsed_args.config_file))) as f:
        args = AttrDict(json.load(f))

    # print("Training/evaluation parameters {}".format(args))
    keybld_model = train_keyBLD(args)
    if parsed_args.mode == 'train':
        keybld_model.train()
    else:
        keybld_model.model = keyBLD()
        keybld_model.model.load_state_dict(torch.load(args.save_dirpath + args.load_pthpath + 'korquad' + '.pth'))
        keybld_model.model = keybld_model.model.to(keybld_model.device)
        total_loss, top_3, top_1 = keybld_model.test()
        print("total_loss", total_loss)
        print("top_3", top_3)
        print("top_1", top_1)


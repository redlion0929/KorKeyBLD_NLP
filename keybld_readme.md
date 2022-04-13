## KorKeyBLD_NLP

이 repository는 [KeyBLD: Selecting Key Blocks with Local Pre-ranking for Long Document Information Retrieval](https://dl.acm.org/doi/10.1145/3404835.3463083) 에서 소개된 모델인 KeyBLD 한국어 데이터셋을 사용하여 학습시킨 모델이다.

## Quick Links

  - [KeyBLD란?](#what_is_keybld)
  - [데이터 셋(Dataset)](#dataset)
  - [학습 결과](#result)
  - [Environment Setting](#environment_setting)
  - [Hyperparameter](#hyperparameter)
  - [Train](#train)
  - [Evaluation](#evaluation)

## KeyBLD란?
KeyBLD란 BERT 기반 cross-encoder 구조를 사용하여 하나의 query 대해 가장 적절한 document를 찾아주는 모델이다.
Language Model은 입력으로 받을 수 있는 토큰의 개수에 제한이 되어있기 때문에 KeyBLD에서는 입력으로 전체 document가 아닌 요약된 document를 받는다.
document 속에서 query 관련된 내용(key block)을 선택한 후 query와 같이 넣어주어 각 document의 score를 구한 후 가장 score가 높은 document를 반환한다.
KeyBLD는 BERT와 간단한 알고리즘 만으로도 효율적인 document-ranking model을 만들 수 있음을 보였다.

본 repository에서는 이러한 원리의 KeyBLD를 직접 구현해보고, 한국어 데이터 셋을 가지고 fine-tuning하였다. 
학습에는 [KorQuAD 1.0](https://korquad.github.io/KorQuad%201.0/) 을 사용하였다. 

## 데이터 셋(Dataset)

KorQuAD 1.0 데이터셋을 사용하였으며,데이터 셋을 처리할 때 아래와 같은 방법을 사용하였다.

1. BM25를 통해 전체 passage중에 query 가장 관련있는 negative passage (hard negative)를 31개 찾음
2. 하나의 데이터는 (query, positive passage, 31개의 negative passage)로 구성됨
3. 각 passage에 대해 Block Selection을 거쳐 하나의 passage에 대해서 [CLS] query [SEP] block1 block2 ... [SEP] 형태를 만든다.
이때 query와 관련성이 높을수록 block 뒤에 붙은 숫자가 작다.
4. 32개의 [CLS] query [SEP] block1 block2 .. [SEP] 가 모여 하나의 데이터를 구성한다. 
즉, 하나의 데이터는 query 하나와 32개의 passage에 대한 정보를 가지고 있다고 할 수 있다.

이때 train dataset으로 KorQuAD train set에서 처음 10016개, validation set, test set은 KorQuAD validation set에서 0-2016, 2017-4017번째 데이터를 사용하였다. 

관련 내용은 `preprocess_korquad.py`, `preprocess.py`에 있다.

## 학습 결과

평가에는 2000개의 문제로 구성된 test set을 사용하였다.

각 query에 대해 32개의 passage 중에서 top 1, top 3 passage를 찾는 실험을 하였으며 성능은 아래 나타나 있다.

| Top k | Accuracy |
|:------|:--------:|
| Top 3 |   98.2   |
| Top 1 |   92.0   |

| Loss     | Value  |
|:---------|:------:|
| Test set | 0.7397 |

1개의 positive passage와 31개의 hard negative 사이에서 ranking accuracy가 상당히 높은 것을 확인할 수 있다.

## Environment Setting

아래와 같은 라이브러리를 사용하여 실험을 진행하였다.

| Module        | Version |
|:--------------|:-------:|
| pytorchltr    |  0.2.1  |
| transformers  | 4.17.0  |
| torch         |  1.7.1  |
| tqdm          | 4.63.0  |
| datasets      | 1.18.3  |
| attrdict      |  2.0.1  |
| rank_bm25     |  0.2.2  |

## Hyperparameter

| Parameter     |          Value          |
|:--------------|:-----------------------:|
| Batch size    |   16 (virtual batch)    |
| Cpu workers   |            2            |
| Epochs        |           16            |
| Learning rate |          5e-3           |
| Optimizer     |          Adam           |
| Scheduler     | Linear warmup scheduler |
| Seq_length    |           128           |

scheduler의 경우 전체 step의 10%까지 warmup을 진행하였다.

학습 중 validation loss가 가장 적은 모델을 최종 모델로 선택하였다.
또한 5번 연속으로 validation loss가 떨어지지 않는 경우, early stopping으로 학습을 종료하였다.

KorQuAD의 특성상 512토큰을 넘는 데이터가 별로 없기 때문에 Block Selection의 효과를 보기 위해 input sequence length를 128로 설정하였다.

## Train

Train을 하기 위해서는, 다음과 같은 명령어를 사용하면 된다.

 `python train.py --mode train`

## Evaluation

Evaluation을 하기 위해서는, 다음과 같은 명령어를 사용하면 된다.

 `python train.py --mode evaluation`

이 외에 mode 인자로 train 외에 다른 문자가 들어와도 evaluation이 진행된다.

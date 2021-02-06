# CLI(Command Line Interface) 환경에서 돌릴 수 있도록 argparse 활용
# 주피터 노트북에서 사용하려면 config를 메인 함수를 불러 넘겨주면 실행될 것.

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from tc.trainer import Trainer
from tc.data_loader import DataLoader

from tc.models.rnn import RNNClassifier
from tc.models.cnn import CNNClassifier

def define_argparser():
    '''
    Define argument parser to set hyper-parameters.
    '''
    p = argparse.ArgumentParser()
    
    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_fn', required=True) # 학습에 쓸 트레이닝 파일

    p.add_argument('--gpu_id', type=int, default=-1) # cpu 같은 경우 보통 -1로 둔다
    p.add_argument('--verbose', type=int, default=2) # 수다스러움 정도. 얼마나 로그 많이 출력해줄거냐
    # 0 아무것도 안 하고, 1은 epoch이 끝날 떄마다, 2는 iteration마다

    p.add_argument('--min_vocab_freq', type=int, default=5) # 최소 다섯 번 이상 나온 단어만 학습하자
    p.add_argument('--max_vocab_size', type=int, default=999999)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=10)

    p.add_argument('--word_vec_size', type=int, default=256)
    p.add_argument('--dropout', type=float, default=.3)

    p.add_argument('--max_length', type=int, default=256)

    p.add_argument('--rnn', action='store_true')
    p.add_argument('--hidden_size', type=int, default=512)
    p.add_argument('--n_layers', type=int, default=4)

    p.add_argument('--cnn', action='store_true')
    p.add_argument('--use_batch_norm', action='store_true')
    p.add_argument('--window_sizes', type=int, nargs='*', default=[3, 4, 5])
    p.add_argument('--n_filters', type=int, nargs='*', default=[100, 100, 100])
    
    # rnn은 좀 더 컨텍스트적인 것을 보고, CNN은 구나 단어, 절 등의 패턴이 있는지 없는지를 보는 경향이 있다.
    # 두 모델의 장단이 있음 -> 앙상블 사용
    # 하나만 켜져 있으면 하나만 사용하지만 둘 다 켜져 있으면 둘 다 사용.
    
    config = p.parse_args()
    
    return config

def main(config):
    loaders = DataLoader(train_fn=config.train_fn,
                         batch_size=config.batch_size,
                         min_freq=config.min_vocab_freq,
                         max_vocab=config.max_vocab_size,
                         device=config.gpu_id
                        )
    
    print(
        '|train| =', len(loaders.train_loader.dataset), # 문장 개수 출력
        '|valid| =', len(loaders.valid_loader.dataset),
    )
    
    vocab_size = len(loaders.text.vocab)
    n_classes = len(loaders.label.vocab)
    print('|vocab| =', vocab_size, '|classes| =', n_classes)
    
    if config.rnn is False and cofig.cnn is False:
        raise Exception('You need to specify an architecture to train. (--rnn or --cnn)')
        # CNN, RNN이 둘 다 꺼져 있으면 에러 출력
    
    # RNN이 켜져 있을 경우
    if config.rnn:
        # Declare model and loss.
        model = RNNClassifier(
            input_size=vocab_size,
            word_vec_size=config.word_vec_size,
            hidden_size=config.hidden_size,
            n_classes=n_classes,
            n_layers=config.n_layers,
            dropout_p=config.dropout,
        )
        optimizer = optim.Adam(model.parameters())
        crit = nn.NLLLoss() # RNN 모델 로그 소프트맥스를 썼기 때문에 크로스 엔트로피 대신 NLL을 로스함수로 사용
        print(model)
        
        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)
            
        rnn_trainer = Trainer(config)
        rnn_model = rnn_trainer.train(
            model,
            crit,
            optimizer,
            loaders.train_loader,
            loaders.valid_loader
        ) # rnn best model

    # CNN이 켜져 있을 경우
    if config.cnn:
        # Declare model and loss.
        model = CNNClassifier(
            input_size=vocab_size,
            word_vec_size=config.word_vec_size,
            n_classes=n_classes,
            use_batch_norm=config.use_batch_norm,
            dropout_p=config.dropout,
            window_sizes=config.window_sizes,
            n_filters=config.n_filters,
        )
        optimizer = optim.Adam(model.parameters())
        crit = nn.NLLLoss()
        print(model)
        
        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)
            
        cnn_trainer = Trainer(config)
        cnn_model = cnn_trainer.train(
            model,
            crit,
            optimizer,
            loaders.train_loader,
            loaders.valid_loader
        )
        
    torch.save({
        'rnn': rnn_model.state_dict() if config.rnn else None, # rnn이 켜져 있으면 state_dict() 저장
        'cnn': cnn_model.state_dict() if config.cnn else None,
        'config': config,
        'vocab': loaders.text.vocab,
        'classes': loaders.label.vocab,
    }, config.model_fn) # 딕셔너리를 config.model_fn 이름으로 저장
    
if __name__ == '__main__':
    config = define_argparser()
    main(config) # 아규먼트 받아서 메인 함수로 넘겨준다
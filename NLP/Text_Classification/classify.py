import sys
import argpass

import torch
import torch.nn as nn
from torchtext import data

from TC.models.rnn import RNNClassifier
from TC.models.cnn import CNNClassifier

def define_argparser():
    '''
    Define argument parser to take inference using pre-trained model.
    '''
    
    p = argparse.ArgumentParser()
    
    p.add_argumnet('--model_fn', required=True) # 어떤 모델 파일 불러올지
    p.add_argumnet('--gpu_id', type=int, default=-1)
    p.add_argumnet('--batch_size', type=int, default=256)
    p.add_argumnet('--top_k', type=int, default=1) # 확률이 높은 하나 이상의 클래스 후보들을 보여줄 수 있다.
    
    config = p.parse_args()
    
    return config
    
def read_text():
    '''
    Read text from standard input for inference.
    표준 입출력: 버퍼를 사용해 값을 프로그램에 전달.
    '''
    lines = []
    
    for line in sys.stdin:
        if line.strip() != '':
            lines += [line.strip().split(' ')]
            
    return lines

def define_field():
    '''
    To avoid use DataLoader class, just declare dummy fields.
    With those fields, we can restore mapping table between words and indice.
    '''
    return (
        data.Field(
            use_vocab=True,
            batch_first=True,
            include_lengths=False,
        ),
        data.Field(
            sequential=False,
            use_vocab=True,
            unk_toke=None,
        )
    )

def main(config):
    saved_data = torch.load(
        config.model_fn,
        map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
    )
    
    train_config = saved_data['config']
    rnn_best = saved_data['rnn']
    cnn_best = saved_data['cnn']
    vocab = saved_data['vocab']
    classes = saved_data['classes']
    
    vocab_size = len(vocab)
    n_classes = len(classes)
    
    text_field, label_field = define_field()
    text_filed.vocab = vocab
    label_field.vocab = classes
    
    lines = read_text()
    
    with torch.no_grad():
        # Converts string to list of index
        x = text_field.numericalize(
            text_field.pad(lines),
            devices='cuda: %d' % config.gpu_id if config.gpu_id >= 0 else 'cpu',
        )
        
        ensemble = []
        if rnn_best is not None:
            # Declare model and load pre-trained weights.
            model = RNNClassifier(
                input_size=vocab_size,
                word_vec_size=train_config.word_vec_size,
                hidden_size=train_config.hidden_size,
                n_classes=n_classes,
                n_layers=train_config.n_layers,
                dropout_p=train_config.dropout,
            )
            model.load_state_dict(rnn_best)
            ensemble += [model]
        if cnn_best is not None:
            # Declare model and load pre-trained weights.
            model = CNNClassifier(
                input_size=vocab_size,
                word_vec_size=train_config.word_vec_size,
                n_classes=n_classes,
                use_batch_norm=train_config.use_batch_norm,
                dropout_p=train_config.dropout,
                window_sizes=train_config.window_sizes,
                n_filters=train_config.n_filters,
            )
            model.load_state_dict(cnn_best)
            ensemble += [model]
            
        y_hats = []
        # Get prediction with iteration on ensemble.
        for model in ensemble:
            if config.gpu_id >= 0:
                model.cuda(config.gpu_id)
            # Don't forget turn-on evalutaion mode.
        # CUDA ("Compute Unified Device Architecture", 쿠다)는 그래픽 처리 장치(GPU)에서 수행하는
        # (병렬 처리) 알고리즘을 C 프로그래밍 언어를 비롯한 산업 표준 언어를 사용하여 작성할 수 있도록 하는 GPGPU 기술
        
        # GPGPU(General-Purpose computing on Graphics Processing Units, GPU 상의 범용 계산)는
        # 일반적으로 컴퓨터 그래픽스를 위한 계산만 맡았던 그래픽 처리 장치(GPU)를,
        # 전통적으로 중앙 처리 장치(CPU)가 맡았던 응용 프로그램들의 계산에 사용하는 기술
            model.eval()
            
            y_hat = []
            for idx in range(0, len(lines), config.batch_size): # batch_size로 스텝을 찍는 for문
                y_hat += [model(x[idx:idx + config.batch_size])]
            # Concatenate the mini-batch wise result
            y_hat = torch.cat(y_hat, dim=0)
            # |y_hat| = (len(lines), n_classes)
            
            y_hats += [y_hat]
        # Merge to one tensor for ensemble result and make probability from log-prob.
        y_hats = torch.stack(y_hats).exp() # 로그 확률에서 그냥 확률이 된다.
        # |y_hats| = (len(ensemble), len(lines), n_classes)
        y_hats = y_hats.sum(dim=0) / len(ensemble) # Get average
        # |y_hats| = (len(lines), n_classes)
        
        probs, indice = y_hats.cpu().topk(config.top_k)
        
        for i in range(len(lines)):
            sys.stdout.write('%s\t%s\n' % (
                ' '.join([classes.itos[indice[i][j]] for j in range(config.top_k)]), # itos: integer to string
                ' '.join(lines[i]))
            )

if __name__ == '__main__':
    config = define_argparser()
    main(config)
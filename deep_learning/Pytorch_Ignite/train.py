import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import ImageClassifier # model.py
from trainer import Trainer # trainer.py
from data_loader import get_loaders # data_loader.py

def define_argparser():
    p = argparse.ArgumentParser()
    
    p.add_argument('--model_fn', required=True) # model file name? 무조건 필요하다라고 돼 있는 것
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1) # GPU에서 돌릴건지
    
    p.add_argument('--train_ratio', type=float, default=.8)
    
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)
    # 얼마나 수다스럽게 할건지, 한 epoch이 아니라 한 iteration마다 정보를 보고 싶어 할 수도 있다. 사용자 설정에 따라 출력을 다르게 보여줌.
    
    config = p.parse_args()
    
    return config

def main(config):
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)
    
    train_loader, valid_loader, test_loader = get_loaders(config)
    
    print("Train:", len(train_loader.dataset))
    print("Valid:", len(valid_loader.dataset))
    print("Test:", len(test_loader.dataset))
    
    model = ImageClassifier(28**2, 10).to(device) # 입력은 784, 10개 클래스로 분류, GPU면 여기로 옮겨줘?
    optimizer = optim.Adam(model.parameters()) # model.parameters() 하면 모델 내 웨이트 파라미터들이 이터레이티브하게 나온다
    crit = nn.CrossEntropyLoss() # 분류를 해야하기 때문에 교차엔트로피 쓴다, 모델에서 소프트맥스로 끝나는 이유
    
    trainer = Trainer(config) # 이그나이트를 써서 짜놓은 메서드
    trainer.train(model, crit, optimizer, train_loader, valid_loader)
    
if __name__ == '__main__':
    config = define_argparser()
    main(config)
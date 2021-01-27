import torch

from torch.utils.data import Dataset, DataLoader

class MnistDataset(Dataset):
    
    def __init__(self, data, labels, flatten=True): # flatten을 해줄지 말지, FCL을 해주기 위해서는 플래튼 해야 함
        # 하지만 차후 CNN, RNN으로 모델을 개선해줄 예정이기 떄문에 설계할 때부터 확장성 있게 설계
        self.data = data
        self.labels = labels
        self.faltten = flatten
        
        super().__init__()
        
    def __len__(self):
        return self.data.size(0) # 첫번째 차원의 크기, 몇개의 샘플이 있나
    
    def __getitem__(self, idx):
        x = self.data[idx] # |x| = (28, 28) => (784,)
        y = self.labels[idx] # |y| = (1,)
        
        if self.flatten:
            x = x.view(-1) # flatten이 켜져 있으면 플래튼 해주면 됨
            
        return x, y
    
def load_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms
    
    dataset = datasets.MNIST(
        '../data', train=is_train,download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )
    
    x = dataset.data.float() / 255.
    y = dataset.targets
    
    if flatten:
        x = x.view(x.size(0), -1)
        
    return x, y

def get_loaders(config):
    x, y = load_mnist(is_train=True, flatten=False)
    
    train_cnt = int(x.size(0) * config.train_ration)
    valid_cnt = x.size(0) - train_cnt # 48000, 12000, MNIST 데이터 셋은 총 6만장의 트레인 셋과 1만장의 테스트셋으로 구성
    
    # Shuffle dataset to split into train/valid set.
    
    indices = torch.randperm(x.size(0)) # x.size(0) = 60000, 0~59999개의 랜덤한 수열 생성
    train_x, valid_x = torch.index_select( # |train_x| = (48000, 28, 28), |valid_x| = (12000, 28, 28)
        x, # |x| = (60000, 28, 28) 
        dim=0, # 60000 부분
        index=indices # 인덱스대로 셔플링해라
    ).split([train_cnt, valid_cnt], dim=0) # 그 결과에 대해서 48000과 12000개로 split해라
    train_y, valid_y = torch.index_select(
        y,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)
    
    # train, valid set DataLoader에 넣어주기
    train_loader = DataLoader(
        dataset=MnistDataset(train_x, train_y, flatten=True),
        batch_size=config.batch_size, # 배치 사이즈는 미리 지정해 놔야 한다.
        shuffle=True, # 트레인 셋은 무조건 셔플링 해줘야 함, False면 학습 제대로 안 됨
    )
    valid_loader = DataLoader(
        dataset=MnistDataset(valid_x, valid_y, flatten=True),
        batch_size=config.batch_size,
        shuffle=True, # 밸리드 셋은 해도 되고 안 해도 되고
    )
    
    test_x, test_y = load_mnist(is_train=False, flatten=False)
    test_loader = DataLoader(
        dataset=MnistDataset(test_x, test_y, flatten=True),
        batch_size=config.batch_size,
        shuffle=False, #보통 False로 둔다. 사이즈가 큰 태스크를 테스트할 때 유리하다. 같은 데이터 셋의 앞 부분만 서로 비교해줘도 되니까.
    )
    
    return train_loader, valid_loader, test_loader
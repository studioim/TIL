import torch
import torch.nn as nn

class ImageClassifier(nn.Module):
    
    def __init__(self,
                input_size,
                output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        super().__init__()
        # super(): 부모 클래스에 있는 함수의 코드를 자식 클래스에서 재사용할 떄 사용.
        # 여기서는 nn.Moudle 클래스의 이닛 함수의 내용을 그대로 상속하게 됨.
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, 500),
            nn.LeakyReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 400),
            nn.LeakyReLU(),
            nn.BatchNorm1d(400),
            nn.Linear(400, 300),
            nn.LeakyReLU(),
            nn.BatchNorm1(300),
            nn.Linear(300, 200),
            nn.LeakyReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, 100),
            nn.LeakyReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, output_size),
            nn.softmax(dim=-1),
        )
        
    def forward(self, x):
        # |x| = (batch_size, input_size)
        
        y = self.layers(x)
        # |y| = (batch_size, output_size)
        
        return y
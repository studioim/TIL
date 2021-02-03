import torch.nn as nn

class RNNClassifier(nn.Module): # nn.Module 상속
    
    def __init__( # __init__ 메소드
        self,
        input_size, # vocabulary_size, 코퍼스에 의존적인 하이퍼 파라미터
        word_vec_size, # 워드 임베딩 벡터가 몇차원으로 프로젝션 될거냐
        hidden_size, # bi-directional LSTM 쓸 건데 bi-directional LSTM의 히든 사이즈(hidden state와 cell state의 사이즈)는 어떻게 되냐
        n_classes, # 최종적으로 분류할 건데 클래스의 개수는?
        n_layers=4, # bi-directional LSTM 몇개 레이어 쌓을 것인가?
        dropout_p=.3, # 레이어와 레이어 사이에 드랍아웃은 얼마나?
    ):
        # 파라미터 어사인
        self.input_size = input_size # vocabulary_size
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        super().__init__()
        
        self.emb = nn.Embedding(input_size, word_vec_size) # 임베딩 레이어는 리니어 레이어, 어떤 차원의 입력(vocab size) 받아서 어떤 차원의 출력(몇차원의 워드 임베딩 벡터?)으로 내보낼지
        self.rnn = nn.LSTM(
            input_size=word_vec_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout_p,
            batch_first=True, # batch_size를 벡터의 가장 앞으로 (bs, length, hs * 2)
            bidirectional=True, # Non-autoregressive한 태스크를 수행하기 떄문에 사용 가능, 사용할 수 있는데 False로 놓을 이유가 거의 없다
        )
        self.generator = nn.Linear(hidden_size * 2, n_classes) # softmax 먹이기 전에 슬라이싱(차원축소) 과정, bidirectional LSTM의 경우 정방향 역방향 히든스테이트가 출력으로 나오기 때문에 *2를 해서 받아주고, number of classes로 차원축소
        # We use LogSoftmax + NLLLoss instead of Softmax + CrossEntropy
        self.activation = nn.LogSoftmax(dim=-1) # LogSoftmax를 쓰면 NLLLoss 사용, 속도도 좀 더 빠름, 맨 마지막 차원에 소프트맥스 씌워준다
        
    def forward(self, x): # forward 메소드
        # |x| = (batch_size, length)
        x = self.emb(x)
        # |x| = (batch_size, length, word_vec_size)
        x, _ = self.rnn(x) # LSTM은 출력이 두 개, 아웃풋과 마지막 타임 스텝의 히든 스테이트(이것만 쓴다)와 셀 스테이트
        # |x| = (batch_size, length, hidden_size * 2)
        y = self.activation(self.generator(x[:, -1]))
        # |y| = (batch_size, n_classes)
        
        return y
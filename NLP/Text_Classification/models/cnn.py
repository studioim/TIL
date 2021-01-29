import torch
import torch.nn as nn

class CNNClassifier(nn.Module):
    
    def __init__( # 오버라이딩
        self,
        input_size, # vocab size
        word_vec_size, # 단어 벡터의 사이즈
        n_classes,
        use_batch_norm=False, # batchnorm을 쓸건지 dropout을 쓸건지
        dropout_p=.5,
        window_sizes=[3, 4, 5], # 3단어, 4단어, 5단어짜리 text classifier
        n_filters=[100, 100, 100], # 3단어짜리 100개의 패턴, 4단어짜리 100개의 패턴, 5단어짜리 100개의 패턴
    ):
        self.input_size = input_size # vocab size
        self.word_vec_size = word_vec_size
        self.n_classes = n_classes
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p
        # window_size means that how many words a pattern covers.
        self.window_sizes = window_sizes
        # n_filters means that how may patterns to cover.
        self.n_filters = n_filters
        
        super().__init__()
        
        self.emb = nn.Embedding(input_size, word_vec_size)
        #Use nn.ModuleList to register each sub-modules.
        self.feature_extractors = nn.ModuleList() # array list 같은 애, 그냥 리스트 쓰면 등록이 안 됨.
        for window_size, n_filter in zip(window_sizes, n_filters):
            self.feature_extractors.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=1, # 인풋 채널. We only use one embedding layer.
                        out_channels=n_filter,
                        kernel_size=(window_size, word_vec_size),
                    ),
                    nn.ReLU(),
                    nn.BatchNorm2d(n_filter) if use_batch_norm else nn.Dropout(dropout_p),
                )
            )
        
        #An input of generator layer is max values from each filter.
        self.generator = nn.Linear(sum(n_filters), n_classes)
        #We use LogSoftmax + NLLLoss instead of Softmax + CrossEntorpy
        self.activation = nn.LogSoftmax(dim=-1)
        
    def forward(self, x): # 오버라이딩
        # |x| = (batch_size, length)
        x = self.emb(x)
        # |x| = (batch_size, length, word_vec_size)
        min_length = max(self.window_sizes) # 5
        
        if min_length > x.size(1): # 윈도우 크기가 문장 길이(length)보다 길면 모델이 죽는다.
            # Because some input does not long enough for maximum length of window size,
            # We add zero tensor for padding.
            pad = x.new(x.size(0), min_length - x.size(1), self.word_vec_size).zero_()
            # |pad| = (batch_size, min_length - length, word_vec_size)
            x = torch.cat([x, pad], dim=1)
            # |x| = (batch_size, min_length, word_vec_size)
            
            # In ordinary case of vision task, you may have 3 cahnnels on tensor,
            # but in this case, you would have just 1 channel,
            # which is added by 'unsqueeze' method in below:
            x =  x.unsqueeze(1)
            # |x| = (batch_size, 1, length, word_vec_size), CNN은 보통 4개 차원 입력으로 받는다(batch, channels, H, W)
            
            cnn_outs = []
            for block in self.feature_extractors:
                cnn_out = block(x)
                # |cnn_out| = (batch_size, n_filter, length - window_size + 1, 1)
                
                # In case of max pooling, we does not know the pooling size,
                # because it depends on the length of the sentence.
                # Therefore, we use instant fucntion using 'nn.functional' package.
                # This is the beauty of PyTorch. :)
                cnn_out = nn.functional.max_pool1d( # 맥스풀링은 학습하는 게 없다. 함수처럼 쓸 수 있다.
                    input=cnn_out.squeeze(-1),
                    kernel_size=cnn_out.size(-2) # length - window_size + 1
                ).squeeze(-1)
                # |cnn_out| = (batch_size, n_filter)
                cnn_outs += [cnn_out]
            
            # Merge output tensors from each convolution layer.
            cnn_outs = torch.cat(cnn_outs, dim=-1)
            # |cnn_outs| = (batch_size, sum(n_filters)) # sum(n_filters) = 100 + 100 + 100 = 300
            y = self.activation(self.generator(cnn_outs))
            # |y| = (batch_size, n_classes)
            
            return y
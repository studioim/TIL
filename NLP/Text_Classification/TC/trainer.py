# trainer 클래스를 바깥에 있는 train.py에서 불러서 학습시킨다.

from copy import deepcopy

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as torch_utils

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar
# epoch의 시작 iteration의 시작 등에 맞춰서 콜백함수를 등록할 수 있는 엔진을 제공하는 것이 이그나이트

from tc.utils import get_grad_norm, get_parameter_norm

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

class MyEngine(Engine):
    
    def __init__(self, func, model, crit, optimizer, config):
        # Ignite Engine does not have objects in below lines.
        # Thus, we assign class variables to access these object, during the procedure.
        self.model = model
        self.crit = crit # criterion(기준) = loss
        self.optimizer = optimizer
        self.config = config
        
        super().__init__(func) # Ignite Engine only needs function to run.
        
        self.best_loss = np.inf
        self.best_model = None
        
        self.device = next(model.parameters()).device
        
    @staticmethod
    def train(engine, mini_batch):
        # You have to reset the gradients of all model parameters
        # before to take another step in gradient descent.
        engine.model.train() # Because we assign model as class variable, we can easily access to it.
        # valdation에서 트레인 모드로 스위치
        engine.optimizer.zero_grad() # 모델 내 그레디언트 초기화
        
        x, y = mini_batch.text, mini_batch.label # 원핫벡터에 인덱스가 들어 있는 롱 텐서, 클래스 인덱스가 들어 있는 롱 텐서
        x, y = x.to((engine.device)), y.to(engine.device) # 모델과 같은 디바이스로 옮겨준다
        
        x = x[:, :engine.config.max_length] # |x| = [batch_size, length]무의미하게 긴 문장을 256까지로 잘라준다
        
        # Take feed_forward
        y_hat = engine.model(x) # |y hat| = (bs, |C|), 로그 확률값
        
        loss = engine.crit(y_hat, y) # NLLLoss
        loss.backward() # 역전파
        
        # 로스 말고 어큐러시도 보자.
        # Calculate accuracy only if 'y' is LongTensor,
        # which means that 'y' is one-hot representation.
        if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
            accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0)) # 맞는 개수, y.size(0) = 미니배치 사이즈
        else:
            accuracy = 0
            
        p_norm = float(get_parameter_norm(engine.model.parameters())) # 모델 내 파라미터들의 L2 놈의 합4
        # 보통 학습 진행되면서 파라미터 놈은 커지게 된다. 그걸 막는 방법으로 Weight Decay가 있다.
        # 파라미터 놈을 통해서 학습이 잘 진행되고 있는지를 어느 정도 확인할 수 있다. 계속해서 커져야 한다.
        g_norm = float(get_grad_norm(engine.model.parameters())) # 모델의 웨이트 파라미터의 그레디언트의 L2 놈의 합
        # 학습이 진행될수록 줄어드는 양상을 보이게 된다. 그레디언트는 점점 줄어들면서 수렴하니까.
        # 그레디언트가 커지면 학습이 안정적이지 않은 것. 발산한다고 생각할 수도 있다.
        
        # Take a step of gradient descent.
        engine.optimizer.step() # 그레디언트 디센트 수행
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            '|param|': p_norm,
            '|g_param|': g_norm,
        } # attach에서 이걸 그대로 받아서 출력해준다. 실시간으로 학습 현황 볼 수 있다.
    
    @staticmethod
    def validate(engine, mini_batch): # 역전파와 그레디언트 디센트 없을 뿐 거의 같다.
        engine.model.eval()
        
        with torch.no_grad():
            x, y = mini_batch.text, mini_batch.label
            x, y = x.to(engine.device), y.to(engine.device)
            
            x = x[:, :engine.config.max_length]
            
            y_hat = engine.model(x)
            
            loss = engine.crit(y_hat, y)
            
            if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
            else:
                accuracy = 0
        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
        }
    
    @staticmethod
    def attach(train_engine, validation_engine, verbose=VERBOSE_BATCH_WISE):
        # Attaching would be repeated for several metrics.
        # Thus, we can reduce the repeated codes by using this function.
        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine,
                metric_name,
            )
        
        training_metric_names = ['loss', 'accuracy', '|param|', '|g_param|']
        
        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)
            
        # If the verbosity is set, progress bar would be shown for mini-batch iterations.
        # Without ignite, you can use tqdm to implement progress bar.
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, training_metric_names)
            
        #If the verbosity is set, statistics would be shown after each epoch.
        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                print('Epoch {} - |param|={:.2e} |g_param|={:.2e} loss={:.4e} accuracy={:.4f}'.format(
                    engine.state.epoch,
                    engine.state.metrics['|param|'],
                    engine.state.metrics['|g_param|'],
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],                    
                ))
        
        validation_metric_names = ['loss', 'accuracy']
        
        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)
            
        # Do same things for validation engine.
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_engine, validation_metric_names)
            
        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                print('validation - loss={:.4e} accuracy={:.4f} best_loss={:.4e}'.format(
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],                    
                    engine.best_loss,
                ))
                
    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss: # If current epoch returns lower validation loss,
            engine.best_loss = loss # Update lowest validation loss.
            engine.best_model = deepcopy(engine.model.state_dict()) # Update best model weights.
            # 강의 기준으로 state_dict()를 딥카피하지 않으면 잘 안 되는 상황인데 파이토치 버전 바뀌면 될 수도.

    @staticmethod
    # 매 epoch마다 저장
    def save_model(engine, train_engine, config, **kwargs):
        torch.save(
            {
                'model': engine.best_model,
                'config': config,
                **kwargs
            }, config.model_fn
        )
        
            
class Trainer():
    
    def __init__(self, config):
        self.config = config
        
    def train(
        self,
        model, crit, optimizer,
        train_loader, valid_loader
    ):
        train_engine = MyEngine(  # 매 미니배치마다 실행
            MyEngine.train,
            model, crit, optimizer, self.config
        )
        validation_engine = MyEngine( # 매 미니배치마다 실행
            MyEngine.validate,
            model, crit, optimizer, self.config
        )
        
        # 단순히 통계 출력
        MyEngine.attach(
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )
        
        # 트레이닝 한 epoch이 끝날 때마다 밸리데이션도 한 epoch 돌려준다
        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)
        
        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # event
            run_validation, # function
            validation_engine, valid_loader, # arguments
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # event
            MyEngine.check_best, # function
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # event
            MyEngine.save_model, # function
            train_engine, self.config, # arguments
        )
        
        # 비로소 트레인 엔진에 돌린다
        train_engine.run(
            train_loader,
            max_epochs=self.config.n_epochs,
        )
        
        model.load_state_dict(validation_engine.best_model)
        
        return model
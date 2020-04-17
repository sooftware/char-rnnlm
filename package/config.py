class Config:
    def __init__(self,
                 use_cuda=True,
                 hidden_size=512,
                 dropout_p=0.5,
                 n_layers=4,
                 batch_size=32,
                 max_epochs=40,
                 lr=0.0001,
                 teacher_forcing_ratio=1.0,
                 seed=1,
                 max_len=428,
                 worker_num=1
                 ):
        self.use_cuda = use_cuda
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.seed = seed
        self.max_len = max_len
        self.worker_num = worker_num

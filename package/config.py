

class Config():
    def __init__(self,
                 use_cuda = True,
                 pack_by_length = True,
                 augment_ratio = 1.0,
                 hidden_size = 512,
                 wordvec_size = 256,
                 dropout_p = 0.5,
                 n_layers = 4,
                 batch_size = 32,
                 max_epochs = 40,
                 lr = 0.0001,
                 teacher_forcing = 1.0,
                 seed = 1,
                 max_len = 428,
                 worker_num=1
                 ):
        self.use_cuda = use_cuda
        self.pack_by_length = pack_by_length
        self.augment_ratio = augment_ratio
        self.hidden_size = hidden_size
        self.wordvec_size = wordvec_size
        self.dropout_p = dropout_p
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.teacher_forcing = teacher_forcing
        self.seed = seed
        self.max_len = max_len
        self.worker_num = worker_num
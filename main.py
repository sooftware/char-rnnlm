import os
import time
import torch
import torch.nn as nn
import queue
import math
from random import random
from torch import optim
from package.config import Config
from package.definition import char2id, logger, SOS_token, EOS_token, PAD_token
from package.data_loader import CustomDataset, load_corpus, CustomDataLoader
from package.loss import Perplexity
from package.trainer import supervised_train
from model import LanguageModel

# Character-level Recurrent Neural Network Language Model implement in Pytorch
# https://github.com/sooftware/char-rnnlm

if __name__ == '__main__':
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # if you use Multi-GPU, delete this line
    logger.info("device : %s" % torch.cuda.get_device_name(0))
    logger.info("CUDA is available : %s" % (torch.cuda.is_available()))
    logger.info("CUDA version : %s" % (torch.version.cuda))
    logger.info("PyTorch version : %s" % (torch.__version__))

    config = Config(
        use_cuda=True,
        hidden_size=512,
        dropout_p=0.5,
        n_layers=4,
        batch_size=16,
        max_epochs=40,
        lr=0.0001,
        teacher_forcing_ratio=1.0,
        seed=1,
        max_len=428,
        worker_num=1
    )

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    cuda = config.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    model = LanguageModel(
        n_class=len(char2id),
        n_layers=config.n_layers,
        hidden_size=config.hidden_size,
        dropout_p=config.dropout_p,
        max_length=config.max_len,
        sos_id=SOS_token,
        eos_id=EOS_token,
        device=device
    )
    model.flatten_parameters()
    model = nn.DataParallel(model).to(device)

    for param in model.parameters():
        param.data.uniform_(-0.08, 0.08)

    # Prepare loss
    weight = torch.ones(len(char2id)).to(device)
    perplexity = Perplexity(weight, PAD_token, device)
    optimizer = optim.Adam(model.module.parameters(), lr=config.lr)

    corpus = load_corpus('./data/corpus_df.bin')
    total_time_step = math.ceil(len(corpus) / config.batch_size)

    train_set = CustomDataset(corpus[:-10000], SOS_token, EOS_token, config.batch_size)
    valid_set = CustomDataset(corpus[-10000:], SOS_token, EOS_token, config.batch_size)

    logger.info('start')
    train_begin = time.time()

    for epoch in range(config.max_epochs):
        train_queue = queue.Queue(config.worker_num << 1)
        train_set.shuffle()

        train_loader = CustomDataLoader(train_set, train_queue, config.batch_size, 0)
        train_loader.start()

        train_loss = supervised_train(
            model=model,
            queue=train_queue,
            total_time_step=total_time_step,
            train_begin=train_begin,
            perplexity=perplexity,
            optimizer=optimizer,
            device=device,
            print_every=10,
            teacher_forcing_ratio=config.teacher_forcing_ratio,
            worker_num=config.worker_num
        )

        torch.save(model, "./data/epoch%s.pt" % str(epoch))
        logger.info('Epoch %d (Training) Loss %0.4f' % (epoch, train_loss))
        train_loader.join()

        valid_queue = queue.Queue(config.worker_num << 1)
        valid_loader = CustomDataLoader(valid_set, valid_queue, config.batch_size, 0)
        valid_loader.start()

        valid_loss = evaluate(model, valid_queue, perplexity, device)
        valid_loader.join()

        logger.info('Epoch %d (Evaluate) Loss %0.4f' % (epoch, valid_loss))
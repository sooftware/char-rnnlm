import os
import time
import torch
import logging, sys
import torch.nn as nn
import queue
import math
from random import random
from torch import optim
from torchtext import data
from src.config import Config
from src.data_loader import KoreanDataset, load_scripts, BaseDataLoader
from src.definition import char2id, logger, SOS_token, EOS_token, PAD_token
from src.lm import LanguageModel
from src.trainer import supervised_train


if __name__ == '__main__':
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # if you use Multi-GPU, delete this line
    logger.info("device : %s" % torch.cuda.get_device_name(0))
    logger.info("CUDA is available : %s" % (torch.cuda.is_available()))
    logger.info("CUDA version : %s" % (torch.version.cuda))
    logger.info("PyTorch version : %s" % (torch.__version__))

    config = Config(
        use_cuda=True,
        hidden_size=256,
        dropout_p=0.5,
        n_layers=5,
        batch_size=32,
        max_epochs=40,
        lr=0.0001,
        teacher_forcing=1.0,
        seed=1,
        max_len=428,
        worker_num=1
    )

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    cuda = config.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    model = LanguageModel(
        n_class=len(char2id),
        n_layers=4,
        wordvec_size=256,
        hidden_size=512,
        dropout_p=0.5,
        max_length=428,
        sos_id=SOS_token,
        eos_id=EOS_token,
        device=device
    )
    model.flatten_parameters()
    model = nn.DataParallel(model).to(device)

    optimizer = optim.Adam(model.module.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_token).to(device)

    scripts = load_scripts('./data/ko_dataset.csv', encoding='utf-8')
    total_time_step = math.ceil(len(scripts) / config.batch_size)

    train_dataset = KoreanDataset(scripts, SOS_token, EOS_token, config.batch_size)
    train_iterator = data.BucketIterator(train_dataset, batch_size=config.batch_size, device=device, shuffle=True)

    logger.info('start')
    train_begin = time.time()

    for epoch in range(config.max_epochs):
        train_queue = queue.Queue(2)
        train_dataset.shuffle()

        train_loader = BaseDataLoader(scripts, queue, config.batch_size, 0)
        train_loader.start()

        train_loss, train_cer = supervised_train(
            model=model,
            total_time_step=total_time_step,
            train_begin=train_begin,
            queue=train_queue,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            print_time_step=10,
            teacher_forcing_ratio=config.teacher_forcing,
            worker_num=config.worker_num
        )

        torch.save(model, "./data/epoch%s.pt" % str(epoch))
        logger.info('Epoch %d (Training) Loss %0.4f CER %0.4f' % (epoch, train_loss, train_cer))

        train_loader.join()



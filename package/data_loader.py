import threading
from torch.utils.data import Dataset
from package.definition import logger
from package.utils import get_label, get_input
import csv
import random
import pandas as pd
import torch
import math


class BaseDataLoader(threading.Thread):
    def __init__(self, dataset, queue, batch_size, thread_id):
        threading.Thread.__init__(self)
        self.dataset = dataset
        self.batch_size = batch_size
        self.queue = queue
        self.index = 0
        self.thread_id = thread_id
        self.dataset_count = dataset.count()

    def create_empty_batch(self):
        sequences = torch.zeros(0, 0, 0).to(torch.long)
        targets = torch.zeros(0, 0, 0).to(torch.long)

        sequence_lengths = list()
        target_lengths = list()

        return sequences, targets, sequence_lengths, target_lengths

    def run(self):
        logger.debug('loader %d start' % (self.thread_id))
        while True:
            items = list()

            for _ in range(self.batch_size):
                if self.index >= self.dataset_count:
                    break

                input, label = self.dataset.get_item(self.index)

                if input is not None:
                    items.append((input, label))

                self.index += 1

            if len(items) == 0:
                batch = self.create_empty_batch()
                self.queue.put(batch)
                break

            random.shuffle(items)

            batch = self.collate_fn(items)
            self.queue.put(batch)

        logger.debug('loader %d stop' % (self.thread_id))

    def count(self):
        return math.ceil(self.dataset_count / self.batch_size)


def _collate_fn(batch):
    """ functions that pad to the maximum sequence length """
    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    seq_lengths = [len(s[0]) for s in batch]
    target_lengths = [len(s[1]) for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = max_seq_sample.size(0)
    max_target_size = len(max_target_sample)

    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)

    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    from package.definition import PAD_token
    targets.fill_(PAD_token)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)

        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    return seqs, targets, seq_lengths, target_lengths


class BaseDataset(Dataset):
    def __init__(self, dataset, sos_id, eos_id, batch_size):
        self.dataset = dataset
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.batch_size = batch_size

    def get_item(self, index):
        input = get_input(self.dataset[index], self.sos_id)
        label = get_label(self.dataset[index], self.eos_id)

        return input, label

    def shuffle(self):
        random.shuffle(self.dataset)


def load_label(label_path, encoding='utf-8'):
    char2id = dict()
    id2char = dict()

    with open(label_path, 'r', encoding=encoding) as f:
        labels = csv.reader(f, delimiter=',')
        next(labels)

        for row in labels:
            char2id[row[1]] = row[0]
            id2char[int(row[0])] = row[1]

    return char2id, id2char


def load_dataset(filepath, encoding='utf-8'):
    dataset = list(pd.read_csv(filepath, encoding=encoding)['ko'])
    return dataset
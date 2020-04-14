import threading
import csv
import random
import pandas as pd
import torch
import math
import pickle
from torch.utils.data import Dataset
from package.definition import logger
from package.utils import get_label, get_input


class CustomDataLoader(threading.Thread):
    def __init__(self, dataset, queue, batch_size, thread_id):
        threading.Thread.__init__(self)
        self.collate_fn = _collate_fn
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
        logger.debug('loader %d start' % self.thread_id)
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

        logger.debug('loader %d stop' % self.thread_id)


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
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size).to(torch.long)
    targets = torch.zeros(batch_size, max_target_size).to(torch.long)

    from package.definition import PAD_token
    targets.fill_(PAD_token)
    seqs.fill_(PAD_token)

    for idx in range(batch_size):
        sample = batch[idx]
        tensor = sample[0]
        target = sample[1]

        seqs[idx].narrow(0, 0, len(tensor)).copy_(torch.LongTensor(tensor))
        targets[idx].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    return seqs, targets, seq_lengths, target_lengths


class CustomDataset(Dataset):
    def __init__(self, corpus, sos_id, eos_id, batch_size):
        self.corpus = corpus
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.batch_size = batch_size

    def get_item(self, index):
        input = get_input(self.corpus[index], self.sos_id)
        label = get_label(self.corpus[index], self.eos_id)

        return input, label

    def shuffle(self):
        random.shuffle(self.corpus)

    def count(self):
        return len(self.corpus)


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


def load_corpus(filepath):
    with open(filepath, 'rb') as f:
        corpus = pickle.load(f)
        corpus = list(corpus['id'])

        return corpus

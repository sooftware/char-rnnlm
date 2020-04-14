import Levenshtein as Lev
import torch


def get_label(script, eos_id):
    tokens = script.split()

    label = list()
    for token in tokens:
        label.append(int(token))
    label.append(int(eos_id))

    return label


def get_input(script, sos_id):
    tokens = script.split()

    label = list()
    label.append(int(sos_id))
    for token in tokens:
        label.append(int(token))

    return torch.LongTensor(label)


def label_to_string(labels, id2char, eos_id):
    if len(labels.shape) == 1:
        sentence = str()
        for label in labels:
            if label.item() == eos_id:
                break
            sentence += id2char[label.item()]
        return sentence

    elif len(labels.shape) == 2:
        sentences = list()
        for batch in labels:
            sentence = str()
            for label in batch:
                if label.item() == eos_id:
                    break
                sentence += id2char[label.item()]
            sentences.append(sentence)
        return sentences

    else:
        raise ValueError("shape Error !!")


def char_distance(target, y_hat):
    target = target.replace(' ', '')
    y_hat = y_hat.replace(' ', '')

    dist = Lev.distance(y_hat, target)
    length = len(target.replace(' ', ''))

    return dist, length

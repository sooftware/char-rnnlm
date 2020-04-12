import Levenshtein as Lev
import pandas as pd

def get_label(script, eos_id):
    tokens = script.split(' ')

    label = list()
    for token in tokens:
        label.append(int(token))
    label.append(int(eos_id))

    return label


def get_input(script, sos_id):
    tokens = script.split(' ')

    label = list()
    label.append(int(sos_id))
    for token in tokens:
        label.append(int(token))

    return label


def get_distance(targets, y_hats, id2char, eos_id):
    total_dist = 0
    total_length = 0

    for (target, y_hat) in zip(targets, y_hats):
        script = label_to_string(target, id2char, eos_id)
        pred = label_to_string(y_hat, id2char, eos_id)

        dist, length = char_distance(script, pred)

        total_dist += dist
        total_length += length

    return total_dist, total_length


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


def get_distance(targets, y_hats, id2char, eos_id):
    total_dist = 0
    total_length = 0

    for (target, y_hat) in zip(targets, y_hats):
        script = label_to_string(target, id2char, eos_id)
        pred = label_to_string(y_hat, id2char, eos_id)

        dist, length = char_distance(script, pred)

        total_dist += dist
        total_length += length

    return total_dist, total_length


def save_epoch_result(train_result, valid_result):
    """ save result of training (unit : epoch) """
    train_dict, train_loss, train_cer = train_result
    valid_dict, valid_loss, valid_cer = valid_result

    train_dict["loss"].append(train_loss)
    train_dict["cer"].append(train_cer)
    valid_dict["loss"].append(valid_loss)
    valid_dict["cer"].append(valid_cer)

    train_df = pd.DataFrame(train_dict)
    valid_df = pd.DataFrame(valid_dict)

    train_df.to_csv('./data/epoch_train_result.csv', encoding="cp949", index=False)
    valid_df.to_csv('./data/epoch_valid_result.csv', encoding="cp949", index=False)


def save_step_result(train_step_result, loss, cer):
    """ save result of training (unit : K time step) """
    train_step_result["loss"].append(loss)
    train_step_result["cer"].append(cer)
    train_step_df = pd.DataFrame(train_step_result)
    train_step_df.to_csv('./data/step_results.csv', encoding="cp949", index=False)
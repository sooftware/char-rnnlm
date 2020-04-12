import time
import torch
from package.definition import logger, id2char, EOS_token
from package.utils import get_distance, save_step_result

train_step_result = {'loss': [], 'cer': []}


def supervised_train(model, queue, criterion, optimizer, device, print_time_step, teacher_forcing_ratio, worker_num,
                     total_time_step, epoch, train_begin):
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    time_step = 0

    model.train()
    begin = epoch_begin = time.time()

    while True:
        inputs, targets, input_lens, target_lens = queue.get()

        if inputs.shape[0] == 0:
            # empty feats means closing one loader
            worker_num -= 1
            logger.debug('left train_loader: %d' % (worker_num))

            if worker_num == 0:
                break
            else:
                continue

        inputs = inputs.to(device)
        targets = targets.to(device)

        model.module.flatten_parameters()
        y_hat, logit = model(inputs, teacher_forcing_ratio=teacher_forcing_ratio)

        loss = criterion(logit.contiguous().view(-1, logit.size(-1)), targets.contiguous().view(-1))
        total_loss += loss.item()

        total_num += sum(input_lens)
        dist, length = get_distance(targets, y_hat, id2char, EOS_token)
        total_dist += dist
        total_length += length

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        time_step += 1
        torch.cuda.empty_cache()

        if time_step % print_time_step == 0:
            current = time.time()
            elapsed = current - begin
            epoch_elapsed = (current - epoch_begin) / 60.0
            train_elapsed = (current - train_begin) / 3600.0

            logger.info('timestep: {:4d}/{:4d}, loss: {:.4f}, cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h'.format(
                time_step,
                total_time_step,
                total_loss / total_num,
                total_dist / total_length,
                elapsed, epoch_elapsed, train_elapsed)
            )
            begin = time.time()

        if time_step % 1000 == 0:
            save_step_result(train_step_result, total_loss / total_num, total_dist / total_length)

    logger.info('train() completed')

    return total_loss / total_num, total_dist / total_length

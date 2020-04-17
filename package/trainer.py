import time
import torch
from package.definition import logger

train_step_result = {'loss': [], 'cer': []}


def supervised_train(model, queue, perplexity, optimizer, device, print_every,
                     teacher_forcing_ratio, worker_num, total_time_step, train_begin):
    print_loss_total = 0  # Reset every print_every
    epoch_loss_total = 0  # Reset every epoch
    total_num = 0
    time_step = 0

    model.train()
    begin = epoch_begin = time.time()

    while True:
        loss = perplexity
        inputs, targets, input_lens, target_lens = queue.get()

        if inputs.shape[0] == 0:
            # empty feats means closing one loader
            worker_num -= 1
            logger.debug('left train_loader: %d' % worker_num)

            if worker_num == 0:
                break
            else:
                continue

        inputs = inputs.to(device)
        targets = targets.to(device)

        model.module.flatten_parameters()
        outputs = model(inputs, teacher_forcing_ratio=teacher_forcing_ratio)

        # Get loss
        loss.reset()
        for step, step_output in enumerate(outputs):
            batch_size = targets.size(0)
            loss.eval_batch(step_output.contiguous().view(batch_size, -1), targets[:, step])
        # Backpropagation
        model.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.get_loss()

        epoch_loss_total += loss
        print_loss_total += loss
        total_num += sum(input_lens)

        time_step += 1
        torch.cuda.empty_cache()

        if time_step % print_every == 0:
            current = time.time()
            elapsed = current - begin
            epoch_elapsed = (current - epoch_begin) / 60.0
            train_elapsed = (current - train_begin) / 3600.0

            logger.info('timestep: {:4d}/{:4d}, perplexity: {:.4f}, elapsed: {:.2f}s {:.2f}m {:.2f}h'.format(
                time_step,
                total_time_step,
                print_loss_total / print_every,
                elapsed, epoch_elapsed, train_elapsed
            ))
            print_loss_total = 0
            begin = time.time()

    logger.info('train() completed')

    return epoch_loss_total / total_num

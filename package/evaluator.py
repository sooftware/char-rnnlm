import torch
from package.definition import logger


def evaluate(model, queue, perplexity, device):
    logger.info('evaluate() start')

    total_loss = 0
    total_num = 0

    model.eval()

    with torch.no_grad():
        while True:
            loss = perplexity

            inputs, targets, input_lengths, target_lengths = queue.get()

            if inputs.shape[0] == 0:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            model.module.flatten_parameters()
            outputs = model(inputs, teacher_forcing_ratio=teacher_forcing_ratio)

            loss.reset()
            for step, step_output in enumerate(outputs):
                batch_size = targets.size(0)
                loss.eval_batch(step_output.contiguous().view(batch_size, -1), targets[:, step])

            loss = loss.get_loss()

            total_loss += loss
            total_num += sum(input_lens)

    logger.info('evaluate() completed')

    return total_loss / total_num

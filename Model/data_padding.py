import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    return batch.permute(0, 2, 1)


def collate_fn(batch):

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, _, _,speaker_id in batch:
        tensors += [waveform]
        targets += [torch.tensor(speaker_id, dtype=torch.int)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets

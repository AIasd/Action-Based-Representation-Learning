import numpy as np
import random
import os

import torch

from torch.utils.data.sampler import Sampler
from torch import optim
from torch.autograd import Variable

from configs import g_conf


# TODO: When putting sequences, the steering continuity and integrity needs to be verified
def get_rank(input_array):

    rank = 0
    while True:
        try:
            length = len(input_array)
            input_array = input_array[0]
            rank += 1
        except:
            return rank


class RandomSampler(Sampler):
    r"""Samples elements randomly from a given list

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, keys, executed_iterations):
        self.iterations_to_execute = ((g_conf.NUMBER_ITERATIONS) * g_conf.BATCH_SIZE) -\
                                     (executed_iterations)


        self.keys = keys

    def __iter__(self):

        return iter([random.choice(self.keys) for _ in range(self.iterations_to_execute)])


    def __len__(self):
        return self.iterations_to_execute


class RandomSequenceSampler(Sampler):
    r"""Samples random sequences. The sequences can have a stride.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, keys, executed_iterations, batch_size,
                 sequence_stride, sequence_size, drop_last=True, weights=None, SamplerClass=RandomSampler):
        """
        Args
            keys: All the keys that exist on the dataset. For a dataset of size 1000 you have
            keys from 0 to 1000
            executed_iterations: The number of executed iterations so far.
            batch_size: The batch size of the training
            sequence_stride: The number of images  jumped when a sequence is sampled
            sequence_size: The number of images per sequence
            drop_last: If you reduce remove the last and incomplete batch
        """
        self.keys = keys
        if batch_size % sequence_size != 0:
            raise ValueError (" For now the batch size must be divisible by the batch size")

        sampler = SamplerClass(keys, executed_iterations/sequence_size, weigths=weights,
                                start_iterations=g_conf.NUMBER_ITERATIONS / sequence_size)

        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(sequence_size, int) or isinstance(sequence_size, bool) or \
                sequence_size <= 0:
            raise ValueError("sequence should be a positive integeral value, "
                             "but got sequence_size={}".format(sequence_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.sequence_size = sequence_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sequence_stride = sequence_stride

    def __iter__(self):
        sampled_vec = []
        batch = []
        count = 0
        # GET THE INITIAL SAMPLES
        for idx in self.sampler:
            count += 1
            sampled_vec.append(int(idx))
            if count == int(self.batch_size/self.sequence_size):
                for seq in range(0, self.sequence_size * self.sequence_stride,
                                 self.sequence_stride):
                    for sample in sampled_vec:
                        batch.append(sample + seq)
                yield batch
                batch = []
                sampled_vec = []
                count = 0


    def __len__(self):
        if self.drop_last:
            return len(self.sampler)
        else:
            return len(self.sampler)


class SubsetSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)



class PreSplittedSampler(Sampler):
    """ Sample on a list of keys that was previously splitted

    """


    def __init__(self, keys, executed_iterations, weights=None):

        self.keys = keys
        if weights is None:
            self.weights = np.asarray([1.0/float(len(self.keys))]*len(self.keys), dtype=np.float)
        else:
            self.weights = np.asarray(weights)


        self.iterations_to_execute = g_conf.NUMBER_ITERATIONS * g_conf.BATCH_SIZE -\
                                     executed_iterations + g_conf.BATCH_SIZE
        self.replacement = True

    def __iter__(self):
        """

            OBS: One possible thing to be done is the possibility to have a matrix of ids
            of rank N
            OBS2: Probably we dont need weights right now


        Returns:
            Iterator to get ids for the dataset

        """


        rank_keys = get_rank(self.keys)


        # First we check how many subdivisions there are
        weights = torch.from_numpy(self.weights)

        if rank_keys == 2:
            idx = torch.multinomial(weights, self.iterations_to_execute, True)
            idx = idx.tolist()
            return iter([random.choice(self.keys[i]) for i in idx])

        elif rank_keys == 3:
            weights = torch.tensor([1.0 / float(len(self.keys))] * len(self.keys),
                                   dtype=torch.double)
            idx = torch.multinomial(weights, self.iterations_to_execute, True)
            idx = idx.tolist()
            weights = torch.tensor([1.0 / float(len(self.keys[0]))] * len(self.keys[0]),
                                   dtype=torch.double)
            idy = torch.multinomial(weights, self.iterations_to_execute, True)
            idy = idy.tolist()


            return iter([random.choice(self.keys[i][j]) for i, j in zip(idx,idy)])

        else:
            raise ValueError("Keys have invalid rank")


    def __len__(self):
        return self.iterations_to_execute


class LogitSplittedSampler(Sampler):
    """ Sample on a list of keys that was previously splitted
        weights for sampling are logits and have to be softmaxed

    """


    def __init__(self, keys, executed_iterations, weights=None):


        self.keys = keys
        if weights is None:
            self.weights = torch.tensor([1.0/float(len(self.keys))]*len(self.keys), dtype=torch.double)
        else:
            self.weights = torch.from_numpy(weights)
        self.weights = Variable(self.weights, requires_grad=True)

        assert len(self.weights) == len(self.keys), "Number of weights and keys should be the same"
        self.iterations_to_execute = g_conf.NUMBER_ITERATIONS * g_conf.BATCH_SIZE -\
                                     executed_iterations + g_conf.BATCH_SIZE
        self.replacement = True
        self.optim = optim.Adam([self.weights,], lr=0.01)

    def __iter__(self):
        """

            OBS: One possible thing to be done is the possibility to have a matrix of ids
            of rank N
            OBS2: Probably we dont need weights right now


        Returns:
            Iterator to get ids for the dataset

        """
        weights = F.softmax(self.weights)
        idx = torch.multinomial(weights, self.iterations_to_execute, True)
        idx = idx.tolist()
        return iter([random.choice(self.keys[i]) for i in idx])

    def update_weights(self, advantage, perturb=False):
        self.optim.zero_grad()
        obj = torch.sum(-self.weights * advantage)
        obj.backward()
        self.optim.step()
        if perturb:
            N = len(self.weights)
            self.weights = self.weights + torch.normal(torch.zeros(N), perturb * torch.ones(N)).double()

    def __len__(self):
        return self.iterations_to_execute


class BatchSequenceSampler(object):
    r"""Wraps another sampler to yield a mini-batch of indices taking a certain sequence size

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    """

    def __init__(self, keys, executed_iterations,
                 batch_size, sequence_size, sequence_stride, drop_last=True):
        sampler = PreSplittedSampler(keys, executed_iterations)

        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(sequence_size, int) or isinstance(sequence_size, bool) or \
                sequence_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(sequence_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.sequence_size = sequence_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sequence_stride = sequence_stride

    def __iter__(self):

        batch = []
        for idx in self.sampler:
            for seq in range(0, self.sequence_size * self.sequence_stride, self.sequence_stride):
                batch.append(int(idx)+seq)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch


        #if len(batch) > 0 and not self.drop_last:
        #    yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.sequence_size
        else:
            return (len(self.sampler) + self.sequence_size - 1) // self.sequence_size

""" Word-based, for now. Switch to sub-word eventually, o a combination of character- and word-based input."""

import random
import numpy as np

import torch
from torch.autograd import Variable


class DataServer(object):
    """ Iterates through a data source, i.e. a corpus of sentences represented as a list."""
    def __init__(self, data, vocab, max_sent_len, batch_size, shuffle=True, freq_bound=0, pad=True, volatile=False):
        self.data = data
        self.vocab = vocab
        self.max_sent_len = max_sent_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.freq_bound = freq_bound
        self.pad = pad
        self.volatile = volatile

        self.pointer = 0

        if self.shuffle:
            zipped = list(zip(*self.data))
            random.shuffle(zipped)
            self.data = list(zip(*zipped))

    def sent_to_idx(self, sent):
        """ Transforms a sequence of strings to the corresponding sequence of indices. """
        idx_list = [self.vocab.word_to_index[word] if self.vocab.word_to_count[word] >= self.freq_bound else 1 for word in
                    sent.split()]
        # Pad to the desired sentence length
        if self.pad:
            # In case padding is wished for, but no max_sent_len has been specified
            if self.max_sent_len is None:
                self.max_sent_len = self.vocab.observed_msl
            # Pad items to maximum length
            diff = self.max_sent_len - len(idx_list)
            if diff >= 1:
                idx_list += [0] * diff
        return idx_list

    def __iter__(self):
        """ Returns an iterator object. """
        return self

    def __next__(self):
        """ Returns the next batch from within the iterator source. """
        if (self.pointer + self.batch_size) >= self.get_length():
            raise StopIteration
        else:
            self.pointer += self.batch_size
            lower_bound = self.pointer - self.batch_size
            upper_bound = min(self.pointer, len(self.data[0]))

            # Assemble batches
            s1_batch = list()
            s2_batch = list()
            label_batch = list()

            # For similarity estimator training using a similarity corpus
            for i in range(lower_bound, upper_bound):
                s1 = self.sent_to_idx(self.data[0][i][0])
                s2 = self.sent_to_idx(self.data[0][i][1])
                label = [float(self.data[1][i])]
                s1_batch.append(np.array(s1))
                s2_batch.append(np.array(s2))
                label_batch.append(label)

            # Convert to variables
            s1_var = Variable(torch.LongTensor(np.stack(s1_batch, 0)), volatile=self.volatile)
            s2_var = Variable(torch.LongTensor(np.stack(s2_batch, 0)), volatile=self.volatile)
            label_var = Variable(torch.FloatTensor(np.stack(label_batch, 0)), volatile=self.volatile)

            return s1_var, s2_var, label_var

    def get_length(self):
        return len(self.data[0])

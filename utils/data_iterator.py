""" Word-based, for now. Switch to sub-word eventually, o a combination of character- and word-based input."""

import random
import numpy as np


def sent_to_idx(sent, vocab, max_sent_len, freq_bound, pad):
    """ Transforms a sequence of strings to the corresponding sequence of indices. """
    idx_list = [vocab.word_to_index[word] if vocab.word_to_count[word] >= freq_bound else 1 for word in sent.split()]
    # Pad to the desired sentence length
    if pad:
        # In case padding is wished for, but no max_sent_len has been specified
        if max_sent_len is None:
            max_sent_len = vocab.observed_msl

        diff = max_sent_len - len(idx_list)
        if diff >= 1:
            idx_list += [0] * diff
    return idx_list


class DataIterator(object):
    """ Iterates through a data source, i.e. a corpus of sentences represented as a list."""
    def __init__(self, data, data_vocab, max_sent_len, batch_size, shuffle=True, freq_bound=0, pad=True,
                 similarity_corpus=False):
        self.data = data
        self.data_vocab = data_vocab
        self.max_sent_len = max_sent_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.freq_bound = freq_bound
        self.pad = pad
        self.similarity_corpus = similarity_corpus

        self.pointer = 0

        if self.shuffle:
            if similarity_corpus:
                zipped = list(zip(*self.data))
                random.shuffle(zipped)
                self.data = list(zip(*zipped))
            else:
                random.shuffle(self.data)

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
                s1 = sent_to_idx(self.data[0][i][0], self.data_vocab, self.max_sent_len,
                                 self.freq_bound, self.pad)
                s2 = sent_to_idx(self.data[0][i][1], self.data_vocab, self.max_sent_len,
                                 self.freq_bound, self.pad)
                label = [float(self.data[1][i])]
                s1_batch.append(np.array(s1))
                s2_batch.append(np.array(s2))
                label_batch.append(label)
            return np.stack(s1_batch, 0), np.stack(s2_batch, 0), np.stack(label_batch, 0)

    def get_length(self):
        if self.similarity_corpus:
            return len(self.data[0])
        else:
            return len(self.data)


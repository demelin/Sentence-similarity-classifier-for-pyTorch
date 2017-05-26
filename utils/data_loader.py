""" Word-based, for now. Switch to sub-word eventually, o a combination of character- and word-based input."""

import random
import numpy as np


def sent_to_idx(sent, vocab, max_sent_len, freq_bound, pad, mark_borders):
    """ Transforms a sequence of strings to the corresponding sequence of indices. """
    idx_list = [vocab.word_to_index[word] if vocab.word_to_count[word] >= freq_bound else 3 for word in sent.split()]
    # Pad to the desired sentence length
    if pad:
        # In case padding is wished for, but no max_sent_len has been specified
        if max_sent_len is None:
            max_sent_len = vocab.observed_msl

        diff = max_sent_len - len(idx_list)
        if diff >= 1:
            idx_list += [2] * diff
    # Append indices corresponding to <SOS> and <EOS> tokens
    if mark_borders:
        idx_list = [0] + idx_list + [1]
    return idx_list


class DataIterator(object):
    """ Iterates through a data source, i.e. a corpus of sentences represented as a list."""
    def __init__(self, data, data_vocab, max_sent_len, batch_size, shuffle=True, freq_bound=0, pad=True,
                 mark_borders=True, similarity_corpus=False):
        self.data = data
        self.data_vocab = data_vocab
        self.max_sent_len = max_sent_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.freq_bound = freq_bound
        self.pad = pad
        self.mark_borders = mark_borders
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

            if self.similarity_corpus:
                upper_bound = min(self.pointer, len(self.data[0]))
                # Assemble batches
                s1_seq = list()
                s2_seq = list()
                label_seq = list()
                # For similarity estimator training using a similarity corpus
                for i in range(lower_bound, upper_bound):
                    s1 = sent_to_idx(self.data[0][i][0], self.data_vocab, self.max_sent_len,
                                     self.freq_bound, self.pad, self.mark_borders)
                    s2 = sent_to_idx(self.data[0][i][1], self.data_vocab, self.max_sent_len,
                                     self.freq_bound, self.pad, self.mark_borders)
                    label = [float(self.data[1][i])]
                    s1_seq.append(np.array(s1))
                    s2_seq.append(np.array(s2))
                    label_seq.append(label)
                return np.stack(s1_seq, 0), np.stack(s2_seq, 0), np.stack(label_seq, 0)

            else:
                upper_bound = min(self.pointer, len(self.data))
                item_seq = list()
                for j in range(lower_bound, upper_bound):
                    idx_sequence = sent_to_idx(self.data[j], self.data_vocab, self.max_sent_len,
                                               self.freq_bound, self.pad, self.mark_borders)
                    item_seq.append(idx_sequence)
                return np.stack(item_seq, 0)

    def get_length(self):
        if self.similarity_corpus:
            return len(self.data[0])
        else:
            return len(self.data)


class PairedData(object):
    """ Pairs together sentences from separate sources. Accessed by the DataLoader class. """
    def __init__(self, source_iter_object, target_iter_object, batch_size, max_dataset_size):
        self.source_iter_object = source_iter_object
        self.target_iter_object = target_iter_object
        self.max_dataset_size = max_dataset_size
        self.batch_size = batch_size
        # Corpus looping criteria
        self.stop_source = False
        self.stop_target = False

    def __iter__(self):
        """ Returns an iterator object. """
        self.stop_source = False
        self.stop_target = False
        # Initialize iterators over the two datasets
        self.source_iter = iter(self.source_iter_object)
        self.target_iter = iter(self.target_iter_object)
        self.iter = 0
        return self

    def __next__(self):
        """ Returns the next item from within the iterator source. """
        source_batch = None
        target_batch = None
        try:
            source_batch = next(self.source_iter)
        except StopIteration:
            if source_batch is None:
                # Mark loop as completed, reinitialize the iterator object
                self.stop_source = True
                self.source_iter = iter(self.source_iter_object)
                source_batch = next(self.source_iter)
        # Same for target sentence
        try:
            target_batch = next(self.target_iter)
        except StopIteration:
            if target_batch is None:
                self.stop_target = True
                self.target_iter = iter(self.target_iter_object)
                target_batch = next(self.target_iter)
        # Stop iteration once both datasets have been exhausted or the maximum dataset size has been reached
        if (self.stop_source and self.stop_target) is True or self.iter * self.batch_size > self.max_dataset_size:
            self.stop_source = False
            self.stop_target = False
            raise StopIteration
        else:
            self.iter += 1
            return {'source_batch': source_batch, 'target_batch': target_batch}


class DataLoader(object):
    """ Generates data batches consisting of index vectors optionally padded and controlled for word frequency """

    def __init__(self, source_vocab, target_vocab, source_sents, target_sents, opt):
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.source_sents = source_sents
        self.target_sents = target_sents
        self.opt = opt
        # Initialize the iterators operating on the unaligned datasets
        self.source_iter_object = DataIterator(self.source_sents, self.source_vocab, self.opt.max_sent_len,
                                               self.opt.batch_size, shuffle=not self.opt.serial_batches,
                                               freq_bound=self.opt.freq_bound, pad=self.opt.pad,
                                               mark_borders=self.opt.mark_borders, similarity_corpus=False)
        self.target_iter_object = DataIterator(self.target_sents, self.target_vocab, self.opt.max_sent_len,
                                               self.opt.batch_size, shuffle=not self.opt.serial_batches,
                                               freq_bound=self.opt.freq_bound, pad=self.opt.pad,
                                               mark_borders=self.opt.mark_borders, similarity_corpus=False)
        # Pair the data, if so specified
        if self.opt.pair_data:
            self.paired_iterator_object = PairedData(self.source_iter_object, self.target_iter_object,
                                                     self.opt.batch_size, self.opt.max_dataset_size)

    def load_data(self):
        """ Returns an iterator object which is called during training or testing to generate batches. """
        print('Compiling the data loader object(s) ...\n')
        if self.opt.pair_data:
            return self.paired_iterator_object
        else:
            return self.source_iter_object, self.target_iter_object

    def __len__(self):
        """ Returns the length of the corpus."""
        return min(max(len(self.source_sents), len(self.target_sents)), self.opt.max_dataset_size)

"""
# Test area

from text_to_dict import prepare_data
from options import DataLoaderOptions

opt = DataLoaderOptions()

sv, ss = prepare_data(opt, opt.source_a, opt.name_a)
tv, ts = prepare_data(opt, opt.source_b, opt.name_b)

loader = DataLoader(sv, tv, ss, ts, opt)
ds_s, ds_t = loader.load_data()
for i, data in enumerate(ds_s):
    print(i, data)
    #print(type(data))
    #print(type(data[0]))
"""
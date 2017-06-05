""" Word-based, for now. Switch to sub-word eventually, o a combination of character- and word-based input."""

import random
import numpy as np

import torch
from torch.autograd import Variable

from similarity_estimator.sim_util import perform_bucketing


class DataServer(object):
    """ Iterates through a data source, i.e. a list of sentences or list of buckets containing sentences of similar length.
    Produces batch-major batches, i.e. of shape=[seq_len, batch_size]. """
    def __init__(self, data, vocab, opt, is_train=False, shuffle=True, use_buckets=True, volatile=False):
        self.data = data
        self.vocab = vocab
        self.opt = opt
        self.volatile = volatile
        self.use_buckets = use_buckets
        self.pair_id = 0
        self.buckets = None
        # Obtain bucket data
        if self.use_buckets:
            self.buckets, self.data = perform_bucketing(self.opt, self.data)
            self.bucket_id = 0
        # Select appropriate batch size
        if is_train:
            self.batch_size = self.opt.train_batch_size
        else:
            self.batch_size = self.opt.test_batch_size
        # Shuffle data (either batch-wise or as a whole)
        if shuffle:
            if self.use_buckets:
                # Shuffle within buckets
                for i in range(len(self.data)):
                    zipped = list(zip(*self.data[i]))
                    random.shuffle(zipped)
                    self.data[i] = list(zip(*zipped))
                # Shuffle buckets, also
                bucket_all = list(zip(self.buckets, self.data))
                random.shuffle(bucket_all)
                self.buckets, self.data = zip(*bucket_all)
            else:
                zipped = list(zip(*self.data))
                random.shuffle(zipped)
                self.data = list(zip(*zipped))

    def sent_to_idx(self, sent):
        """ Transforms a sequence of strings to the corresponding sequence of indices. """
        idx_list = [self.vocab.word_to_index[word] if self.vocab.word_to_count[word] >= self.opt.freq_bound else 1 for
                    word in sent.split()]
        # Pad to the desired sentence length
        if self.opt.pad:
            if self.use_buckets:
                # Pad to bucket upper length bound
                max_len = self.buckets[self.bucket_id][1]
            else:
                # In case of no bucketing, pad all corpus sentence to a uniform, specified length
                max_len = self.vocab.target_len
            # Adjust padding for single sentence-pair evalualtion (i.e. no buckets, singleton batches)
            if self.batch_size == 1:
                max_len = max(len(self.data[0][self.pair_id][0].split()), len(self.data[0][self.pair_id][1].split()))
            # Pad items to maximum length
            diff = max_len - len(idx_list)
            if diff >= 1:
                idx_list += [0] * diff
        return idx_list

    def __iter__(self):
        """ Returns an iterator object. """
        return self

    def __next__(self):
        """ Returns the next batch from within the iterator source. """
        try:
            if self.use_buckets:
                s1_batch, s2_batch, label_batch = self.bucketed_next()
            else:
                s1_batch, s2_batch, label_batch = self.corpus_next()
        except IndexError:
            raise StopIteration

        # Covert batches into batch major form
        s1_batch = torch.LongTensor(s1_batch).t().contiguous()
        s2_batch = torch.LongTensor(s2_batch).t().contiguous()
        label_batch = torch.FloatTensor(label_batch).contiguous()
        # Convert to variables
        s1_var = Variable(s1_batch, volatile=self.volatile)
        s2_var = Variable(s2_batch, volatile=self.volatile)
        label_var = Variable(label_batch, volatile=self.volatile)
        return s1_var, s2_var, label_var

    def bucketed_next(self):
        """ Samples the next batch from the current bucket. """
        # Assemble batches
        s1_batch = list()
        s2_batch = list()
        label_batch = list()

        if self.bucket_id < self.opt.num_buckets:
            # Fill batches
            while len(s1_batch) < self.batch_size:
                try:
                    s1 = self.sent_to_idx(self.data[self.bucket_id][0][self.pair_id][0])
                    s2 = self.sent_to_idx(self.data[self.bucket_id][0][self.pair_id][1])
                    label = [float(self.data[self.bucket_id][1][self.pair_id])]
                    s1_batch.append(s1)
                    s2_batch.append(s2)
                    label_batch.append(label)
                    self.pair_id += 1
                except IndexError:
                    # Finish batch prematurely if bucket or corpus has been emptied
                    self.pair_id = 0
                    self.bucket_id += 1
                    break
            # Check if bucket is empty, to avoid generation of empty batches
            try:
                if self.pair_id == len(self.data[self.bucket_id][0]):
                    self.bucket_id += 1
            except IndexError:
                pass
        else:
            raise IndexError

        return s1_batch, s2_batch, label_batch

    def corpus_next(self):
        """ Samples the next batch from the un-bucketed corpus. """
        # Assemble batches
        s1_batch = list()
        s2_batch = list()
        label_batch = list()

        # Without bucketing
        if self.pair_id < self.get_length():
            while len(s1_batch) < self.batch_size:
                try:
                    s1 = self.sent_to_idx(self.data[0][self.pair_id][0])
                    s2 = self.sent_to_idx(self.data[0][self.pair_id][1])
                    label = [float(self.data[1][self.pair_id])]
                    s1_batch.append(s1)
                    s2_batch.append(s2)
                    label_batch.append(label)
                    self.pair_id += 1
                except IndexError:
                    break
        else:
            raise IndexError

        return s1_batch, s2_batch, label_batch

    def get_length(self):
        # Return corpus length in sentence pairs
        if self.use_buckets:
            return sum([len(bucket[0]) for bucket in self.data])
        else:
            return len(self.data[0])

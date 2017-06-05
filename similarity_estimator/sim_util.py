import numpy as np
import pandas as pd


class Indexer(object):
    """ Translates words to their respective indices and vice versa. """

    def __init__(self, name):
        self.name = name
        self.word_to_index = dict()
        self.word_to_count = dict()
        # Specify start-and-end-of-sentence tokens
        self.index_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.n_words = 2

        self.target_len = None

    def add_sentence(self, sentence):
        """ Adds sentence contents to index dict. """
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        """ Adds words to index dict. """
        if word not in self.word_to_index:
            self.word_to_index[word] = self.n_words
            self.index_to_word[self.n_words] = word
            self.word_to_count[word] = 1
            self.n_words += 1
        else:
            self.word_to_count[word] += 1

    def set_target_len(self, value):
        self.target_len = value


def perform_bucketing(opt, labeled_pair_list):
    """ Groups the provided sentence pairs into the specified number of buckets of similar size based on the length of
    their longest member. """
    # Obtain sentence lengths
    sentence_pair_lens = [(len(pair[0].split()), len(pair[1].split())) for pair in labeled_pair_list[0]]

    # Calculate bucket size
    buckets = [[0, 0] for _ in range(opt.num_buckets)]
    avg_bucket = len(labeled_pair_list[0]) // opt.num_buckets
    max_lens = [max(pair[0], pair[1]) for pair in sentence_pair_lens]
    len_counts = [(sent_len, max_lens.count(sent_len)) for sent_len in set(max_lens)]
    len_counts.sort(key=lambda x: x[0])

    bucket_pointer = 0
    len_pointer = 0

    while bucket_pointer < opt.num_buckets and len_pointer < len(len_counts):
        target_bucket = buckets[bucket_pointer]
        # Set lower limit on the bucket's lengths
        target_bucket[0] = len_counts[len_pointer][0]
        bucket_load = 0
        while True:
            try:
                len_count_pair = len_counts[len_pointer]
                deficit = avg_bucket - bucket_load
                surplus = (bucket_load + len_count_pair[1]) - avg_bucket
                if deficit >= surplus or bucket_pointer == opt.num_buckets - 1:
                    bucket_load += len_count_pair[1]
                    # Update upper limit on the bucket's lengths
                    target_bucket[1] = len_count_pair[0]
                    len_pointer += 1
                else:
                    bucket_pointer += 1
                    break
            except IndexError:
                break

    # Populate buckets
    bucketed = [([], []) for _ in range(opt.num_buckets)]
    for k in range(len(labeled_pair_list[0])):
        pair_len = max(sentence_pair_lens[k][0], sentence_pair_lens[k][1])
        for l in range(len(buckets)):
            if buckets[l][0] <= pair_len <= buckets[l][1]:
                bucketed[l][0].append(labeled_pair_list[0][k])
                bucketed[l][1].append(labeled_pair_list[1][k])
    return buckets, bucketed


def load_similarity_data(opt, corpus_location, corpus_name):
    """ Converts the extended SICK/ combined STM corpus into a list of tuples of the form (sent_a, sent_b, sim_class),
    used to train the content similarity estimator used within the tGAN model. """
    # Read in the corpus
    df_sim = pd.read_table(corpus_location, header=None, names=['sentence_A', 'sentence_B', 'relatedness_score'],
                           skip_blank_lines=True)

    # Generate corpus list of sentences and labels, and the collections of sentences used for the word to index mapping
    sim_data = [[], []]
    sim_sents = list()
    # Track sentence lengths for max and mean length calculations
    sent_lens = list()
    for i in range(len(df_sim['relatedness_score'])):
        sent_a = df_sim.iloc[i, 0].strip()
        sent_b = df_sim.iloc[i, 1].strip()
        label = "{:.4f}".format(float(df_sim.iloc[i, 2]))

        # Assemble a list of tuples containing the compared sentences, while tracking the maximum observed length
        sim_data[0].append((sent_a, sent_b))
        sim_data[1].append(label)
        sim_sents += [sent_a, sent_b]
        sent_lens += [len(sent_a.split()), len(sent_b.split())]

    # Filter corpus according to specified sentence length parameters
    filtered = [[], []]
    filtered_sents =  list()
    filtered_lens = list()

    # Sent filtering method to truncation by default (in case of anomalous input)
    if opt.sent_select == 'drop' or opt.sent_select == 'truncate' or opt.sent_select is None:
        sent_select = opt.sent_select
    else:
        sent_select = 'truncate'

    # Set filtering size to mean_len + (max_len - mean_len) // 2 by default
    observed_max_len = max(sent_lens)
    if opt.max_sent_len:
        target_len = opt.max_sent_len
    elif opt.sent_select is None:
        target_len = observed_max_len
    else:
        observed_mean_len = int(np.round(np.mean(sent_lens)))
        target_len = observed_mean_len + (observed_max_len - observed_mean_len) // 2

    for i in range(len(sim_data[0])):
        pair = sim_data[0][i]
        if len(pair[0].split()) > target_len or len(pair[1].split()) > target_len:
            if sent_select == 'drop':
                continue
            elif sent_select is None:
                pass
            else:
                pair_0 = ' '.join(pair[0].split()[:target_len])
                pair_1 = ' '.join(pair[1].split()[:target_len])
                pair = (pair_0, pair_1)

        filtered[0].append(pair)
        filtered[1].append(sim_data[1][i])
        filtered_sents += [pair[0], pair[1]]
        filtered_lens.append((len(pair[0]), len(pair[1])))

    # Generate SICK index dictionary and a list of pre-processed
    sim_vocab = Indexer(corpus_name)
    sim_vocab.set_target_len(target_len)

    print('Assembling index dictionary ...')
    for i in range(len(filtered_sents)):
        sim_vocab.add_sentence(filtered_sents[i])
    # Summarize the final data
    print('Registered %d unique words for the %s corpus.\n' % (sim_vocab.n_words, sim_vocab.name))
    return sim_vocab, filtered

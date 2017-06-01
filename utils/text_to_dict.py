""" Majority of the used classes and functions are adopted from the pyTorch NLP tutorial and 
adjusted as needed. """


class Indexer(object):
    """ Translates words to their respective indices and vice versa. """

    def __init__(self, name):
        self.name = name
        self.word_to_index = dict()
        self.word_to_count = dict()
        # Specify start-and-end-of-sentence tokens
        self.index_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.n_words = 2

        self.observed_msl = None

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

    def set_observed_msl(self, value):
        self.observed_msl = value


def read_text_file(text_file, sent_select='truncate', max_sent_len=None, lower=False):
    """ Processes a raw text file into a list of lines. """
    print('Reading in %s ...' % text_file)
    # Generates a list of lines
    lines = open(text_file).read().strip().split('\n')
    filtered = list()
    observed_msl = 0
    for line in lines:
        if max_sent_len:
            if sent_select == 'drop':
                if len(line.split()) > max_sent_len:
                    continue
            elif sent_select == 'truncate':
                line = ' '.join(line.split()[:max_sent_len])
            else:
                raise ValueError('sent_select may equal either \'truncate\' or \'drop\'.')
        else:
            # If no max_sent_len is specified, get the maximum observed sentence length for subsequent padding
            if len(line.split()) > observed_msl:
                observed_msl = len(line.split())
        if lower:
            line = line.lower()
        filtered.append(line)
    print('Read in %d sentences.' % len(filtered))
    return filtered, observed_msl


def prepare_data(opt, corpus_source, corpus_name):
    """ Compiles index dictionaries for some source text file. """
    # Set translation directionality
    corpus_sents, observed_msl = read_text_file(corpus_source, opt.sent_select, opt.max_sent_len, opt.lower)
    corpus_vocab = Indexer(corpus_name)
    corpus_vocab.set_observed_msl(observed_msl)

    print('Assembling index dictionary ...')
    for i in range(len(corpus_sents)):
        corpus_vocab.add_sentence(corpus_sents[i])

    # Summarize the final data
    print('Registered %d unique words for the %s corpus.\n' % (corpus_vocab.n_words, corpus_vocab.name))
    # Return pre-processed data
    return corpus_vocab, corpus_sents

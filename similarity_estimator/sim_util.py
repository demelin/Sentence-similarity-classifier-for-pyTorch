import pandas as pd

from utils.text_to_dict import Indexer


def load_similarity_data(opt, corpus_location, corpus_name):
    """ Converts the extended SICK/ combined STM corpus into a list of tuples of the form (sent_a, sent_b, sim_class),
    used to train the content similarity estimator used within the tGAN model. """
    # Read in the corpus
    df_sim = pd.read_table(corpus_location, header=None, names=['sentence_A', 'sentence_B', 'relatedness_score'],
                           skip_blank_lines=True)

    # Generate corpus list of sentences and labels, and the collections of sentences used for the word to index mapping
    sim_data = [[], []]
    sim_sents = list()
    # Track maximum sentence length for latter use in padding of index vectors
    observed_msl = 0
    for i in range(len(df_sim['relatedness_score'])):
        sent_a = df_sim.iloc[i, 0].strip()
        sent_b = df_sim.iloc[i, 1].strip()
        label = "{:.4f}".format(float(df_sim.iloc[i, 2]))

        # Control for the desired maximum sentence length
        if opt.max_sent_len:
            if opt.sent_select == 'drop':
                if len(sent_a.split()) > opt.max_sent_len or len(sent_b.split()) > opt.max_sent_len:
                    continue
            elif opt.sent_select == 'truncate':
                sent_a = ' '.join(sent_a.split()[:opt.max_sent_len])
                sent_b = ' '.join(sent_b.split()[:opt.max_sent_len])
            else:
                raise ValueError('sent_select may equal either \'truncate\' or \'drop\'.')

        sim_data[0].append((sent_a, sent_b))
        sim_data[1].append(label)
        sim_sents += [sent_a, sent_b]
        if max(len(sent_a.split()), len(sent_b.split())) > observed_msl:
            observed_msl = max(len(sent_a.split()), len(sent_b.split()))

    # Generate SICK index dictionary and a list of pre-processed
    sim_vocab = Indexer(corpus_name)
    sim_vocab.observed_msl = observed_msl

    print('Assembling index dictionary ...')
    for i in range(len(sim_sents)):
        sim_vocab.add_sentence(sim_sents[i])
    # Summarize the final data
    print('Registered %d unique words for the %s corpus.\n' % (sim_vocab.n_words, sim_vocab.name))
    return sim_vocab, sim_data

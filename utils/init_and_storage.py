""" Various utility and helper functions used throughout the model. """
import os
import torch

from utils.parameter_initialization import xavier_normal


def add_pretrained_embeddings(embedding_table, target_vocab, pretrained_vec_file):
    """ Fills the existing embedding table with pre-trained embeddings. Run after the initialization of the 
    embedding table. """
    print('Adding pre-trained embeddings ... ')

    # Read in the pretrained vector file
    with open(pretrained_vec_file, 'r') as in_file:
        for line in in_file:
            entries = line.split()
            # Check for blank/ incomplete lines
            if len(entries) != 301:
                continue
            word = entries[0]
            vec = [float(n) for n in entries[1:]]
            # Inject pretrained vectors
            try:
                word_row = target_vocab.word_to_index[word]
                embedding_table[word_row][:] = vec
            # Extend the vocabulary
            except KeyError:
                continue

    return torch.FloatTensor(embedding_table)


def extend_embeddings(source_table, source_vocab, target_vocab, pretrained_vec_file):
    """ Extends an existing, trained embedding table with new entries corresponding to new words from some target 
    corpus. """
    print('Extending embedding table with pre-trained embeddings ... ')
    # Consolidate new and old vocabs
    added_words = list()
    source_vocab_start_words = source_vocab.n_words

    for idx_i in range(4, target_vocab.n_words):
        word = target_vocab.index_to_word[idx_i]
        try:
            source_vocab.word_to_index[word]
        except KeyError:
            source_vocab.add_word(word)
            source_vocab.word_to_count[word] = target_vocab.word_to_count[word]
            added_words.append(word)

    # Initialize embedding table extension
    added_embeddings = xavier_normal(torch.FloatTensor(len(added_words), 300)).numpy()

    # Update embedding table
    with open(pretrained_vec_file, 'r') as in_file:
        for line in in_file:
            entries = line.split()
            # Check for blank/ incomplete lines
            if len(entries) != 301:
                continue
            word = entries[0]
            vec = [float(n) for n in entries[1:]]
            # Collect new vectors
            if word in added_words:
                word_row = source_vocab.word_to_index[word] - source_vocab_start_words
                added_embeddings[word_row][:] = vec

    # Concatenate source embedding table with the new additions
    extended_table = torch.cat([source_table, torch.FloatTensor(added_embeddings)], 0)
    return source_vocab, extended_table


def add_all_embeddings(embedding_table, vocab_object, pretrained_vec_file):
    """ Concatenates all missing pre-trained vectors to an existing embedding table and modifies the lookup object
    accordingly. """
    print('Adding pre-trained embeddings ... ')
    # Initialize pretrained embedding table
    pretrained_table = list()
    # Read in the pretrained vector file
    with open(pretrained_vec_file, 'r') as in_file:
        for line in in_file:
            entries = line.split()
            # Check for blank/ incomplete lines
            if len(entries) != 301:
                continue
            word = entries[0]
            vec = [float(n) for n in entries[1:]]
            # Inject pretrained vectors
            try:
                value = vocab_object.word_to_index[word]
                embedding_table[value][:] = vec
            # Extend the vocabulary
            except KeyError:
                vocab_object.add_word(word)
                pretrained_table.append(vec)

    # Join embedding tables
    pretrained_table = torch.FloatTensor(pretrained_table)
    joint_table = torch.cat([embedding_table, pretrained_table], 0)

    return vocab_object, joint_table


def initialize_parameters(network):
    """ Initializes the parameters of the network's weights following the Xavier initialization scheme. """
    params = network.parameters()
    for tensor in params:
        if len(tensor.size()) > 1:
            tensor = xavier_normal(tensor)
        else:
            tensor.data.fill_(0.1)
    print("Initialized weight parameters of %s with Xavier initialization using the normal distribution." %
          network.name)


def update_learning_rate(optimizer, new_lr):
    """ Decreases the learning rate to promote training gains. """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def save_network(network, network_label, active_epoch, save_directory):
    """ Saves the parameters of the specified network under the specified path. """
    file_name = '%s_%s' % (str(active_epoch), network_label)
    save_path = os.path.join(save_directory, file_name)
    torch.save(network.cpu().state_dict(), save_path)
    print('Network %s saved following the completion of epoch %s | Location: %s' %
          (network_label, str(active_epoch), save_path))


def load_network(network, network_label, target_epoch, load_directory):
    """ Helper function for loading network work. """
    load_filename = '%s_%s' % (str(target_epoch), network_label)
    load_path = os.path.join(load_directory, load_filename)
    network.load_state_dict(torch.load(load_path))
    print('Network %s, version %s loaded from location %s' % (network_label, target_epoch, load_path))

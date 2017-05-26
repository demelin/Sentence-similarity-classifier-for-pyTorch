""" Various utility and helper functions used throughout the model. """
import os
import torch
import pickle
import numpy as np

from utils.parameter_initialization import xavier_normal


def add_pretrained_embeddings(embedding_table, vocab_object, pretrained_vec_file):
    """ Concatenates the pre-trained vectors to an existing embedding table and modifies the lookup object
    accordingly. Run after the initialization of the embedding table. """
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
    pretrained_table = np.array(pretrained_table)
    joint_table = torch.FloatTensor(np.vstack((embedding_table, pretrained_table)))

    return vocab_object, joint_table


def adapt_embeddings(opt, classifier, target_vocab):
    """ Extends and filters the embedding table and the lookup object obtained via pre-training, keeping the relevant
     entries for the target corpus. Run after the initialization of the target embedding table. """
    # Load pretrained parameters
    pretrained_path = os.path.join(opt.save_dir, 'pretraining/pretrained.pkl')
    with open(pretrained_path, 'rb') as f:
        pretrained_embeddings, pretrained_vocab = pickle.load(f)

    # Get initial embeddings
    embeddings = classifier.encoder_a.embedding_table.weight.data

    # Add words from the new corpus to the pretrained vocabulary
    for idx_i in range(target_vocab.n_words):
        word = target_vocab.index_to_word[idx_i]
        try:
            idx_p = pretrained_vocab.word_to_index[word]
            embeddings[idx_i] = pretrained_embeddings[idx_p]
        except KeyError:
            continue

    return embeddings


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


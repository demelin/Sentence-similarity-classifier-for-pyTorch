""" Pre-trains the similarity estimator network on the SICK corpus. """

import os
import pickle

import numpy as np
import torch
from utils.data_server import DataServer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from torch.autograd import Variable

from similarity_estimator.networks import SiameseClassifier
from similarity_estimator.options import TestingOptions, ClusterOptions
from similarity_estimator.sick_extender import SickExtender
from similarity_estimator.sim_util import load_similarity_data
from utils.init_and_storage import add_pretrained_embeddings, extend_embeddings, update_learning_rate, save_network
from utils.parameter_initialization import xavier_normal

# Initialize training parameters
opt = TestingOptions()

if opt.pre_training:
    save_dir = opt.pretraining_dir
    sts_corpus_path = os.path.join(opt.data_dir, 'se100.txt')
    vocab, corpus_data = load_similarity_data(opt, sts_corpus_path, 'SemEval13STS_corpus')
    # Initialize an embedding table
    init_embeddings = xavier_normal(torch.randn([vocab.n_words, 300])).numpy()
    # Add FastText embeddings
    fasttext_embeddings = add_pretrained_embeddings(
        init_embeddings, vocab, os.path.join(opt.data_dir, 'fasttext_embeds.txt'))
    # Initialize the similarity estimator network
    classifier = SiameseClassifier(vocab.n_words, opt, is_train=True)
    # Initialize parameters
    classifier.initialize_parameters()
    # Inject the pre-trained embedding table
    classifier.encoder_a.embedding_table.weight.data.copy_(fasttext_embeddings)

else:
    save_dir = opt.save_dir
    # Extend the corpus with synthetic data
    source_corpus_path = os.path.join(opt.data_dir, 'SICK.txt')
    language_model_path = os.path.join(opt.data_dir, 'sick_lm.klm')
    extended_corpus_path = os.path.join(opt.data_dir, 'extended_sick.txt')
    extender = SickExtender(source_corpus_path, opt.data_dir, lm_path=language_model_path)
    if not os.path.exists(extended_corpus_path):
        extender.create_extension()
    # Obtain data
    target_vocab, corpus_data = load_similarity_data(opt, extended_corpus_path, 'sick_corpus')
    # Load pretrained parameters
    pretrained_path = os.path.join(opt.save_dir, 'pretraining/pretrained.pkl')
    with open(pretrained_path, 'rb') as f:
        pretrained_embeddings, pretrained_vocab = pickle.load(f)
    # Extend embeddings
    vocab, extended_embeddings = extend_embeddings(
        pretrained_embeddings, pretrained_vocab, target_vocab, os.path.join(opt.data_dir, 'fasttext_embeds.txt'))
    # Save extended embeddings
    vocab_path = os.path.join(opt.save_dir, 'extended_vocab.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    # Initialize the similarity estimator network
    classifier = SiameseClassifier(vocab.n_words, opt, is_train=True)
    # Initialize parameters
    classifier.initialize_parameters()
    # Inject the pre-trained embedding table
    classifier.encoder_a.embedding_table.weight.data.copy_(extended_embeddings)

# Set up training
learning_rate = opt.learning_rate

# Initialize global tracking variables
best_validation_accuracy = 0
epochs_without_improvement = 0
final_epoch = 0

# Split the data for training and validation (70/30)
train_data, valid_data, train_labels, valid_labels = train_test_split(corpus_data[0], corpus_data[1],
                                                                      test_size=0.3, random_state=0)

# Training loop
for epoch in range(opt.num_epochs):

    # Declare tracking variables
    running_loss = list()
    total_train_loss = list()

    # Initiate the training data loader
    train_loader = DataServer([train_data, train_labels], vocab, opt, is_train=True, volatile=False)

    # Training loop
    for i, data in enumerate(train_loader):
        # Obtain data
        s1_var, s2_var, label_var = data
        classifier.train_step(s1_var, s2_var, label_var)
        train_batch_loss = classifier.loss.data[0]

        running_loss.append(train_batch_loss)
        total_train_loss.append(train_batch_loss)

        if i % opt.report_freq == 0 and i != 0:
            running_avg_loss = sum(running_loss) / len(running_loss)
            print('Epoch: %d | Training Batch: %d | Average loss since batch %d: %.4f' %
                  (epoch, i, i - opt.report_freq, running_avg_loss))
            running_loss = list()

    # Report epoch statistics
    avg_training_accuracy = sum(total_train_loss) / len(total_train_loss)
    print('Average training batch loss at epoch %d: %.4f' % (epoch, avg_training_accuracy))

    # Validate after each epoch; set tracking variables
    if epoch >= opt.start_early_stopping:
        total_valid_loss = list()

        # Initiate the training data loader
        valid_loader = DataServer([valid_data, valid_labels], vocab, opt, is_train=True, volatile=True)

        # Validation loop (i.e. perform inference on the validation set)
        for i, data in enumerate(valid_loader):
            s1_var, s2_var, label_var = data
            # Get predictions and update tracking values
            classifier.test_step(s1_var, s2_var, label_var)
            valid_batch_loss = classifier.loss.data[0]
            total_valid_loss.append(valid_batch_loss)

        # Report fold statistics
        avg_valid_accuracy = sum(total_valid_loss) / len(total_valid_loss)
        print('Average validation fold accuracy at epoch %d: %.4f' % (epoch, avg_valid_accuracy))
        # Save network parameters if performance has improved
        if avg_valid_accuracy <= best_validation_accuracy:
            epochs_without_improvement += 1
        else:
            best_validation_accuracy = avg_valid_accuracy
            epochs_without_improvement = 0
            save_network(classifier.encoder_a, 'sim_classifier', 'latest', save_dir)

    # Save network parameters at the end of each nth epoch
    if epoch % opt.save_freq == 0 and epoch != 0:
        print('Saving model networks after completing epoch %d' % epoch)
        save_network(classifier.encoder_a, 'sim_classifier', epoch, save_dir)

    # Anneal learning rate:
    if epochs_without_improvement == opt.start_annealing:
        old_learning_rate = learning_rate
        learning_rate *= opt.annealing_factor
        update_learning_rate(classifier.optimizer_a, learning_rate)
        print('Learning rate has been updated from %.4f to %.4f' % (old_learning_rate, learning_rate))

    # Terminate training early, if no improvement has been observed for n epochs
    if epochs_without_improvement >= opt.patience:
        print('Stopping training early after %d epochs, following %d epochs without performance improvement.' %
              (epoch, epochs_without_improvement))
        final_epoch = epoch
        break

print('Training procedure concluded after %d epochs total. Best validated epoch: %d.'
      % (final_epoch, final_epoch - opt.patience))

if opt.pre_training:
    # Save pretrained embeddings and the vocab object
    pretrained_path = os.path.join(save_dir, 'pretrained.pkl')
    pretrained_embeddings = classifier.encoder_a.embedding_table.weight.data
    with open(pretrained_path, 'wb') as f:
        pickle.dump((pretrained_embeddings, vocab), f)
    print('Pre-trained parameters saved to %s' % pretrained_path)

if not opt.pre_training:
    """ Regression step over the training set to improve the predictive power of the model """
    # Obtain similarity score predictions for each item within the training corpus
    labels = list()
    predictions = list()

    # Initiate the training data loader
    train_loader = DataServer([train_data, train_labels], vocab, opt, is_train=True, volatile=True)

    # Obtaining predictions
    for i, data in enumerate(train_loader):
        # Obtain data
        s1_var, s2_var, label_var = data
        labels += [l[0] for l in label_var.data.numpy().tolist()]
        classifier.test_step(s1_var, s2_var, label_var)
        batch_predict = classifier.prediction.data.squeeze().numpy().tolist()
        predictions += batch_predict

    labels = np.array(labels)
    predictions = np.array(predictions).reshape(-1, 1)

    # Fit an SVR (following the scikit-learn tutorial)
    sim_svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5, param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                                                           "gamma": np.logspace(-2, 2, 5)})

    sim_svr.fit(predictions, labels)
    print('SVR complexity and bandwidth selected and model fitted successfully.')

    # Save trained SVR model
    svr_path = os.path.join(save_dir, 'sim_svr.pkl')
    with open(svr_path, 'wb') as f:
        pickle.dump(sim_svr, f)
    print('Trained SVR model saved to %s' % svr_path)

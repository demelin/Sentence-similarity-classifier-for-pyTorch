import os


class TestingOptions(object):
    """ Default options for the siamese similarity estimator network. Use for quick evaluation on home machine. """

    def __init__(self):
        # Data
        self.max_sent_len = None
        self.pad = True
        self.freq_bound = 3
        self.shuffle = True
        self.sent_select = 'truncate'
        self.lower = False
        self.num_buckets = 3

        # Network
        self.embedding_dims = 300
        self.hidden_dims = 50
        self.num_layers = 1
        self.train_batch_size = 16
        self.test_batch_size = 1
        self.clip_value = 0.25
        self.learning_rate = 0.0001
        self.beta_1 = 0.5

        self.pre_training = True
        self.num_epochs = 100

        self.start_early_stopping = 2
        self.patience = 10
        self.start_annealing = 4
        self.annealing_factor = 0.75

        # Training
        self.report_freq = 1
        self.save_freq = 4
        self.home_dir = os.path.join(os.path.dirname(__file__), '..')
        self.data_dir = os.path.join(self.home_dir, 'data')
        self.save_dir = os.path.join(self.home_dir, 'similarity_estimator/models')
        self.pretraining_dir = os.path.join(self.save_dir, 'pretraining')

        # Testing
        self.num_test_samples = 10


class ClusterOptions(object):
    """ Default options for the siamese similarity estimator network. Use for deployment on cluster. """

    def __init__(self):
        # Data
        self.max_sent_len = None
        self.pad = True
        self.freq_bound = 3
        self.shuffle = True
        self.sent_select = 'truncate'
        self.lower = False
        self.num_buckets = 8

        # Network
        self.embedding_dims = 300
        self.hidden_dims = 50
        self.num_layers = 1
        self.train_batch_size = 16
        self.test_batch_size = 1
        self.clip_value = 0.25  # Following pyTorch LM example's default value
        self.learning_rate = 0.0001
        self.beta_1 = 0.5

        # Training
        self.pre_training = True
        self.num_epochs = 1000

        # Mostly arbitrary values from here on; for a more informed approach, consult Early Stopping paper
        self.start_early_stopping = self.num_epochs // 20
        self.patience = self.num_epochs // 50
        self.start_annealing = self.num_epochs // 100
        self.annealing_factor = 0.75

        self.report_freq = 100
        self.save_freq = 20
        self.home_dir = os.path.join(os.path.dirname(__file__), '..')
        self.data_dir = os.path.join(self.home_dir, 'data')
        self.save_dir = os.path.join(self.home_dir, 'similarity_estimator/models')
        self.pretraining_dir = os.path.join(self.save_dir, 'pretraining')

        # Testing
        self.num_test_samples = 10

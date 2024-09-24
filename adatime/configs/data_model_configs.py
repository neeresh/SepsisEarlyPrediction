def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class GTN():
    def __init__(self):

        # GTN
        self.d_model = 512
        self.d_hidden = 1024
        self.q = 8
        self.v = 8
        self.h = 8
        self.N = 8
        self.dropout = 0.2
        self.pe = True
        self.mask = True
        self.lr = 1e-4
        self.batch_size = 16
        self.num_epochs = 20

        self.device = 'cuda'

        # Dataset params
        self.d_input = 336
        self.d_channel = 40
        self.d_output = 2

        # pre-train configs
        self.pretrain_epoch = 10
        self.finetune_epoch = 20

        # fine-tune configs
        self.num_classes_target = 2

        # optimizer parameters
        self.optimizer = 'adam'
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-8  # 3e-4
        self.lr_f = self.lr

        # data parameters
        self.drop_last = True
        self.batch_size = 32

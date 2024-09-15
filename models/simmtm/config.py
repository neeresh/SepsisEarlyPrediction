class Config(object):
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

        # self.target_batch_size = 32  # the size of target dataset (the # of samples used to fine-tune).

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.5
        self.jitter_ratio = 2
        self.max_seg = 12


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 64
        self.timesteps = 50

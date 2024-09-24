
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
        self.batch_size = 32
        self.num_epochs = 20

        self.device = 'cuda'

        # Dataset params
        self.d_input = 336
        self.d_channel = 40
        self.d_output = 2

        # fine-tune configs
        self.num_classes_target = 2

        # optimizer parameters
        self.optimizer = 'adam'
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.learning_rate = 0.0005
        self.weight_decay = 0.0001
        self.step_size = 50
        self.lr_decay = 0.5

        # data parameters
        self.drop_last = True

        # Adatime
        self.normalize = False
        self.shuffle = True
        self.drop_last = True
        self.features_len = 336
        self.final_out_channels = 40
        self.num_classes = 2

        # DeepCoral
        self.coral_wt = 4.23
        self.src_cls_loss_wt = 0.10
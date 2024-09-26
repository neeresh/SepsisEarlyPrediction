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
        self.num_epochs = 40

        self.device = 'cuda'

        # Dataset params
        self.d_input = 336
        self.d_channel = 40
        self.d_output = 2

        # fine-tune configs
        self.num_classes_target = 2

        # optimizer parameters
        self.optimizer = 'adam'
        # self.beta1 = 0.9
        # self.beta2 = 0.99
        # self.learning_rate = 0.0005
        # self.weight_decay = 0.0001
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
        self.deepcoral_coral_wt = 4.23035475456397
        self.deepcoral_learning_rate = 0.0005
        self.deepcoral_src_cls_loss_wt = 0.1013209750429822
        self.deepcoral_weight_decay = 0.0001
        self.deepcoral_step_size = 50
        self.deepcoral_lr_decay = 0.5

        # MMDA
        self.mmda_cond_ent_wt = 1.383002023133561
        self.mmda_coral_wt = 8.36810764913737
        self.mmda_learning_rate = 0.001
        self.mmda_mmd_wt = 3.964042918489996
        self.mmda_src_cls_loss_wt = 6.794522068759213
        self.mmda_weight_decay = 0.0001

        # DANN
        self.dann_domain_loss_wt = 2.943729820531079
        self.dann_learning_rate = 0.001
        self.dann_src_cls_loss_wt = 5.1390077646202
        self.dann_weight_decay = 0.0001
        self.dann_disc_hid_dim = 64

        # CDAN
        self.cdan_cond_ent_wt = 1.2920143348777362
        self.cdan_domain_loss_wt = 9.545761950873414
        self.cdan_learning_rate = 0.001
        self.cdan_src_cls_loss_wt = 9.430292987535724
        self.cdan_weight_decay = 0.0001
        self.cdan_disc_hid_dim = 64

        # SASA
        self.sasa_domain_loss_wt = 7.3937939938562
        self.sasa_learning_rate = 0.005
        self.sasa_src_cls_loss_wt = 4.185814373345016
        self.sasa_weight_decay = 0.0001

        # CoDATS
        self.codats_domain_loss_wt = 3.2750474868706925
        self.codats_learning_rate = 0.001
        self.codats_src_cls_loss_wt = 6.335109786953256
        self.codats_weight_decay = 0.0001
        self.codats_hidden_dim = 64

        # CoTMiX
        self.cotmix_learning_rate = 0.001
        self.cotmix_mix_ratio = 0.9
        self.cotmix_temporal_shift = 14
        self.cotmix_src_cls_weight = 0.78
        self.cotmix_src_supCon_weight = 0.1
        self.cotmix_trg_cont_weight = 0.1
        self.cotmix_trg_entropy_weight = 0.05
        self.cotmix_weight_decay = 1e-4

        # DIRT
        self.dirt_cond_ent_wt = 1.20721518968644
        self.dirt_domain_loss_wt = 1.9012145515129044
        self.dirt_learning_rate = 0.005
        self.dirt_src_cls_loss_wt = 9.67861021290254
        self.dirt_vat_loss_wt = 7.7102843136045855
        self.dirt_weight_decay = 0.0001

        # DSAN
        self.dsan_learning_rate = 0.001
        self.dsan_mmd_wt = 2.0872340713147786
        self.dsan_src_cls_loss_wt = 1.8744909939900247
        self.dsan_domain_loss_wt = 1.59
        self.dsan_weight_decay = 0.0001

        # AdvSKM
        self.advskm_domain_loss_wt = 1.338788378230754
        self.advskm_learning_rate = 0.0005
        self.advskm_src_cls_loss_wt = 2.468525942065072
        self.advskm_weight_decay = 0.0001
        self.advskm_DSKN_disc_hid = 64
        self.advskm_disc_hid_dim = 64

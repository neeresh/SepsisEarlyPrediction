
nn_config = {'epochs_num': 20, 'batch_size': 1, 'input_size': 39, 'hidden_size': 39, 'num_of_heads': 3,
             'num_layers': 4, 'dropout': 0.0, 'lr': 0.0001, 'size_average': True, 'clipping': 50, 'to_concat': True}

transformer_rnn_param = {'input_size': 128, 'hidden_size': 39, 'num_of_heads': 3, 'num_layers': 4,
                         'dropout': 0.25, 'batch_size': 128, 'lr': 0.001, 'epochs_num': 5, "size_average": True,
                         'to_concat': False, 'clipping': 1.0}

# gtn_param = {'d_model': 512, 'd_hidden': 1024, 'q': 8, 'v': 8, 'h': 8, 'N': 8, 'dropout': 0.2, 'pe': True, 'mask': True,
#              'lr': 1e-4, 'batch_size': 3, 'num_epochs': 30}
# gtn_param = {'d_model': 768, 'd_hidden': 2048, 'q': 16, 'v': 16, 'h': 16, 'N': 16, 'dropout': 0.2, 'pe': True, 'mask': True,
#              'lr': 1e-4, 'batch_size': 3, 'num_epochs': 50}
gtn_param = {'d_model': 1024, 'd_hidden': 2048, 'q': 14, 'v': 14, 'h': 14, 'N': 14, 'dropout': 0.2, 'pe': True, 'mask': True,
             'lr': 1e-4, 'batch_size': 3, 'num_epochs': 20}
gtn_cv_param = {'d_model': 512, 'd_hidden': 1024, 'q': 8, 'v': 8, 'h': 8, 'N': 8, 'dropout': 0.2, 'pe': True, 'mask': True,
             'lr': 1e-4, 'batch_size': 3, 'num_epochs': 50}
# masked_gtn_param = {'d_model': 512, 'd_hidden': 1024, 'q': 8, 'v': 8, 'h': 8, 'N': 8, 'dropout': 0.2, 'pe': True, 'mask': True,
#              'lr': 1e-4, 'batch_size': 3, 'num_epochs': 30}
# masked_gtn_param = {'d_model': 768, 'd_hidden': 2048, 'q': 16, 'v': 16, 'h': 16, 'N': 16, 'dropout': 0.2, 'pe': True, 'mask': True,
#              'lr': 1e-4, 'batch_size': 3, 'num_epochs': 50}
masked_gtn_param = {'d_model': 1024, 'd_hidden': 2048, 'q': 14, 'v': 14, 'h': 14, 'N': 14, 'dropout': 0.2, 'pe': True, 'mask': True,
             'lr': 3e-4, 'batch_size': 3, 'num_epochs': 20}
modified_gtn_param = {'d_model': 512, 'd_hidden': 1024, 'q': 8, 'v': 8, 'h': 8, 'N': 8, 'dropout': 0.2, 'pe': True, 'mask': True,
             'lr': 1e-4, 'batch_size': 3, 'num_epochs': 10}

vanilla_param = {'num_epochs': 10, 'batch_size': 3}

pretrain_params = {'d_model': 512, 'd_hidden': 1024, 'q': 8, 'v': 8, 'h': 8, 'N': 8, 'dropout': 0.2, 'pe': True,
                   'mask': True, 'lr': 1e-4, 'batch_size': 3, 'num_epochs': 30}

# TARNet
# Batch size should be divisible by len(train) and len(test)
tarnet_param = {'task_type': 'classification', 'device': 'cuda', 'nclasses': 2, 'seq_len': 336, 'batch': 16,
                'input_size': 191, 'emb_size': 128, 'nhead': 8, 'nhid': 256, 'nhid_tar': 512,  'nhid_task':512,
                'nlayers':4, 'dropout':0.01, 'epochs': 200, 'lr': 0.0001, 'masking_ratio': 0.15,
                # "ratio highest attention" is the proportion of the input sequence that receives the highest attention
                # scores in the attention. Used in the function attention_sampled_masking_heuristic to determine
                # which parts of the input sequence should be masked based on their attention scores.
                'ratio_highest_attention': 0.5, 'avg': 'macro', 'dataset': 'Sepsis',
                # This parameter controls the balance between the reconstruction task and the end task
                # (classification or regression) in the multitask training process.
                # A value closer to 1 means giving more weight to the reconstruction task.
                # A value closer to 0 means giving more weight to the end task (classification or regression).
                'task_rate': 0.5,
                # This parameter controls the balance between the masked and unmasked parts of the reconstruction loss.
                # A value closer to 1 means giving more weight to the masked reconstruction loss.
                # A value closer to 0 means giving more weight to the unmasked reconstruction loss.
                'lamb': 0.80}

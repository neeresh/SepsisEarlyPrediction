nn_config = {'epochs_num': 20, 'batch_size': 1, 'input_size': 39, 'hidden_size': 39, 'num_of_heads': 3,
             'num_layers': 4, 'dropout': 0.0, 'lr': 0.0001, 'size_average': True, 'clipping': 50, 'to_concat': True}

lgb_classifier_params = {'num_leaves': 60, 'min_data_in_leaf': 120, 'objective': 'binary', 'max_depth': -1,
                         'learning_rate': 0.01, 'reg_alpha': 0, 'reg_lambda': 0, 'metric': 'auc', 'verbosity': -1,
                         'early_stopping_rounds': 100, 'scale_pos_weight': 20,
                         # 'feature_fraction': 0.9, 'bagging_freq': 3, 'bagging_fraction': 0.9, 'bagging_seed': 0,
                         # 'feature_fraction_seed': 0, 'is_unbalanced': False,
                         }

transformer_rnn_param = {'input_size': 128, 'hidden_size': 39, 'num_of_heads': 3, 'num_layers': 4,
                         'dropout': 0.25, 'batch_size': 128, 'lr': 0.001, 'epochs_num': 5, "size_average": True,
                         'to_concat': False, 'clipping': 1.0}

# gtn_param = {'d_model': 512, 'd_hidden': 1024, 'q': 8, 'v': 8, 'h': 8, 'N': 8, 'dropout': 0.2, 'pe': True, 'mask': True,
#              'lr': 0.001}

gtn_param = {'d_model': 256, 'd_hidden': 512, 'q': 4, 'v': 4, 'h':4, 'N': 4, 'dropout': 0.0, 'pe': True, 'mask': True,
             'lr': 0.001}


import os
import argparse
from utils.functions import Storage
class ConfigRegression():
    def __init__(self, args):
        HYPER_MODEL_MAP = {
            'MIT-FRNet': self.__MIT-FRNet,
        }
        HYPER_DATASET_MAP = self.__datasetCommonParams()
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]
        if commonArgs['data_missing']: dataArgs = dataArgs['aligned_missing'] if (commonArgs['need_data_aligned'] and 'aligned_missing' in dataArgs) else dataArgs['unaligned_missing']
        else: dataArgs = dataArgs['aligned'] if (commonArgs['need_data_aligned'] and 'aligned' in dataArgs) else dataArgs['unaligned']
        self.args = Storage(dict(vars(args), **dataArgs, **commonArgs, **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],))
    def __datasetCommonParams(self):
        root_dataset_dir = ''
        tmp = {
            'mosi':{
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss',
                    'a_dim': 375,
                    'v_dim': 500,
                    't_dim': 50,
                    'd_out': 64
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/unaligned_50.pkl'),
                    'seq_lens': None,
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss',
                    'a_dim': 375,
                    'v_dim': 500,
                    't_dim': 50,
                    'd_out': 64
                },
                'aligned_missing': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss',
                    'missing_seed': (111, 1111, 11111),
                    'a_dim': 375,
                    'v_dim': 500,
                    't_dim': 50,
                    'd_out': 64
                },
                'unaligned_missing': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/unaligned_50.pkl'),
                    'seq_lens': None,
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss',
                    'missing_seed': (111, 1111, 11111),
                    'a_dim': 375,
                    'v_dim': 500,
                    't_dim': 50,
                    'd_out': 64
                }
            },
            'sims':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'SIMS/Processed/unaligned_39.pkl'),
                    'seq_lens': (39, 400, 55), 
                    'feature_dims': (768, 33, 709), 
                    'train_samples': 1368,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'Loss',
                    'a_dim': 400,
                    'v_dim': 55,
                    't_dim': 39,
                    'd_out': 64
                },
                'unaligned_missing': {
                    'dataPath': os.path.join(root_dataset_dir, 'SIMS/Processed/unaligned_39.pkl'),
                    'seq_lens': None,
                    'feature_dims': (768, 33, 709), 
                    'train_samples': 1368,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'Loss',
                    'missing_seed': (111, 1111, 11111),
                    'a_dim': 400,
                    'v_dim': 55,
                    't_dim': 39,
                    'd_out': 64
                }
            }
        }
        return tmp
    def __MIT-FRNet(self): ##########
        tmp = {
            'commonParas':{
                'data_missing': True,
                'deal_missing': True,
                'need_data_aligned': False,
                'alignmentModule': 'crossmodal_attn',
                'generatorModule': 'linear',
                'fusionModule': 'c_gate',
                'recloss_type': 'combine',
                'without_generator': False,

                'early_stop': 6,
                'use_bert': True,
                'use_bert_finetune': True,
                'attn_mask': True, 
                'update_epochs': 4,
            },
            'datasetParas':{
                'mosi':{
                    'text_dropout': 0.2, 
                    'conv1d_kernel_size_l': 1, 
                    'conv1d_kernel_size_a': 5,
                    'conv1d_kernel_size_v': 3,
                    'attn_dropout': 0.2, 
                    'attn_dropout_a': 0.1,
                    'attn_dropout_v': 0.0,
                    'relu_dropout': 0.2,
                    'embed_dropout': 0.2,
                    'res_dropout': 0.2,
                    'dst_feature_dim_nheads': (30, 6), 
                    'nlevels': 3,
                    'fusion_t_in': 90,
                    'fusion_a_in': 90,
                    'fusion_v_in': 90,
                    'fusion_t_hid': 36,
                    'fusion_a_hid': 20,
                    'fusion_v_hid': 48,
                    'fusion_gru_layers': 3, 
                    'use_linear': True, 
                    'fusion_drop': 0.2, 
                    'cls_hidden_dim': 128,
                    'cls_dropout': 0.0,
                    'grad_clip': 0.8, 
                    'batch_size': 24,
                    'learning_rate_bert': 1e-05,
                    'learning_rate_other': 0.002,
                    'patience': 5,
                    'weight_decay_bert': 0.0001,
                    'weight_decay_other': 0.001,
                    'weight_gen_loss': (5, 2, 20),
                },
                'sims':{
                    'text_dropout': 0.2, 
                    'conv1d_kernel_size_l': 1, 
                    'conv1d_kernel_size_a': 5,
                    'conv1d_kernel_size_v': 3,
                    'attn_dropout': 0.2, 
                    'attn_dropout_a': 0.1,
                    'attn_dropout_v': 0.0,
                    'relu_dropout': 0.0,
                    'embed_dropout': 0.1,
                    'res_dropout': 0.2,
                    'dst_feature_dim_nheads': (30, 6), 
                    'nlevels': 2,
                    'trans_hid_t': 40,
                    'trans_hid_t_drop': 0.0,
                    'trans_hid_a': 80,
                    'trans_hid_a_drop': 0.1,
                    'trans_hid_v': 48,
                    'trans_hid_v_drop': 0.3,
                    'fusion_t_in': 90,
                    'fusion_a_in': 90,
                    'fusion_v_in': 90,
                    'fusion_t_hid': 36,
                    'fusion_a_hid': 20,
                    'fusion_v_hid': 48,
                    'fusion_gru_layers': 3, 
                    'use_linear': True, 
                    'fusion_drop': 0.2, 
                    'cls_hidden_dim': 128,
                    'cls_dropout': 0.1,
                    'grad_clip': 0.8, 
                    'batch_size': 16,
                    'learning_rate_bert': 1e-05,
                    'learning_rate_other': 0.002,
                    'patience': 10,
                    'weight_decay_bert': 0.0001,
                    'weight_decay_other': 0.001,
                    'weight_gen_loss': (1, 0.01, 0.0001),
                },
            },
        }
        return tmp
    def get_config(self):
        return self.args
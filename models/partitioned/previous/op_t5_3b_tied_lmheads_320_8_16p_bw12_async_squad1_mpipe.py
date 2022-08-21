import torch
import torch.linalg
import torch.functional
import math
import torch.nn.functional
import torch.fft
from torch import Tensor
import torch.nn as nn
from itertools import chain
from typing import Optional, Tuple, Iterator, Iterable, OrderedDict, Dict
import collections

from typing import Type
from torch.nn.modules.sparse import Embedding
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from models.normal.NLP_models.stateless import StatelessEmbedding
from models.new_t5_example.modeling_t5 import T5LayerNorm
# this is an auto generated file do not edit unless you know what you are doing


# partition adjacency
# model inputs {0, 8, 15}
# partition 0 {'inputs': {'attention_mask', 'input_ids', 'decoder_input_ids', 'decoder_attention_mask'}, 'outputs': {8, 1}}
# partition 1 {'inputs': {0}, 'outputs': {2}}
# partition 2 {'inputs': {1}, 'outputs': {3}}
# partition 3 {'inputs': {2}, 'outputs': {4}}
# partition 4 {'inputs': {3}, 'outputs': {5}}
# partition 5 {'inputs': {4}, 'outputs': {6}}
# partition 6 {'inputs': {5}, 'outputs': {7}}
# partition 7 {'inputs': {6}, 'outputs': {8}}
# partition 8 {'inputs': {'attention_mask', 0, 'decoder_input_ids', 7}, 'outputs': {9, 10, 11, 12, 13, 14, 15}}
# partition 9 {'inputs': {8}, 'outputs': {10}}
# partition 10 {'inputs': {8, 9}, 'outputs': {11}}
# partition 11 {'inputs': {8, 10}, 'outputs': {12}}
# partition 12 {'inputs': {8, 11}, 'outputs': {13}}
# partition 13 {'inputs': {8, 12}, 'outputs': {14}}
# partition 14 {'inputs': {8, 13}, 'outputs': {15}}
# partition 15 {'inputs': {8, 'labels', 14}, 'outputs': {'output'}}
# model outputs {15}


def create_pipeline_configuration(DEBUG=False, batch_size=8):
    config = {
        'batch_dim': 0,
        'depth': 10000,
        'basic_blocks': (Embedding,Linear,Dropout,StatelessEmbedding,T5LayerNorm),
        'model_inputs': {
            'attention_mask': {
                'shape': torch.Size([8, 320]),
                'dtype': torch.int64,
                'is_batched': True,
                'used_by': [0, 8]},
            'decoder_attention_mask': {
                'shape': torch.Size([8, 8]),
                'dtype': torch.int64,
                'is_batched': True,
                'used_by': [0]},
            'decoder_input_ids': {
                'shape': torch.Size([8, 8]),
                'dtype': torch.int64,
                'is_batched': True,
                'used_by': [0, 8]},
            'input_ids': {
                'shape': torch.Size([8, 320]),
                'dtype': torch.int64,
                'is_batched': True,
                'used_by': [0]},
            'labels': {
                'shape': torch.Size([8, 8]),
                'dtype': torch.int64,
                'is_batched': True,
                'used_by': [15]}},
        'model_outputs': {
            'T5ForConditionalGeneration/torch.nn.functional::cross_entropy_5820': {
                'shape': torch.Size([1]),
                'dtype': torch.float32,
                'is_batched': False,
                'created_by': 15}},
        'stages': {
            0: {
                'stage_cls': Partition0,
                'inputs': {
                    'attention_mask': {
                        'shape': torch.Size([8, 320]),
                        'dtype': torch.int64,
                        'req_grad': False,
                        'is_batched': True,
                        'created_by': -1},
                    'decoder_attention_mask': {
                        'shape': torch.Size([8, 8]),
                        'dtype': torch.int64,
                        'req_grad': False,
                        'is_batched': True,
                        'created_by': -1},
                    'decoder_input_ids': {
                        'shape': torch.Size([8, 8]),
                        'dtype': torch.int64,
                        'req_grad': False,
                        'is_batched': True,
                        'created_by': -1},
                    'input_ids': {
                        'shape': torch.Size([8, 320]),
                        'dtype': torch.int64,
                        'req_grad': False,
                        'is_batched': True,
                        'created_by': -1}},
                'outputs': {
                    'T5ForConditionalGeneration/Parameter[shared_embed_weight]': {
                        'shape': torch.Size([32100, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': False,
                        'used_by': [8]},
                    'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/tuple::__getitem___304_0': {
                        'shape': torch.Size([8, 32, 320, 320]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [1]},
                    'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/prim::TupleConstruct_313_0': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [1]},
                    'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/prim::TupleConstruct_313_1': {
                        'shape': None,
                        'dtype': None,
                        'req_grad': False,
                        'is_batched': False,
                        'used_by': [1]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/Size::__getitem___2087': {
                        'shape': None,
                        'dtype': int,
                        'req_grad': False,
                        'is_batched': False,
                        'used_by': [8]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/Tensor::__mul___2117': {
                        'shape': torch.Size([8, 1, 8, 8]),
                        'dtype': torch.float32,
                        'req_grad': False,
                        'is_batched': True,
                        'used_by': [8]}},
                'devices': ['cpu' if DEBUG else 'cuda:0'],
                'stage_depth': 15},
            1: {
                'stage_cls': Partition1,
                'inputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/tuple::__getitem___304_0': {
                        'shape': torch.Size([8, 32, 320, 320]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 0},
                    'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/prim::TupleConstruct_313_0': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 0},
                    'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/prim::TupleConstruct_313_1': {
                        'shape': None,
                        'dtype': None,
                        'req_grad': False,
                        'is_batched': False,
                        'created_by': 0}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___570': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [2]},
                    'T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___572': {
                        'shape': torch.Size([8, 32, 320, 320]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [2]}},
                'devices': ['cpu' if DEBUG else 'cuda:1'],
                'stage_depth': 14},
            2: {
                'stage_cls': Partition2,
                'inputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___570': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 1},
                    'T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___572': {
                        'shape': torch.Size([8, 32, 320, 320]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 1}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___822': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [3]},
                    'T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___824': {
                        'shape': torch.Size([8, 32, 320, 320]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [3]}},
                'devices': ['cpu' if DEBUG else 'cuda:2'],
                'stage_depth': 13},
            3: {
                'stage_cls': Partition3,
                'inputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___822': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 2},
                    'T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___824': {
                        'shape': torch.Size([8, 32, 320, 320]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 2}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1074': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [4]},
                    'T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1076': {
                        'shape': torch.Size([8, 32, 320, 320]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [4]}},
                'devices': ['cpu' if DEBUG else 'cuda:3'],
                'stage_depth': 12},
            4: {
                'stage_cls': Partition4,
                'inputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1074': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 3},
                    'T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1076': {
                        'shape': torch.Size([8, 32, 320, 320]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 3}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1326': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [5]},
                    'T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1328': {
                        'shape': torch.Size([8, 32, 320, 320]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [5]}},
                'devices': ['cpu' if DEBUG else 'cuda:4'],
                'stage_depth': 11},
            5: {
                'stage_cls': Partition5,
                'inputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1326': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 4},
                    'T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1328': {
                        'shape': torch.Size([8, 32, 320, 320]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 4}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1578': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [6]},
                    'T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1580': {
                        'shape': torch.Size([8, 32, 320, 320]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [6]}},
                'devices': ['cpu' if DEBUG else 'cuda:5'],
                'stage_depth': 10},
            6: {
                'stage_cls': Partition6,
                'inputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1578': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 5},
                    'T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1580': {
                        'shape': torch.Size([8, 32, 320, 320]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 5}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1830': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [7]},
                    'T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1832': {
                        'shape': torch.Size([8, 32, 320, 320]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [7]}},
                'devices': ['cpu' if DEBUG else 'cuda:6'],
                'stage_depth': 9},
            7: {
                'stage_cls': Partition7,
                'inputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1830': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 6},
                    'T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1832': {
                        'shape': torch.Size([8, 32, 320, 320]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 6}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5LayerNorm[final_layer_norm]': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [8]}},
                'devices': ['cpu' if DEBUG else 'cuda:7'],
                'stage_depth': 8},
            8: {
                'stage_cls': Partition8,
                'inputs': {
                    'attention_mask': {
                        'shape': torch.Size([8, 320]),
                        'dtype': torch.int64,
                        'req_grad': False,
                        'is_batched': True,
                        'created_by': -1},
                    'decoder_input_ids': {
                        'shape': torch.Size([8, 8]),
                        'dtype': torch.int64,
                        'req_grad': False,
                        'is_batched': True,
                        'created_by': -1},
                    'T5ForConditionalGeneration/Parameter[shared_embed_weight]': {
                        'shape': torch.Size([32100, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': False,
                        'created_by': 0},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5LayerNorm[final_layer_norm]': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 7},
                    'T5ForConditionalGeneration/T5Stack[decoder]/Size::__getitem___2087': {
                        'shape': None,
                        'dtype': int,
                        'req_grad': False,
                        'is_batched': False,
                        'created_by': 0},
                    'T5ForConditionalGeneration/T5Stack[decoder]/Tensor::__mul___2117': {
                        'shape': torch.Size([8, 1, 8, 8]),
                        'dtype': torch.float32,
                        'req_grad': False,
                        'is_batched': True,
                        'created_by': 0}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_9': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [9]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___2632': {
                        'shape': torch.Size([8, 8, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [9]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___2634': {
                        'shape': torch.Size([8, 32, 8, 8]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [9]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___2636': {
                        'shape': torch.Size([8, 32, 8, 320]),
                        'dtype': torch.float32,
                        'req_grad': False,
                        'is_batched': True,
                        'used_by': [9]}},
                'devices': ['cpu' if DEBUG else 'cuda:8'],
                'stage_depth': 7},
            9: {
                'stage_cls': Partition9,
                'inputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_9': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 8},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___2632': {
                        'shape': torch.Size([8, 8, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 8},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___2634': {
                        'shape': torch.Size([8, 32, 8, 8]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 8},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___2636': {
                        'shape': torch.Size([8, 32, 8, 320]),
                        'dtype': torch.float32,
                        'req_grad': False,
                        'is_batched': True,
                        'created_by': 8}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_10': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [10]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3085': {
                        'shape': torch.Size([8, 8, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [10]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3087': {
                        'shape': torch.Size([8, 32, 8, 8]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [10]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3089': {
                        'shape': torch.Size([8, 32, 8, 320]),
                        'dtype': torch.float32,
                        'req_grad': False,
                        'is_batched': True,
                        'used_by': [10]}},
                'devices': ['cpu' if DEBUG else 'cuda:9'],
                'stage_depth': 6},
            10: {
                'stage_cls': Partition10,
                'inputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_10': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 9},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3085': {
                        'shape': torch.Size([8, 8, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 9},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3087': {
                        'shape': torch.Size([8, 32, 8, 8]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 9},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3089': {
                        'shape': torch.Size([8, 32, 8, 320]),
                        'dtype': torch.float32,
                        'req_grad': False,
                        'is_batched': True,
                        'created_by': 9}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_11': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [11]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3538': {
                        'shape': torch.Size([8, 8, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [11]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3540': {
                        'shape': torch.Size([8, 32, 8, 8]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [11]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3542': {
                        'shape': torch.Size([8, 32, 8, 320]),
                        'dtype': torch.float32,
                        'req_grad': False,
                        'is_batched': True,
                        'used_by': [11]}},
                'devices': ['cpu' if DEBUG else 'cuda:10'],
                'stage_depth': 5},
            11: {
                'stage_cls': Partition11,
                'inputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_11': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 10},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3538': {
                        'shape': torch.Size([8, 8, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 10},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3540': {
                        'shape': torch.Size([8, 32, 8, 8]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 10},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3542': {
                        'shape': torch.Size([8, 32, 8, 320]),
                        'dtype': torch.float32,
                        'req_grad': False,
                        'is_batched': True,
                        'created_by': 10}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_12': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [12]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3991': {
                        'shape': torch.Size([8, 8, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [12]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3993': {
                        'shape': torch.Size([8, 32, 8, 8]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [12]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3995': {
                        'shape': torch.Size([8, 32, 8, 320]),
                        'dtype': torch.float32,
                        'req_grad': False,
                        'is_batched': True,
                        'used_by': [12]}},
                'devices': ['cpu' if DEBUG else 'cuda:11'],
                'stage_depth': 4},
            12: {
                'stage_cls': Partition12,
                'inputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_12': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 11},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3991': {
                        'shape': torch.Size([8, 8, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 11},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3993': {
                        'shape': torch.Size([8, 32, 8, 8]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 11},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3995': {
                        'shape': torch.Size([8, 32, 8, 320]),
                        'dtype': torch.float32,
                        'req_grad': False,
                        'is_batched': True,
                        'created_by': 11}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_13': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [13]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___4444': {
                        'shape': torch.Size([8, 8, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [13]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___4446': {
                        'shape': torch.Size([8, 32, 8, 8]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [13]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___4448': {
                        'shape': torch.Size([8, 32, 8, 320]),
                        'dtype': torch.float32,
                        'req_grad': False,
                        'is_batched': True,
                        'used_by': [13]}},
                'devices': ['cpu' if DEBUG else 'cuda:12'],
                'stage_depth': 3},
            13: {
                'stage_cls': Partition13,
                'inputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_13': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 12},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___4444': {
                        'shape': torch.Size([8, 8, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 12},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___4446': {
                        'shape': torch.Size([8, 32, 8, 8]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 12},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___4448': {
                        'shape': torch.Size([8, 32, 8, 320]),
                        'dtype': torch.float32,
                        'req_grad': False,
                        'is_batched': True,
                        'created_by': 12}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_14': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [14]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___4897': {
                        'shape': torch.Size([8, 8, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [14]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___4899': {
                        'shape': torch.Size([8, 32, 8, 8]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [14]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___4901': {
                        'shape': torch.Size([8, 32, 8, 320]),
                        'dtype': torch.float32,
                        'req_grad': False,
                        'is_batched': True,
                        'used_by': [14]}},
                'devices': ['cpu' if DEBUG else 'cuda:13'],
                'stage_depth': 2},
            14: {
                'stage_cls': Partition14,
                'inputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_14': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 13},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___4897': {
                        'shape': torch.Size([8, 8, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 13},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___4899': {
                        'shape': torch.Size([8, 32, 8, 8]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 13},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___4901': {
                        'shape': torch.Size([8, 32, 8, 320]),
                        'dtype': torch.float32,
                        'req_grad': False,
                        'is_batched': True,
                        'created_by': 13}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_15': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [15]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___5350': {
                        'shape': torch.Size([8, 8, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [15]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___5352': {
                        'shape': torch.Size([8, 32, 8, 8]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [15]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___5354': {
                        'shape': torch.Size([8, 32, 8, 320]),
                        'dtype': torch.float32,
                        'req_grad': False,
                        'is_batched': True,
                        'used_by': [15]}},
                'devices': ['cpu' if DEBUG else 'cuda:14'],
                'stage_depth': 1},
            15: {
                'stage_cls': Partition15,
                'inputs': {
                    'labels': {
                        'shape': torch.Size([8, 8]),
                        'dtype': torch.int64,
                        'req_grad': False,
                        'is_batched': True,
                        'created_by': -1},
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_15': {
                        'shape': torch.Size([8, 320, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 14},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___5350': {
                        'shape': torch.Size([8, 8, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 14},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___5352': {
                        'shape': torch.Size([8, 32, 8, 8]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 14},
                    'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___5354': {
                        'shape': torch.Size([8, 32, 8, 320]),
                        'dtype': torch.float32,
                        'req_grad': False,
                        'is_batched': True,
                        'created_by': 14}},
                'outputs': {
                    'T5ForConditionalGeneration/torch.nn.functional::cross_entropy_5820': {
                        'shape': torch.Size([1]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': False,
                        'used_by': [-1]}},
                'devices': ['cpu' if DEBUG else 'cuda:15'],
                'stage_depth': 0}},
        'stage_to_device_map': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}
    
    
    # switching batch size
    batch_dim = config['batch_dim']
    for d in chain(config['model_inputs'].values(),config['model_outputs'].values()):
        if d['is_batched']:
            shape = d['shape']
            d['shape'] = torch.Size(shape[:batch_dim] + (batch_size,) + shape[batch_dim+1:])
    
    for s in config['stages'].values():
        for d in chain(s['inputs'].values(),s['outputs'].values()):
            if d['is_batched']:
                shape = d['shape']
                d['shape'] = torch.Size(shape[:batch_dim] + (batch_size,) + shape[batch_dim+1:])
    
    return config

class Partition0(nn.Module):
    LAYER_SCOPES = [
            'T5ForConditionalGeneration/T5Stack[encoder]/StatelessEmbedding[embed_tokens]',
            'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Embedding[relative_attention_bias]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout]',
        ]
    TENSORS = [
            'T5ForConditionalGeneration/Parameter[shared_embed_weight]',
        ]
    def __init__(self, layers, tensors, device='cuda:0'):
        super().__init__()

        # Initialize partition layers
        for idx, layer_scope in enumerate(self.LAYER_SCOPES):
            self.add_module(f'l_{idx}' ,layers[layer_scope])

        # Initialize partition tensors (params and buffs)
        b = p = 0
        for tensor_scope in self.TENSORS:
            tensor = tensors[tensor_scope]
            if isinstance(tensor, nn.Parameter):
                self.register_parameter(f'p_{p}', tensor)
                p += 1
            else:
                self.register_buffer(f'b_{b}', tensor)
                b += 1

        self.device = torch.device(device)
        self.input_structure = [1, 1, 1, 1]
        self.lookup = {'l_0': 'encoder.embed_tokens',
                        'l_1': 'encoder.dropout',
                        'l_2': 'encoder.block.0.layer.0.layer_norm',
                        'l_3': 'encoder.block.0.layer.0.SelfAttention.q',
                        'l_4': 'encoder.block.0.layer.0.SelfAttention.k',
                        'l_5': 'encoder.block.0.layer.0.SelfAttention.v',
                        'l_6': 'encoder.block.0.layer.0.SelfAttention.relative_attention_bias',
                        'l_7': 'encoder.block.0.layer.0.SelfAttention.dropout',
                        'l_8': 'encoder.block.0.layer.0.SelfAttention.o',
                        'l_9': 'encoder.block.0.layer.0.dropout',
                        'l_10': 'encoder.block.0.layer.1.layer_norm',
                        'l_11': 'encoder.block.0.layer.1.DenseReluDense.wi',
                        'l_12': 'encoder.block.0.layer.1.DenseReluDense.dropout',
                        'l_13': 'encoder.block.0.layer.1.DenseReluDense.wo',
                        'l_14': 'encoder.block.0.layer.1.dropout',
                        'l_15': 'encoder.block.1.layer.0.layer_norm',
                        'l_16': 'encoder.block.1.layer.0.SelfAttention.q',
                        'l_17': 'encoder.block.1.layer.0.SelfAttention.k',
                        'l_18': 'encoder.block.1.layer.0.SelfAttention.v',
                        'l_19': 'encoder.block.1.layer.0.SelfAttention.dropout',
                        'l_20': 'encoder.block.1.layer.0.SelfAttention.o',
                        'l_21': 'encoder.block.1.layer.0.dropout',
                        'l_22': 'encoder.block.1.layer.1.layer_norm',
                        'l_23': 'encoder.block.1.layer.1.DenseReluDense.wi',
                        'l_24': 'encoder.block.1.layer.1.DenseReluDense.dropout',
                        'l_25': 'encoder.block.1.layer.1.DenseReluDense.wo',
                        'l_26': 'encoder.block.1.layer.1.dropout',
                        'l_27': 'encoder.block.2.layer.0.layer_norm',
                        'l_28': 'encoder.block.2.layer.0.SelfAttention.q',
                        'l_29': 'encoder.block.2.layer.0.SelfAttention.k',
                        'l_30': 'encoder.block.2.layer.0.SelfAttention.v',
                        'l_31': 'encoder.block.2.layer.0.SelfAttention.dropout',
                        'l_32': 'encoder.block.2.layer.0.SelfAttention.o',
                        'l_33': 'encoder.block.2.layer.0.dropout',
                        'l_34': 'encoder.block.2.layer.1.layer_norm',
                        'l_35': 'encoder.block.2.layer.1.DenseReluDense.wi',
                        'l_36': 'encoder.block.2.layer.1.DenseReluDense.dropout',
                        'l_37': 'encoder.block.2.layer.1.DenseReluDense.wo',
                        'l_38': 'encoder.block.2.layer.1.dropout',
                        'p_0': 'shared_embed_weight'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[encoder]/StatelessEmbedding[embed_tokens] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Embedding[relative_attention_bias] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_36
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_37
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout] <=> self.l_38
        # T5ForConditionalGeneration/Parameter[shared_embed_weight] <=> self.p_0
        # input0 <=> attention_mask
        # input1 <=> decoder_attention_mask
        # input2 <=> decoder_input_ids
        # input3 <=> input_ids
        attention_mask, decoder_attention_mask, decoder_input_ids, input_ids = unflatten(args, self.input_structure)
        t_0 = decoder_input_ids.size()
        t_1 = input_ids.size()
        t_1 = t_1[-1]
        t_1 = input_ids.view(-1, t_1)
        t_1 = self.l_0(self.p_0, t_1)
        t_1 = self.l_1(t_1)
        t_2 = attention_mask[(slice(None, None, None), None, None, slice(None, None, None))]
        t_2 = t_2.to(dtype=torch.float32)
        t_2 = 1.0 - t_2
        t_2 = t_2 * -10000.0
        t_3 = self.l_2(t_1)
        t_4 = self.l_3(t_3)
        t_5 = self.l_4(t_3)
        t_6 = self.l_5(t_3)
        t_3 = t_3.shape
        t_3 = t_3[slice(None, 2, None)]
        t_7 = t_3[0]
        t_3 = t_3[1]
        t_4 = t_4.view(t_7, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_5 = t_5.view(t_7, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_6 = t_6.view(t_7, -1, 32, 128)
        t_6 = t_6.transpose(1, 2)
        t_5 = t_5.transpose(3, 2)
        t_5 = torch.matmul(t_4, t_5)
        t_4 = torch.arange(t_3, dtype=torch.int64, device=self.device)
        t_4 = t_4[(slice(None, None, None), None)]
        t_3 = torch.arange(t_3, dtype=torch.int64, device=self.device)
        t_3 = t_3[(None, slice(None, None, None))]
        t_4 = t_3 - t_4
        t_3 = torch.abs(t_4)
        t_4 = t_4 > 0
        t_4 = t_4.to(torch.int64)
        t_4 = t_4 * 16
        t_4 = 0 + t_4
        t_8 = t_3.float()
        t_9 = t_3 < 8
        t_8 = t_8 / 8
        t_8 = torch.log(t_8)
        t_10 = math.log(16.0)
        t_10 = t_8 / t_10
        t_10 = t_10 * 8
        t_10 = t_10.to(torch.int64)
        t_10 = 8 + t_10
        t_8 = torch.full_like(t_10, 15, device=self.device)
        t_8 = torch.min(t_10, t_8)
        t_8 = torch.where(t_9, t_3, t_8)
        t_4 += t_8
        t_8 = t_4
        t_8 = t_8.to(self.device)
        t_8 = self.l_6(t_8)
        t_8 = t_8.permute([2, 0, 1])
        t_8 = t_8.unsqueeze(0)
        t_2 = t_8 + t_2
        t_5 += t_2
        t_8 = t_5.float()
        t_8 = torch.nn.functional.softmax(t_8, dim=-1, _stacklevel=3, dtype=None)
        t_5 = t_8.type_as(t_5)
        t_5 = self.l_7(t_5)
        t_6 = torch.matmul(t_5, t_6)
        t_6 = t_6.transpose(1, 2)
        t_6 = t_6.contiguous()
        t_7 = t_6.view(t_7, -1, 4096)
        t_7 = self.l_8(t_7)
        t_6 = self.l_9(t_7)
        t_6 = t_1 + t_6
        t_2 = (t_7, None, t_2)
        t_6 = (t_6,)
        t_2 = t_2[slice(1, None, None)]
        t_2 = t_6 + t_2
        t_6 = t_2[slice(None, 2, None)]
        t_7 = t_6[0]
        t_1 = self.l_10(t_7)
        t_6 = t_6[1]
        t_2 = t_2[slice(2, None, None)]
        t_1 = self.l_11(t_1)
        t_1 = torch.nn.functional.relu(t_1, inplace=False)
        t_1 = self.l_12(t_1)
        t_1 = self.l_13(t_1)
        t_1 = self.l_14(t_1)
        t_1 = t_7 + t_1
        t_6 = (t_1, t_6)
        t_2 = t_6 + t_2
        t_6 = t_2[slice(None, 2, None)]
        t_6 = t_6[0]
        t_1 = self.l_15(t_6)
        t_2 = t_2[2]
        t_7 = self.l_16(t_1)
        t_5 = self.l_17(t_1)
        t_8 = self.l_18(t_1)
        t_1 = t_1.shape
        t_1 = t_1[slice(None, 2, None)]
        t_1 = t_1[0]
        t_7 = t_7.view(t_1, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_5 = t_5.view(t_1, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_8 = t_8.view(t_1, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_5 = t_5.transpose(3, 2)
        t_5 = torch.matmul(t_7, t_5)
        t_5 += t_2
        t_7 = t_5.float()
        t_7 = torch.nn.functional.softmax(t_7, dim=-1, _stacklevel=3, dtype=None)
        t_5 = t_7.type_as(t_5)
        t_5 = self.l_19(t_5)
        t_8 = torch.matmul(t_5, t_8)
        t_8 = t_8.transpose(1, 2)
        t_8 = t_8.contiguous()
        t_1 = t_8.view(t_1, -1, 4096)
        t_1 = self.l_20(t_1)
        t_8 = self.l_21(t_1)
        t_8 = t_6 + t_8
        t_2 = (t_1, None, t_2)
        t_8 = (t_8,)
        t_2 = t_2[slice(1, None, None)]
        t_2 = t_8 + t_2
        t_8 = t_2[slice(None, 2, None)]
        t_1 = t_8[0]
        t_6 = self.l_22(t_1)
        t_8 = t_8[1]
        t_2 = t_2[slice(2, None, None)]
        t_6 = self.l_23(t_6)
        t_6 = torch.nn.functional.relu(t_6, inplace=False)
        t_6 = self.l_24(t_6)
        t_6 = self.l_25(t_6)
        t_6 = self.l_26(t_6)
        t_6 = t_1 + t_6
        t_8 = (t_6, t_8)
        t_2 = t_8 + t_2
        t_8 = t_2[slice(None, 2, None)]
        t_8 = t_8[0]
        t_6 = self.l_27(t_8)
        t_2 = t_2[2]
        t_1 = self.l_28(t_6)
        t_5 = self.l_29(t_6)
        t_7 = self.l_30(t_6)
        t_6 = t_6.shape
        t_6 = t_6[slice(None, 2, None)]
        t_6 = t_6[0]
        t_1 = t_1.view(t_6, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_5 = t_5.view(t_6, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_7 = t_7.view(t_6, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_5 = t_5.transpose(3, 2)
        t_5 = torch.matmul(t_1, t_5)
        t_5 += t_2
        t_1 = t_5.float()
        t_1 = torch.nn.functional.softmax(t_1, dim=-1, _stacklevel=3, dtype=None)
        t_5 = t_1.type_as(t_5)
        t_5 = self.l_31(t_5)
        t_7 = torch.matmul(t_5, t_7)
        t_7 = t_7.transpose(1, 2)
        t_7 = t_7.contiguous()
        t_6 = t_7.view(t_6, -1, 4096)
        t_6 = self.l_32(t_6)
        t_7 = self.l_33(t_6)
        t_7 = t_8 + t_7
        t_2 = (t_6, None, t_2)
        t_7 = (t_7,)
        t_2 = t_2[slice(1, None, None)]
        t_2 = t_7 + t_2
        t_7 = t_2[slice(None, 2, None)]
        t_6 = t_7[0]
        t_8 = self.l_34(t_6)
        t_7 = t_7[1]
        t_2 = t_2[slice(2, None, None)]
        t_8 = self.l_35(t_8)
        t_8 = torch.nn.functional.relu(t_8, inplace=False)
        t_8 = self.l_36(t_8)
        t_8 = self.l_37(t_8)
        t_8 = self.l_38(t_8)
        t_8 = t_6 + t_8
        t_7 = (t_8, t_7)
        t_8 = t_0[-1]
        t_6 = t_0[0]
        t_0 = t_0[1]
        t_5 = torch.arange(t_0, device=self.device)
        t_1 = t_5[(None, None, slice(None, None, None))]
        t_0 = t_1.repeat(t_6, t_0, 1)
        t_5 = t_5[(None, slice(None, None, None), None)]
        t_5 = t_0 <= t_5
        t_0 = decoder_attention_mask.dtype
        t_0 = t_5.to(t_0)
        t_0 = t_0[(slice(None, None, None), None, slice(None, None, None), slice(None, None, None))]
        t_5 = decoder_attention_mask[(slice(None, None, None), None, None, slice(None, None, None))]
        t_5 = t_0 * t_5
        t_5 = t_5.to(dtype=torch.float32)
        t_5 = 1.0 - t_5
        t_5 = t_5 * -10000.0
        # Returning:
        # T5ForConditionalGeneration/Parameter[shared_embed_weight]
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/tuple::__getitem___304
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/prim::TupleConstruct_313
        # T5ForConditionalGeneration/T5Stack[decoder]/Size::__getitem___2087
        # T5ForConditionalGeneration/T5Stack[decoder]/Tensor::__mul___2117
        return list(flatten((self.p_0, t_2, t_7, t_8, t_5)))

    def state_dict(self, *args, **kwargs):
        # we return the state dict of this part as it should be in the original model
        return state_dict(self, *args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return load_state_dict(self, *args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        # we return the named parameters of this part as it should be in the original model
        return named_parameters(self, *args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        # we return the named buffers of this part as it should be in the original model
        return named_buffers(self, *args, **kwargs)

    def cpu(self):
        return cpu(self)

    def cuda(self, device=None):
        return cuda(self, device=device)

    def to(self, *args, **kwargs):
        return to(self, *args, **kwargs)


class Partition1(nn.Module):
    LAYER_SCOPES = [
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout]',
        ]
    TENSORS = [
        ]
    def __init__(self, layers, tensors, device='cuda:1'):
        super().__init__()

        # Initialize partition layers
        for idx, layer_scope in enumerate(self.LAYER_SCOPES):
            self.add_module(f'l_{idx}' ,layers[layer_scope])

        # Initialize partition tensors (params and buffs)
        b = p = 0
        for tensor_scope in self.TENSORS:
            tensor = tensors[tensor_scope]
            if isinstance(tensor, nn.Parameter):
                self.register_parameter(f'p_{p}', tensor)
                p += 1
            else:
                self.register_buffer(f'b_{b}', tensor)
                b += 1

        self.device = torch.device(device)
        self.input_structure = [(1,), (1, 1)]
        self.lookup = {'l_0': 'encoder.block.3.layer.0.layer_norm',
                        'l_1': 'encoder.block.3.layer.0.SelfAttention.q',
                        'l_2': 'encoder.block.3.layer.0.SelfAttention.k',
                        'l_3': 'encoder.block.3.layer.0.SelfAttention.v',
                        'l_4': 'encoder.block.3.layer.0.SelfAttention.dropout',
                        'l_5': 'encoder.block.3.layer.0.SelfAttention.o',
                        'l_6': 'encoder.block.3.layer.0.dropout',
                        'l_7': 'encoder.block.3.layer.1.layer_norm',
                        'l_8': 'encoder.block.3.layer.1.DenseReluDense.wi',
                        'l_9': 'encoder.block.3.layer.1.DenseReluDense.dropout',
                        'l_10': 'encoder.block.3.layer.1.DenseReluDense.wo',
                        'l_11': 'encoder.block.3.layer.1.dropout',
                        'l_12': 'encoder.block.4.layer.0.layer_norm',
                        'l_13': 'encoder.block.4.layer.0.SelfAttention.q',
                        'l_14': 'encoder.block.4.layer.0.SelfAttention.k',
                        'l_15': 'encoder.block.4.layer.0.SelfAttention.v',
                        'l_16': 'encoder.block.4.layer.0.SelfAttention.dropout',
                        'l_17': 'encoder.block.4.layer.0.SelfAttention.o',
                        'l_18': 'encoder.block.4.layer.0.dropout',
                        'l_19': 'encoder.block.4.layer.1.layer_norm',
                        'l_20': 'encoder.block.4.layer.1.DenseReluDense.wi',
                        'l_21': 'encoder.block.4.layer.1.DenseReluDense.dropout',
                        'l_22': 'encoder.block.4.layer.1.DenseReluDense.wo',
                        'l_23': 'encoder.block.4.layer.1.dropout',
                        'l_24': 'encoder.block.5.layer.0.layer_norm',
                        'l_25': 'encoder.block.5.layer.0.SelfAttention.q',
                        'l_26': 'encoder.block.5.layer.0.SelfAttention.k',
                        'l_27': 'encoder.block.5.layer.0.SelfAttention.v',
                        'l_28': 'encoder.block.5.layer.0.SelfAttention.dropout',
                        'l_29': 'encoder.block.5.layer.0.SelfAttention.o',
                        'l_30': 'encoder.block.5.layer.0.dropout',
                        'l_31': 'encoder.block.5.layer.1.layer_norm',
                        'l_32': 'encoder.block.5.layer.1.DenseReluDense.wi',
                        'l_33': 'encoder.block.5.layer.1.DenseReluDense.dropout',
                        'l_34': 'encoder.block.5.layer.1.DenseReluDense.wo',
                        'l_35': 'encoder.block.5.layer.1.dropout'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/tuple::__getitem___304 <=> x0
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[2]/prim::TupleConstruct_313 <=> x1
        x0, x1 = unflatten(args, self.input_structure)
        t_0 = x1 + x0
        t_1 = t_0[slice(None, 2, None)]
        t_1 = t_1[0]
        t_2 = self.l_0(t_1)
        t_0 = t_0[2]
        t_3 = self.l_1(t_2)
        t_4 = self.l_2(t_2)
        t_5 = self.l_3(t_2)
        t_2 = t_2.shape
        t_2 = t_2[slice(None, 2, None)]
        t_2 = t_2[0]
        t_3 = t_3.view(t_2, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_4 = t_4.view(t_2, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_5 = t_5.view(t_2, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_3, t_4)
        t_4 += t_0
        t_3 = t_4.float()
        t_3 = torch.nn.functional.softmax(t_3, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_3.type_as(t_4)
        t_4 = self.l_4(t_4)
        t_5 = torch.matmul(t_4, t_5)
        t_5 = t_5.transpose(1, 2)
        t_5 = t_5.contiguous()
        t_2 = t_5.view(t_2, -1, 4096)
        t_2 = self.l_5(t_2)
        t_5 = self.l_6(t_2)
        t_5 = t_1 + t_5
        t_0 = (t_2, None, t_0)
        t_5 = (t_5,)
        t_0 = t_0[slice(1, None, None)]
        t_0 = t_5 + t_0
        t_5 = t_0[slice(None, 2, None)]
        t_2 = t_5[0]
        t_1 = self.l_7(t_2)
        t_5 = t_5[1]
        t_0 = t_0[slice(2, None, None)]
        t_1 = self.l_8(t_1)
        t_1 = torch.nn.functional.relu(t_1, inplace=False)
        t_1 = self.l_9(t_1)
        t_1 = self.l_10(t_1)
        t_1 = self.l_11(t_1)
        t_1 = t_2 + t_1
        t_5 = (t_1, t_5)
        t_0 = t_5 + t_0
        t_5 = t_0[slice(None, 2, None)]
        t_5 = t_5[0]
        t_1 = self.l_12(t_5)
        t_0 = t_0[2]
        t_2 = self.l_13(t_1)
        t_4 = self.l_14(t_1)
        t_3 = self.l_15(t_1)
        t_1 = t_1.shape
        t_1 = t_1[slice(None, 2, None)]
        t_1 = t_1[0]
        t_2 = t_2.view(t_1, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_4 = t_4.view(t_1, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_3 = t_3.view(t_1, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_2, t_4)
        t_4 += t_0
        t_2 = t_4.float()
        t_2 = torch.nn.functional.softmax(t_2, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_2.type_as(t_4)
        t_4 = self.l_16(t_4)
        t_3 = torch.matmul(t_4, t_3)
        t_3 = t_3.transpose(1, 2)
        t_3 = t_3.contiguous()
        t_1 = t_3.view(t_1, -1, 4096)
        t_1 = self.l_17(t_1)
        t_3 = self.l_18(t_1)
        t_3 = t_5 + t_3
        t_0 = (t_1, None, t_0)
        t_3 = (t_3,)
        t_0 = t_0[slice(1, None, None)]
        t_0 = t_3 + t_0
        t_3 = t_0[slice(None, 2, None)]
        t_1 = t_3[0]
        t_5 = self.l_19(t_1)
        t_3 = t_3[1]
        t_0 = t_0[slice(2, None, None)]
        t_5 = self.l_20(t_5)
        t_5 = torch.nn.functional.relu(t_5, inplace=False)
        t_5 = self.l_21(t_5)
        t_5 = self.l_22(t_5)
        t_5 = self.l_23(t_5)
        t_5 = t_1 + t_5
        t_3 = (t_5, t_3)
        t_0 = t_3 + t_0
        t_3 = t_0[slice(None, 2, None)]
        t_3 = t_3[0]
        t_5 = self.l_24(t_3)
        t_0 = t_0[2]
        t_1 = self.l_25(t_5)
        t_4 = self.l_26(t_5)
        t_2 = self.l_27(t_5)
        t_5 = t_5.shape
        t_5 = t_5[slice(None, 2, None)]
        t_5 = t_5[0]
        t_1 = t_1.view(t_5, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_4 = t_4.view(t_5, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_2 = t_2.view(t_5, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_1, t_4)
        t_4 += t_0
        t_1 = t_4.float()
        t_1 = torch.nn.functional.softmax(t_1, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_1.type_as(t_4)
        t_4 = self.l_28(t_4)
        t_2 = torch.matmul(t_4, t_2)
        t_2 = t_2.transpose(1, 2)
        t_2 = t_2.contiguous()
        t_5 = t_2.view(t_5, -1, 4096)
        t_5 = self.l_29(t_5)
        t_2 = self.l_30(t_5)
        t_2 = t_3 + t_2
        t_0 = (t_5, None, t_0)
        t_2 = (t_2,)
        t_0 = t_0[slice(1, None, None)]
        t_0 = t_2 + t_0
        t_2 = t_0[slice(None, 2, None)]
        t_5 = t_2[0]
        t_3 = self.l_31(t_5)
        t_2 = t_2[1]
        t_0 = t_0[slice(2, None, None)]
        t_3 = self.l_32(t_3)
        t_3 = torch.nn.functional.relu(t_3, inplace=False)
        t_3 = self.l_33(t_3)
        t_3 = self.l_34(t_3)
        t_3 = self.l_35(t_3)
        t_3 = t_5 + t_3
        t_2 = (t_3, t_2)
        t_0 = t_2 + t_0
        t_2 = t_0[slice(None, 2, None)]
        t_2 = t_2[0]
        t_0 = t_0[2]
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___570
        # T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___572
        return list(flatten((t_2, t_0)))

    def state_dict(self, *args, **kwargs):
        # we return the state dict of this part as it should be in the original model
        return state_dict(self, *args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return load_state_dict(self, *args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        # we return the named parameters of this part as it should be in the original model
        return named_parameters(self, *args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        # we return the named buffers of this part as it should be in the original model
        return named_buffers(self, *args, **kwargs)

    def cpu(self):
        return cpu(self)

    def cuda(self, device=None):
        return cuda(self, device=device)

    def to(self, *args, **kwargs):
        return to(self, *args, **kwargs)


class Partition2(nn.Module):
    LAYER_SCOPES = [
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout]',
        ]
    TENSORS = [
        ]
    def __init__(self, layers, tensors, device='cuda:2'):
        super().__init__()

        # Initialize partition layers
        for idx, layer_scope in enumerate(self.LAYER_SCOPES):
            self.add_module(f'l_{idx}' ,layers[layer_scope])

        # Initialize partition tensors (params and buffs)
        b = p = 0
        for tensor_scope in self.TENSORS:
            tensor = tensors[tensor_scope]
            if isinstance(tensor, nn.Parameter):
                self.register_parameter(f'p_{p}', tensor)
                p += 1
            else:
                self.register_buffer(f'b_{b}', tensor)
                b += 1

        self.device = torch.device(device)
        self.input_structure = [1, 1]
        self.lookup = {'l_0': 'encoder.block.6.layer.0.layer_norm',
                        'l_1': 'encoder.block.6.layer.0.SelfAttention.q',
                        'l_2': 'encoder.block.6.layer.0.SelfAttention.k',
                        'l_3': 'encoder.block.6.layer.0.SelfAttention.v',
                        'l_4': 'encoder.block.6.layer.0.SelfAttention.dropout',
                        'l_5': 'encoder.block.6.layer.0.SelfAttention.o',
                        'l_6': 'encoder.block.6.layer.0.dropout',
                        'l_7': 'encoder.block.6.layer.1.layer_norm',
                        'l_8': 'encoder.block.6.layer.1.DenseReluDense.wi',
                        'l_9': 'encoder.block.6.layer.1.DenseReluDense.dropout',
                        'l_10': 'encoder.block.6.layer.1.DenseReluDense.wo',
                        'l_11': 'encoder.block.6.layer.1.dropout',
                        'l_12': 'encoder.block.7.layer.0.layer_norm',
                        'l_13': 'encoder.block.7.layer.0.SelfAttention.q',
                        'l_14': 'encoder.block.7.layer.0.SelfAttention.k',
                        'l_15': 'encoder.block.7.layer.0.SelfAttention.v',
                        'l_16': 'encoder.block.7.layer.0.SelfAttention.dropout',
                        'l_17': 'encoder.block.7.layer.0.SelfAttention.o',
                        'l_18': 'encoder.block.7.layer.0.dropout',
                        'l_19': 'encoder.block.7.layer.1.layer_norm',
                        'l_20': 'encoder.block.7.layer.1.DenseReluDense.wi',
                        'l_21': 'encoder.block.7.layer.1.DenseReluDense.dropout',
                        'l_22': 'encoder.block.7.layer.1.DenseReluDense.wo',
                        'l_23': 'encoder.block.7.layer.1.dropout',
                        'l_24': 'encoder.block.8.layer.0.layer_norm',
                        'l_25': 'encoder.block.8.layer.0.SelfAttention.q',
                        'l_26': 'encoder.block.8.layer.0.SelfAttention.k',
                        'l_27': 'encoder.block.8.layer.0.SelfAttention.v',
                        'l_28': 'encoder.block.8.layer.0.SelfAttention.dropout',
                        'l_29': 'encoder.block.8.layer.0.SelfAttention.o',
                        'l_30': 'encoder.block.8.layer.0.dropout',
                        'l_31': 'encoder.block.8.layer.1.layer_norm',
                        'l_32': 'encoder.block.8.layer.1.DenseReluDense.wi',
                        'l_33': 'encoder.block.8.layer.1.DenseReluDense.dropout',
                        'l_34': 'encoder.block.8.layer.1.DenseReluDense.wo',
                        'l_35': 'encoder.block.8.layer.1.dropout'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___570 <=> x0
        # T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___572 <=> x1
        x0, x1 = unflatten(args, self.input_structure)
        t_0 = self.l_0(x0)
        t_1 = self.l_1(t_0)
        t_2 = self.l_2(t_0)
        t_3 = self.l_3(t_0)
        t_0 = t_0.shape
        t_0 = t_0[slice(None, 2, None)]
        t_0 = t_0[0]
        t_1 = t_1.view(t_0, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_2 = t_2.view(t_0, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_3 = t_3.view(t_0, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_2 = t_2.transpose(3, 2)
        t_2 = torch.matmul(t_1, t_2)
        t_2 += x1
        t_1 = t_2.float()
        t_1 = torch.nn.functional.softmax(t_1, dim=-1, _stacklevel=3, dtype=None)
        t_2 = t_1.type_as(t_2)
        t_2 = self.l_4(t_2)
        t_3 = torch.matmul(t_2, t_3)
        t_3 = t_3.transpose(1, 2)
        t_3 = t_3.contiguous()
        t_0 = t_3.view(t_0, -1, 4096)
        t_0 = self.l_5(t_0)
        t_3 = self.l_6(t_0)
        t_3 = x0 + t_3
        t_0 = (t_0, None, x1)
        t_3 = (t_3,)
        t_0 = t_0[slice(1, None, None)]
        t_0 = t_3 + t_0
        t_3 = t_0[slice(None, 2, None)]
        t_2 = t_3[0]
        t_1 = self.l_7(t_2)
        t_3 = t_3[1]
        t_0 = t_0[slice(2, None, None)]
        t_1 = self.l_8(t_1)
        t_1 = torch.nn.functional.relu(t_1, inplace=False)
        t_1 = self.l_9(t_1)
        t_1 = self.l_10(t_1)
        t_1 = self.l_11(t_1)
        t_1 = t_2 + t_1
        t_3 = (t_1, t_3)
        t_0 = t_3 + t_0
        t_3 = t_0[slice(None, 2, None)]
        t_3 = t_3[0]
        t_1 = self.l_12(t_3)
        t_0 = t_0[2]
        t_2 = self.l_13(t_1)
        t_4 = self.l_14(t_1)
        t_5 = self.l_15(t_1)
        t_1 = t_1.shape
        t_1 = t_1[slice(None, 2, None)]
        t_1 = t_1[0]
        t_2 = t_2.view(t_1, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_4 = t_4.view(t_1, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_5 = t_5.view(t_1, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_2, t_4)
        t_4 += t_0
        t_2 = t_4.float()
        t_2 = torch.nn.functional.softmax(t_2, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_2.type_as(t_4)
        t_4 = self.l_16(t_4)
        t_5 = torch.matmul(t_4, t_5)
        t_5 = t_5.transpose(1, 2)
        t_5 = t_5.contiguous()
        t_1 = t_5.view(t_1, -1, 4096)
        t_1 = self.l_17(t_1)
        t_5 = self.l_18(t_1)
        t_5 = t_3 + t_5
        t_0 = (t_1, None, t_0)
        t_5 = (t_5,)
        t_0 = t_0[slice(1, None, None)]
        t_0 = t_5 + t_0
        t_5 = t_0[slice(None, 2, None)]
        t_1 = t_5[0]
        t_3 = self.l_19(t_1)
        t_5 = t_5[1]
        t_0 = t_0[slice(2, None, None)]
        t_3 = self.l_20(t_3)
        t_3 = torch.nn.functional.relu(t_3, inplace=False)
        t_3 = self.l_21(t_3)
        t_3 = self.l_22(t_3)
        t_3 = self.l_23(t_3)
        t_3 = t_1 + t_3
        t_5 = (t_3, t_5)
        t_0 = t_5 + t_0
        t_5 = t_0[slice(None, 2, None)]
        t_5 = t_5[0]
        t_3 = self.l_24(t_5)
        t_0 = t_0[2]
        t_1 = self.l_25(t_3)
        t_4 = self.l_26(t_3)
        t_2 = self.l_27(t_3)
        t_3 = t_3.shape
        t_3 = t_3[slice(None, 2, None)]
        t_3 = t_3[0]
        t_1 = t_1.view(t_3, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_4 = t_4.view(t_3, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_2 = t_2.view(t_3, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_1, t_4)
        t_4 += t_0
        t_1 = t_4.float()
        t_1 = torch.nn.functional.softmax(t_1, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_1.type_as(t_4)
        t_4 = self.l_28(t_4)
        t_2 = torch.matmul(t_4, t_2)
        t_2 = t_2.transpose(1, 2)
        t_2 = t_2.contiguous()
        t_3 = t_2.view(t_3, -1, 4096)
        t_3 = self.l_29(t_3)
        t_2 = self.l_30(t_3)
        t_2 = t_5 + t_2
        t_0 = (t_3, None, t_0)
        t_2 = (t_2,)
        t_0 = t_0[slice(1, None, None)]
        t_0 = t_2 + t_0
        t_2 = t_0[slice(None, 2, None)]
        t_3 = t_2[0]
        t_5 = self.l_31(t_3)
        t_2 = t_2[1]
        t_0 = t_0[slice(2, None, None)]
        t_5 = self.l_32(t_5)
        t_5 = torch.nn.functional.relu(t_5, inplace=False)
        t_5 = self.l_33(t_5)
        t_5 = self.l_34(t_5)
        t_5 = self.l_35(t_5)
        t_5 = t_3 + t_5
        t_2 = (t_5, t_2)
        t_0 = t_2 + t_0
        t_2 = t_0[slice(None, 2, None)]
        t_2 = t_2[0]
        t_0 = t_0[2]
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___822
        # T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___824
        return list(flatten((t_2, t_0)))

    def state_dict(self, *args, **kwargs):
        # we return the state dict of this part as it should be in the original model
        return state_dict(self, *args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return load_state_dict(self, *args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        # we return the named parameters of this part as it should be in the original model
        return named_parameters(self, *args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        # we return the named buffers of this part as it should be in the original model
        return named_buffers(self, *args, **kwargs)

    def cpu(self):
        return cpu(self)

    def cuda(self, device=None):
        return cuda(self, device=device)

    def to(self, *args, **kwargs):
        return to(self, *args, **kwargs)


class Partition3(nn.Module):
    LAYER_SCOPES = [
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout]',
        ]
    TENSORS = [
        ]
    def __init__(self, layers, tensors, device='cuda:3'):
        super().__init__()

        # Initialize partition layers
        for idx, layer_scope in enumerate(self.LAYER_SCOPES):
            self.add_module(f'l_{idx}' ,layers[layer_scope])

        # Initialize partition tensors (params and buffs)
        b = p = 0
        for tensor_scope in self.TENSORS:
            tensor = tensors[tensor_scope]
            if isinstance(tensor, nn.Parameter):
                self.register_parameter(f'p_{p}', tensor)
                p += 1
            else:
                self.register_buffer(f'b_{b}', tensor)
                b += 1

        self.device = torch.device(device)
        self.input_structure = [1, 1]
        self.lookup = {'l_0': 'encoder.block.9.layer.0.layer_norm',
                        'l_1': 'encoder.block.9.layer.0.SelfAttention.q',
                        'l_2': 'encoder.block.9.layer.0.SelfAttention.k',
                        'l_3': 'encoder.block.9.layer.0.SelfAttention.v',
                        'l_4': 'encoder.block.9.layer.0.SelfAttention.dropout',
                        'l_5': 'encoder.block.9.layer.0.SelfAttention.o',
                        'l_6': 'encoder.block.9.layer.0.dropout',
                        'l_7': 'encoder.block.9.layer.1.layer_norm',
                        'l_8': 'encoder.block.9.layer.1.DenseReluDense.wi',
                        'l_9': 'encoder.block.9.layer.1.DenseReluDense.dropout',
                        'l_10': 'encoder.block.9.layer.1.DenseReluDense.wo',
                        'l_11': 'encoder.block.9.layer.1.dropout',
                        'l_12': 'encoder.block.10.layer.0.layer_norm',
                        'l_13': 'encoder.block.10.layer.0.SelfAttention.q',
                        'l_14': 'encoder.block.10.layer.0.SelfAttention.k',
                        'l_15': 'encoder.block.10.layer.0.SelfAttention.v',
                        'l_16': 'encoder.block.10.layer.0.SelfAttention.dropout',
                        'l_17': 'encoder.block.10.layer.0.SelfAttention.o',
                        'l_18': 'encoder.block.10.layer.0.dropout',
                        'l_19': 'encoder.block.10.layer.1.layer_norm',
                        'l_20': 'encoder.block.10.layer.1.DenseReluDense.wi',
                        'l_21': 'encoder.block.10.layer.1.DenseReluDense.dropout',
                        'l_22': 'encoder.block.10.layer.1.DenseReluDense.wo',
                        'l_23': 'encoder.block.10.layer.1.dropout',
                        'l_24': 'encoder.block.11.layer.0.layer_norm',
                        'l_25': 'encoder.block.11.layer.0.SelfAttention.q',
                        'l_26': 'encoder.block.11.layer.0.SelfAttention.k',
                        'l_27': 'encoder.block.11.layer.0.SelfAttention.v',
                        'l_28': 'encoder.block.11.layer.0.SelfAttention.dropout',
                        'l_29': 'encoder.block.11.layer.0.SelfAttention.o',
                        'l_30': 'encoder.block.11.layer.0.dropout',
                        'l_31': 'encoder.block.11.layer.1.layer_norm',
                        'l_32': 'encoder.block.11.layer.1.DenseReluDense.wi',
                        'l_33': 'encoder.block.11.layer.1.DenseReluDense.dropout',
                        'l_34': 'encoder.block.11.layer.1.DenseReluDense.wo',
                        'l_35': 'encoder.block.11.layer.1.dropout'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___822 <=> x0
        # T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___824 <=> x1
        x0, x1 = unflatten(args, self.input_structure)
        t_0 = self.l_0(x0)
        t_1 = self.l_1(t_0)
        t_2 = self.l_2(t_0)
        t_3 = self.l_3(t_0)
        t_0 = t_0.shape
        t_0 = t_0[slice(None, 2, None)]
        t_0 = t_0[0]
        t_1 = t_1.view(t_0, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_2 = t_2.view(t_0, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_3 = t_3.view(t_0, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_2 = t_2.transpose(3, 2)
        t_2 = torch.matmul(t_1, t_2)
        t_2 += x1
        t_1 = t_2.float()
        t_1 = torch.nn.functional.softmax(t_1, dim=-1, _stacklevel=3, dtype=None)
        t_2 = t_1.type_as(t_2)
        t_2 = self.l_4(t_2)
        t_3 = torch.matmul(t_2, t_3)
        t_3 = t_3.transpose(1, 2)
        t_3 = t_3.contiguous()
        t_0 = t_3.view(t_0, -1, 4096)
        t_0 = self.l_5(t_0)
        t_3 = self.l_6(t_0)
        t_3 = x0 + t_3
        t_0 = (t_0, None, x1)
        t_3 = (t_3,)
        t_0 = t_0[slice(1, None, None)]
        t_0 = t_3 + t_0
        t_3 = t_0[slice(None, 2, None)]
        t_2 = t_3[0]
        t_1 = self.l_7(t_2)
        t_3 = t_3[1]
        t_0 = t_0[slice(2, None, None)]
        t_1 = self.l_8(t_1)
        t_1 = torch.nn.functional.relu(t_1, inplace=False)
        t_1 = self.l_9(t_1)
        t_1 = self.l_10(t_1)
        t_1 = self.l_11(t_1)
        t_1 = t_2 + t_1
        t_3 = (t_1, t_3)
        t_0 = t_3 + t_0
        t_3 = t_0[slice(None, 2, None)]
        t_3 = t_3[0]
        t_1 = self.l_12(t_3)
        t_0 = t_0[2]
        t_2 = self.l_13(t_1)
        t_4 = self.l_14(t_1)
        t_5 = self.l_15(t_1)
        t_1 = t_1.shape
        t_1 = t_1[slice(None, 2, None)]
        t_1 = t_1[0]
        t_2 = t_2.view(t_1, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_4 = t_4.view(t_1, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_5 = t_5.view(t_1, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_2, t_4)
        t_4 += t_0
        t_2 = t_4.float()
        t_2 = torch.nn.functional.softmax(t_2, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_2.type_as(t_4)
        t_4 = self.l_16(t_4)
        t_5 = torch.matmul(t_4, t_5)
        t_5 = t_5.transpose(1, 2)
        t_5 = t_5.contiguous()
        t_1 = t_5.view(t_1, -1, 4096)
        t_1 = self.l_17(t_1)
        t_5 = self.l_18(t_1)
        t_5 = t_3 + t_5
        t_0 = (t_1, None, t_0)
        t_5 = (t_5,)
        t_0 = t_0[slice(1, None, None)]
        t_0 = t_5 + t_0
        t_5 = t_0[slice(None, 2, None)]
        t_1 = t_5[0]
        t_3 = self.l_19(t_1)
        t_5 = t_5[1]
        t_0 = t_0[slice(2, None, None)]
        t_3 = self.l_20(t_3)
        t_3 = torch.nn.functional.relu(t_3, inplace=False)
        t_3 = self.l_21(t_3)
        t_3 = self.l_22(t_3)
        t_3 = self.l_23(t_3)
        t_3 = t_1 + t_3
        t_5 = (t_3, t_5)
        t_0 = t_5 + t_0
        t_5 = t_0[slice(None, 2, None)]
        t_5 = t_5[0]
        t_3 = self.l_24(t_5)
        t_0 = t_0[2]
        t_1 = self.l_25(t_3)
        t_4 = self.l_26(t_3)
        t_2 = self.l_27(t_3)
        t_3 = t_3.shape
        t_3 = t_3[slice(None, 2, None)]
        t_3 = t_3[0]
        t_1 = t_1.view(t_3, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_4 = t_4.view(t_3, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_2 = t_2.view(t_3, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_1, t_4)
        t_4 += t_0
        t_1 = t_4.float()
        t_1 = torch.nn.functional.softmax(t_1, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_1.type_as(t_4)
        t_4 = self.l_28(t_4)
        t_2 = torch.matmul(t_4, t_2)
        t_2 = t_2.transpose(1, 2)
        t_2 = t_2.contiguous()
        t_3 = t_2.view(t_3, -1, 4096)
        t_3 = self.l_29(t_3)
        t_2 = self.l_30(t_3)
        t_2 = t_5 + t_2
        t_0 = (t_3, None, t_0)
        t_2 = (t_2,)
        t_0 = t_0[slice(1, None, None)]
        t_0 = t_2 + t_0
        t_2 = t_0[slice(None, 2, None)]
        t_3 = t_2[0]
        t_5 = self.l_31(t_3)
        t_2 = t_2[1]
        t_0 = t_0[slice(2, None, None)]
        t_5 = self.l_32(t_5)
        t_5 = torch.nn.functional.relu(t_5, inplace=False)
        t_5 = self.l_33(t_5)
        t_5 = self.l_34(t_5)
        t_5 = self.l_35(t_5)
        t_5 = t_3 + t_5
        t_2 = (t_5, t_2)
        t_0 = t_2 + t_0
        t_2 = t_0[slice(None, 2, None)]
        t_2 = t_2[0]
        t_0 = t_0[2]
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1074
        # T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1076
        return list(flatten((t_2, t_0)))

    def state_dict(self, *args, **kwargs):
        # we return the state dict of this part as it should be in the original model
        return state_dict(self, *args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return load_state_dict(self, *args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        # we return the named parameters of this part as it should be in the original model
        return named_parameters(self, *args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        # we return the named buffers of this part as it should be in the original model
        return named_buffers(self, *args, **kwargs)

    def cpu(self):
        return cpu(self)

    def cuda(self, device=None):
        return cuda(self, device=device)

    def to(self, *args, **kwargs):
        return to(self, *args, **kwargs)


class Partition4(nn.Module):
    LAYER_SCOPES = [
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout]',
        ]
    TENSORS = [
        ]
    def __init__(self, layers, tensors, device='cuda:4'):
        super().__init__()

        # Initialize partition layers
        for idx, layer_scope in enumerate(self.LAYER_SCOPES):
            self.add_module(f'l_{idx}' ,layers[layer_scope])

        # Initialize partition tensors (params and buffs)
        b = p = 0
        for tensor_scope in self.TENSORS:
            tensor = tensors[tensor_scope]
            if isinstance(tensor, nn.Parameter):
                self.register_parameter(f'p_{p}', tensor)
                p += 1
            else:
                self.register_buffer(f'b_{b}', tensor)
                b += 1

        self.device = torch.device(device)
        self.input_structure = [1, 1]
        self.lookup = {'l_0': 'encoder.block.12.layer.0.layer_norm',
                        'l_1': 'encoder.block.12.layer.0.SelfAttention.q',
                        'l_2': 'encoder.block.12.layer.0.SelfAttention.k',
                        'l_3': 'encoder.block.12.layer.0.SelfAttention.v',
                        'l_4': 'encoder.block.12.layer.0.SelfAttention.dropout',
                        'l_5': 'encoder.block.12.layer.0.SelfAttention.o',
                        'l_6': 'encoder.block.12.layer.0.dropout',
                        'l_7': 'encoder.block.12.layer.1.layer_norm',
                        'l_8': 'encoder.block.12.layer.1.DenseReluDense.wi',
                        'l_9': 'encoder.block.12.layer.1.DenseReluDense.dropout',
                        'l_10': 'encoder.block.12.layer.1.DenseReluDense.wo',
                        'l_11': 'encoder.block.12.layer.1.dropout',
                        'l_12': 'encoder.block.13.layer.0.layer_norm',
                        'l_13': 'encoder.block.13.layer.0.SelfAttention.q',
                        'l_14': 'encoder.block.13.layer.0.SelfAttention.k',
                        'l_15': 'encoder.block.13.layer.0.SelfAttention.v',
                        'l_16': 'encoder.block.13.layer.0.SelfAttention.dropout',
                        'l_17': 'encoder.block.13.layer.0.SelfAttention.o',
                        'l_18': 'encoder.block.13.layer.0.dropout',
                        'l_19': 'encoder.block.13.layer.1.layer_norm',
                        'l_20': 'encoder.block.13.layer.1.DenseReluDense.wi',
                        'l_21': 'encoder.block.13.layer.1.DenseReluDense.dropout',
                        'l_22': 'encoder.block.13.layer.1.DenseReluDense.wo',
                        'l_23': 'encoder.block.13.layer.1.dropout',
                        'l_24': 'encoder.block.14.layer.0.layer_norm',
                        'l_25': 'encoder.block.14.layer.0.SelfAttention.q',
                        'l_26': 'encoder.block.14.layer.0.SelfAttention.k',
                        'l_27': 'encoder.block.14.layer.0.SelfAttention.v',
                        'l_28': 'encoder.block.14.layer.0.SelfAttention.dropout',
                        'l_29': 'encoder.block.14.layer.0.SelfAttention.o',
                        'l_30': 'encoder.block.14.layer.0.dropout',
                        'l_31': 'encoder.block.14.layer.1.layer_norm',
                        'l_32': 'encoder.block.14.layer.1.DenseReluDense.wi',
                        'l_33': 'encoder.block.14.layer.1.DenseReluDense.dropout',
                        'l_34': 'encoder.block.14.layer.1.DenseReluDense.wo',
                        'l_35': 'encoder.block.14.layer.1.dropout'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1074 <=> x0
        # T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1076 <=> x1
        x0, x1 = unflatten(args, self.input_structure)
        t_0 = self.l_0(x0)
        t_1 = self.l_1(t_0)
        t_2 = self.l_2(t_0)
        t_3 = self.l_3(t_0)
        t_0 = t_0.shape
        t_0 = t_0[slice(None, 2, None)]
        t_0 = t_0[0]
        t_1 = t_1.view(t_0, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_2 = t_2.view(t_0, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_3 = t_3.view(t_0, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_2 = t_2.transpose(3, 2)
        t_2 = torch.matmul(t_1, t_2)
        t_2 += x1
        t_1 = t_2.float()
        t_1 = torch.nn.functional.softmax(t_1, dim=-1, _stacklevel=3, dtype=None)
        t_2 = t_1.type_as(t_2)
        t_2 = self.l_4(t_2)
        t_3 = torch.matmul(t_2, t_3)
        t_3 = t_3.transpose(1, 2)
        t_3 = t_3.contiguous()
        t_0 = t_3.view(t_0, -1, 4096)
        t_0 = self.l_5(t_0)
        t_3 = self.l_6(t_0)
        t_3 = x0 + t_3
        t_0 = (t_0, None, x1)
        t_3 = (t_3,)
        t_0 = t_0[slice(1, None, None)]
        t_0 = t_3 + t_0
        t_3 = t_0[slice(None, 2, None)]
        t_2 = t_3[0]
        t_1 = self.l_7(t_2)
        t_3 = t_3[1]
        t_0 = t_0[slice(2, None, None)]
        t_1 = self.l_8(t_1)
        t_1 = torch.nn.functional.relu(t_1, inplace=False)
        t_1 = self.l_9(t_1)
        t_1 = self.l_10(t_1)
        t_1 = self.l_11(t_1)
        t_1 = t_2 + t_1
        t_3 = (t_1, t_3)
        t_0 = t_3 + t_0
        t_3 = t_0[slice(None, 2, None)]
        t_3 = t_3[0]
        t_1 = self.l_12(t_3)
        t_0 = t_0[2]
        t_2 = self.l_13(t_1)
        t_4 = self.l_14(t_1)
        t_5 = self.l_15(t_1)
        t_1 = t_1.shape
        t_1 = t_1[slice(None, 2, None)]
        t_1 = t_1[0]
        t_2 = t_2.view(t_1, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_4 = t_4.view(t_1, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_5 = t_5.view(t_1, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_2, t_4)
        t_4 += t_0
        t_2 = t_4.float()
        t_2 = torch.nn.functional.softmax(t_2, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_2.type_as(t_4)
        t_4 = self.l_16(t_4)
        t_5 = torch.matmul(t_4, t_5)
        t_5 = t_5.transpose(1, 2)
        t_5 = t_5.contiguous()
        t_1 = t_5.view(t_1, -1, 4096)
        t_1 = self.l_17(t_1)
        t_5 = self.l_18(t_1)
        t_5 = t_3 + t_5
        t_0 = (t_1, None, t_0)
        t_5 = (t_5,)
        t_0 = t_0[slice(1, None, None)]
        t_0 = t_5 + t_0
        t_5 = t_0[slice(None, 2, None)]
        t_1 = t_5[0]
        t_3 = self.l_19(t_1)
        t_5 = t_5[1]
        t_0 = t_0[slice(2, None, None)]
        t_3 = self.l_20(t_3)
        t_3 = torch.nn.functional.relu(t_3, inplace=False)
        t_3 = self.l_21(t_3)
        t_3 = self.l_22(t_3)
        t_3 = self.l_23(t_3)
        t_3 = t_1 + t_3
        t_5 = (t_3, t_5)
        t_0 = t_5 + t_0
        t_5 = t_0[slice(None, 2, None)]
        t_5 = t_5[0]
        t_3 = self.l_24(t_5)
        t_0 = t_0[2]
        t_1 = self.l_25(t_3)
        t_4 = self.l_26(t_3)
        t_2 = self.l_27(t_3)
        t_3 = t_3.shape
        t_3 = t_3[slice(None, 2, None)]
        t_3 = t_3[0]
        t_1 = t_1.view(t_3, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_4 = t_4.view(t_3, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_2 = t_2.view(t_3, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_1, t_4)
        t_4 += t_0
        t_1 = t_4.float()
        t_1 = torch.nn.functional.softmax(t_1, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_1.type_as(t_4)
        t_4 = self.l_28(t_4)
        t_2 = torch.matmul(t_4, t_2)
        t_2 = t_2.transpose(1, 2)
        t_2 = t_2.contiguous()
        t_3 = t_2.view(t_3, -1, 4096)
        t_3 = self.l_29(t_3)
        t_2 = self.l_30(t_3)
        t_2 = t_5 + t_2
        t_0 = (t_3, None, t_0)
        t_2 = (t_2,)
        t_0 = t_0[slice(1, None, None)]
        t_0 = t_2 + t_0
        t_2 = t_0[slice(None, 2, None)]
        t_3 = t_2[0]
        t_5 = self.l_31(t_3)
        t_2 = t_2[1]
        t_0 = t_0[slice(2, None, None)]
        t_5 = self.l_32(t_5)
        t_5 = torch.nn.functional.relu(t_5, inplace=False)
        t_5 = self.l_33(t_5)
        t_5 = self.l_34(t_5)
        t_5 = self.l_35(t_5)
        t_5 = t_3 + t_5
        t_2 = (t_5, t_2)
        t_0 = t_2 + t_0
        t_2 = t_0[slice(None, 2, None)]
        t_2 = t_2[0]
        t_0 = t_0[2]
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1326
        # T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1328
        return list(flatten((t_2, t_0)))

    def state_dict(self, *args, **kwargs):
        # we return the state dict of this part as it should be in the original model
        return state_dict(self, *args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return load_state_dict(self, *args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        # we return the named parameters of this part as it should be in the original model
        return named_parameters(self, *args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        # we return the named buffers of this part as it should be in the original model
        return named_buffers(self, *args, **kwargs)

    def cpu(self):
        return cpu(self)

    def cuda(self, device=None):
        return cuda(self, device=device)

    def to(self, *args, **kwargs):
        return to(self, *args, **kwargs)


class Partition5(nn.Module):
    LAYER_SCOPES = [
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout]',
        ]
    TENSORS = [
        ]
    def __init__(self, layers, tensors, device='cuda:5'):
        super().__init__()

        # Initialize partition layers
        for idx, layer_scope in enumerate(self.LAYER_SCOPES):
            self.add_module(f'l_{idx}' ,layers[layer_scope])

        # Initialize partition tensors (params and buffs)
        b = p = 0
        for tensor_scope in self.TENSORS:
            tensor = tensors[tensor_scope]
            if isinstance(tensor, nn.Parameter):
                self.register_parameter(f'p_{p}', tensor)
                p += 1
            else:
                self.register_buffer(f'b_{b}', tensor)
                b += 1

        self.device = torch.device(device)
        self.input_structure = [1, 1]
        self.lookup = {'l_0': 'encoder.block.15.layer.0.layer_norm',
                        'l_1': 'encoder.block.15.layer.0.SelfAttention.q',
                        'l_2': 'encoder.block.15.layer.0.SelfAttention.k',
                        'l_3': 'encoder.block.15.layer.0.SelfAttention.v',
                        'l_4': 'encoder.block.15.layer.0.SelfAttention.dropout',
                        'l_5': 'encoder.block.15.layer.0.SelfAttention.o',
                        'l_6': 'encoder.block.15.layer.0.dropout',
                        'l_7': 'encoder.block.15.layer.1.layer_norm',
                        'l_8': 'encoder.block.15.layer.1.DenseReluDense.wi',
                        'l_9': 'encoder.block.15.layer.1.DenseReluDense.dropout',
                        'l_10': 'encoder.block.15.layer.1.DenseReluDense.wo',
                        'l_11': 'encoder.block.15.layer.1.dropout',
                        'l_12': 'encoder.block.16.layer.0.layer_norm',
                        'l_13': 'encoder.block.16.layer.0.SelfAttention.q',
                        'l_14': 'encoder.block.16.layer.0.SelfAttention.k',
                        'l_15': 'encoder.block.16.layer.0.SelfAttention.v',
                        'l_16': 'encoder.block.16.layer.0.SelfAttention.dropout',
                        'l_17': 'encoder.block.16.layer.0.SelfAttention.o',
                        'l_18': 'encoder.block.16.layer.0.dropout',
                        'l_19': 'encoder.block.16.layer.1.layer_norm',
                        'l_20': 'encoder.block.16.layer.1.DenseReluDense.wi',
                        'l_21': 'encoder.block.16.layer.1.DenseReluDense.dropout',
                        'l_22': 'encoder.block.16.layer.1.DenseReluDense.wo',
                        'l_23': 'encoder.block.16.layer.1.dropout',
                        'l_24': 'encoder.block.17.layer.0.layer_norm',
                        'l_25': 'encoder.block.17.layer.0.SelfAttention.q',
                        'l_26': 'encoder.block.17.layer.0.SelfAttention.k',
                        'l_27': 'encoder.block.17.layer.0.SelfAttention.v',
                        'l_28': 'encoder.block.17.layer.0.SelfAttention.dropout',
                        'l_29': 'encoder.block.17.layer.0.SelfAttention.o',
                        'l_30': 'encoder.block.17.layer.0.dropout',
                        'l_31': 'encoder.block.17.layer.1.layer_norm',
                        'l_32': 'encoder.block.17.layer.1.DenseReluDense.wi',
                        'l_33': 'encoder.block.17.layer.1.DenseReluDense.dropout',
                        'l_34': 'encoder.block.17.layer.1.DenseReluDense.wo',
                        'l_35': 'encoder.block.17.layer.1.dropout'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1326 <=> x0
        # T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1328 <=> x1
        x0, x1 = unflatten(args, self.input_structure)
        t_0 = self.l_0(x0)
        t_1 = self.l_1(t_0)
        t_2 = self.l_2(t_0)
        t_3 = self.l_3(t_0)
        t_0 = t_0.shape
        t_0 = t_0[slice(None, 2, None)]
        t_0 = t_0[0]
        t_1 = t_1.view(t_0, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_2 = t_2.view(t_0, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_3 = t_3.view(t_0, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_2 = t_2.transpose(3, 2)
        t_2 = torch.matmul(t_1, t_2)
        t_2 += x1
        t_1 = t_2.float()
        t_1 = torch.nn.functional.softmax(t_1, dim=-1, _stacklevel=3, dtype=None)
        t_2 = t_1.type_as(t_2)
        t_2 = self.l_4(t_2)
        t_3 = torch.matmul(t_2, t_3)
        t_3 = t_3.transpose(1, 2)
        t_3 = t_3.contiguous()
        t_0 = t_3.view(t_0, -1, 4096)
        t_0 = self.l_5(t_0)
        t_3 = self.l_6(t_0)
        t_3 = x0 + t_3
        t_0 = (t_0, None, x1)
        t_3 = (t_3,)
        t_0 = t_0[slice(1, None, None)]
        t_0 = t_3 + t_0
        t_3 = t_0[slice(None, 2, None)]
        t_2 = t_3[0]
        t_1 = self.l_7(t_2)
        t_3 = t_3[1]
        t_0 = t_0[slice(2, None, None)]
        t_1 = self.l_8(t_1)
        t_1 = torch.nn.functional.relu(t_1, inplace=False)
        t_1 = self.l_9(t_1)
        t_1 = self.l_10(t_1)
        t_1 = self.l_11(t_1)
        t_1 = t_2 + t_1
        t_3 = (t_1, t_3)
        t_0 = t_3 + t_0
        t_3 = t_0[slice(None, 2, None)]
        t_3 = t_3[0]
        t_1 = self.l_12(t_3)
        t_0 = t_0[2]
        t_2 = self.l_13(t_1)
        t_4 = self.l_14(t_1)
        t_5 = self.l_15(t_1)
        t_1 = t_1.shape
        t_1 = t_1[slice(None, 2, None)]
        t_1 = t_1[0]
        t_2 = t_2.view(t_1, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_4 = t_4.view(t_1, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_5 = t_5.view(t_1, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_2, t_4)
        t_4 += t_0
        t_2 = t_4.float()
        t_2 = torch.nn.functional.softmax(t_2, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_2.type_as(t_4)
        t_4 = self.l_16(t_4)
        t_5 = torch.matmul(t_4, t_5)
        t_5 = t_5.transpose(1, 2)
        t_5 = t_5.contiguous()
        t_1 = t_5.view(t_1, -1, 4096)
        t_1 = self.l_17(t_1)
        t_5 = self.l_18(t_1)
        t_5 = t_3 + t_5
        t_0 = (t_1, None, t_0)
        t_5 = (t_5,)
        t_0 = t_0[slice(1, None, None)]
        t_0 = t_5 + t_0
        t_5 = t_0[slice(None, 2, None)]
        t_1 = t_5[0]
        t_3 = self.l_19(t_1)
        t_5 = t_5[1]
        t_0 = t_0[slice(2, None, None)]
        t_3 = self.l_20(t_3)
        t_3 = torch.nn.functional.relu(t_3, inplace=False)
        t_3 = self.l_21(t_3)
        t_3 = self.l_22(t_3)
        t_3 = self.l_23(t_3)
        t_3 = t_1 + t_3
        t_5 = (t_3, t_5)
        t_0 = t_5 + t_0
        t_5 = t_0[slice(None, 2, None)]
        t_5 = t_5[0]
        t_3 = self.l_24(t_5)
        t_0 = t_0[2]
        t_1 = self.l_25(t_3)
        t_4 = self.l_26(t_3)
        t_2 = self.l_27(t_3)
        t_3 = t_3.shape
        t_3 = t_3[slice(None, 2, None)]
        t_3 = t_3[0]
        t_1 = t_1.view(t_3, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_4 = t_4.view(t_3, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_2 = t_2.view(t_3, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_1, t_4)
        t_4 += t_0
        t_1 = t_4.float()
        t_1 = torch.nn.functional.softmax(t_1, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_1.type_as(t_4)
        t_4 = self.l_28(t_4)
        t_2 = torch.matmul(t_4, t_2)
        t_2 = t_2.transpose(1, 2)
        t_2 = t_2.contiguous()
        t_3 = t_2.view(t_3, -1, 4096)
        t_3 = self.l_29(t_3)
        t_2 = self.l_30(t_3)
        t_2 = t_5 + t_2
        t_0 = (t_3, None, t_0)
        t_2 = (t_2,)
        t_0 = t_0[slice(1, None, None)]
        t_0 = t_2 + t_0
        t_2 = t_0[slice(None, 2, None)]
        t_3 = t_2[0]
        t_5 = self.l_31(t_3)
        t_2 = t_2[1]
        t_0 = t_0[slice(2, None, None)]
        t_5 = self.l_32(t_5)
        t_5 = torch.nn.functional.relu(t_5, inplace=False)
        t_5 = self.l_33(t_5)
        t_5 = self.l_34(t_5)
        t_5 = self.l_35(t_5)
        t_5 = t_3 + t_5
        t_2 = (t_5, t_2)
        t_0 = t_2 + t_0
        t_2 = t_0[slice(None, 2, None)]
        t_2 = t_2[0]
        t_0 = t_0[2]
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1578
        # T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1580
        return list(flatten((t_2, t_0)))

    def state_dict(self, *args, **kwargs):
        # we return the state dict of this part as it should be in the original model
        return state_dict(self, *args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return load_state_dict(self, *args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        # we return the named parameters of this part as it should be in the original model
        return named_parameters(self, *args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        # we return the named buffers of this part as it should be in the original model
        return named_buffers(self, *args, **kwargs)

    def cpu(self):
        return cpu(self)

    def cuda(self, device=None):
        return cuda(self, device=device)

    def to(self, *args, **kwargs):
        return to(self, *args, **kwargs)


class Partition6(nn.Module):
    LAYER_SCOPES = [
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout]',
        ]
    TENSORS = [
        ]
    def __init__(self, layers, tensors, device='cuda:6'):
        super().__init__()

        # Initialize partition layers
        for idx, layer_scope in enumerate(self.LAYER_SCOPES):
            self.add_module(f'l_{idx}' ,layers[layer_scope])

        # Initialize partition tensors (params and buffs)
        b = p = 0
        for tensor_scope in self.TENSORS:
            tensor = tensors[tensor_scope]
            if isinstance(tensor, nn.Parameter):
                self.register_parameter(f'p_{p}', tensor)
                p += 1
            else:
                self.register_buffer(f'b_{b}', tensor)
                b += 1

        self.device = torch.device(device)
        self.input_structure = [1, 1]
        self.lookup = {'l_0': 'encoder.block.18.layer.0.layer_norm',
                        'l_1': 'encoder.block.18.layer.0.SelfAttention.q',
                        'l_2': 'encoder.block.18.layer.0.SelfAttention.k',
                        'l_3': 'encoder.block.18.layer.0.SelfAttention.v',
                        'l_4': 'encoder.block.18.layer.0.SelfAttention.dropout',
                        'l_5': 'encoder.block.18.layer.0.SelfAttention.o',
                        'l_6': 'encoder.block.18.layer.0.dropout',
                        'l_7': 'encoder.block.18.layer.1.layer_norm',
                        'l_8': 'encoder.block.18.layer.1.DenseReluDense.wi',
                        'l_9': 'encoder.block.18.layer.1.DenseReluDense.dropout',
                        'l_10': 'encoder.block.18.layer.1.DenseReluDense.wo',
                        'l_11': 'encoder.block.18.layer.1.dropout',
                        'l_12': 'encoder.block.19.layer.0.layer_norm',
                        'l_13': 'encoder.block.19.layer.0.SelfAttention.q',
                        'l_14': 'encoder.block.19.layer.0.SelfAttention.k',
                        'l_15': 'encoder.block.19.layer.0.SelfAttention.v',
                        'l_16': 'encoder.block.19.layer.0.SelfAttention.dropout',
                        'l_17': 'encoder.block.19.layer.0.SelfAttention.o',
                        'l_18': 'encoder.block.19.layer.0.dropout',
                        'l_19': 'encoder.block.19.layer.1.layer_norm',
                        'l_20': 'encoder.block.19.layer.1.DenseReluDense.wi',
                        'l_21': 'encoder.block.19.layer.1.DenseReluDense.dropout',
                        'l_22': 'encoder.block.19.layer.1.DenseReluDense.wo',
                        'l_23': 'encoder.block.19.layer.1.dropout',
                        'l_24': 'encoder.block.20.layer.0.layer_norm',
                        'l_25': 'encoder.block.20.layer.0.SelfAttention.q',
                        'l_26': 'encoder.block.20.layer.0.SelfAttention.k',
                        'l_27': 'encoder.block.20.layer.0.SelfAttention.v',
                        'l_28': 'encoder.block.20.layer.0.SelfAttention.dropout',
                        'l_29': 'encoder.block.20.layer.0.SelfAttention.o',
                        'l_30': 'encoder.block.20.layer.0.dropout',
                        'l_31': 'encoder.block.20.layer.1.layer_norm',
                        'l_32': 'encoder.block.20.layer.1.DenseReluDense.wi',
                        'l_33': 'encoder.block.20.layer.1.DenseReluDense.dropout',
                        'l_34': 'encoder.block.20.layer.1.DenseReluDense.wo',
                        'l_35': 'encoder.block.20.layer.1.dropout'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1578 <=> x0
        # T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1580 <=> x1
        x0, x1 = unflatten(args, self.input_structure)
        t_0 = self.l_0(x0)
        t_1 = self.l_1(t_0)
        t_2 = self.l_2(t_0)
        t_3 = self.l_3(t_0)
        t_0 = t_0.shape
        t_0 = t_0[slice(None, 2, None)]
        t_0 = t_0[0]
        t_1 = t_1.view(t_0, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_2 = t_2.view(t_0, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_3 = t_3.view(t_0, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_2 = t_2.transpose(3, 2)
        t_2 = torch.matmul(t_1, t_2)
        t_2 += x1
        t_1 = t_2.float()
        t_1 = torch.nn.functional.softmax(t_1, dim=-1, _stacklevel=3, dtype=None)
        t_2 = t_1.type_as(t_2)
        t_2 = self.l_4(t_2)
        t_3 = torch.matmul(t_2, t_3)
        t_3 = t_3.transpose(1, 2)
        t_3 = t_3.contiguous()
        t_0 = t_3.view(t_0, -1, 4096)
        t_0 = self.l_5(t_0)
        t_3 = self.l_6(t_0)
        t_3 = x0 + t_3
        t_0 = (t_0, None, x1)
        t_3 = (t_3,)
        t_0 = t_0[slice(1, None, None)]
        t_0 = t_3 + t_0
        t_3 = t_0[slice(None, 2, None)]
        t_2 = t_3[0]
        t_1 = self.l_7(t_2)
        t_3 = t_3[1]
        t_0 = t_0[slice(2, None, None)]
        t_1 = self.l_8(t_1)
        t_1 = torch.nn.functional.relu(t_1, inplace=False)
        t_1 = self.l_9(t_1)
        t_1 = self.l_10(t_1)
        t_1 = self.l_11(t_1)
        t_1 = t_2 + t_1
        t_3 = (t_1, t_3)
        t_0 = t_3 + t_0
        t_3 = t_0[slice(None, 2, None)]
        t_3 = t_3[0]
        t_1 = self.l_12(t_3)
        t_0 = t_0[2]
        t_2 = self.l_13(t_1)
        t_4 = self.l_14(t_1)
        t_5 = self.l_15(t_1)
        t_1 = t_1.shape
        t_1 = t_1[slice(None, 2, None)]
        t_1 = t_1[0]
        t_2 = t_2.view(t_1, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_4 = t_4.view(t_1, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_5 = t_5.view(t_1, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_2, t_4)
        t_4 += t_0
        t_2 = t_4.float()
        t_2 = torch.nn.functional.softmax(t_2, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_2.type_as(t_4)
        t_4 = self.l_16(t_4)
        t_5 = torch.matmul(t_4, t_5)
        t_5 = t_5.transpose(1, 2)
        t_5 = t_5.contiguous()
        t_1 = t_5.view(t_1, -1, 4096)
        t_1 = self.l_17(t_1)
        t_5 = self.l_18(t_1)
        t_5 = t_3 + t_5
        t_0 = (t_1, None, t_0)
        t_5 = (t_5,)
        t_0 = t_0[slice(1, None, None)]
        t_0 = t_5 + t_0
        t_5 = t_0[slice(None, 2, None)]
        t_1 = t_5[0]
        t_3 = self.l_19(t_1)
        t_5 = t_5[1]
        t_0 = t_0[slice(2, None, None)]
        t_3 = self.l_20(t_3)
        t_3 = torch.nn.functional.relu(t_3, inplace=False)
        t_3 = self.l_21(t_3)
        t_3 = self.l_22(t_3)
        t_3 = self.l_23(t_3)
        t_3 = t_1 + t_3
        t_5 = (t_3, t_5)
        t_0 = t_5 + t_0
        t_5 = t_0[slice(None, 2, None)]
        t_5 = t_5[0]
        t_3 = self.l_24(t_5)
        t_0 = t_0[2]
        t_1 = self.l_25(t_3)
        t_4 = self.l_26(t_3)
        t_2 = self.l_27(t_3)
        t_3 = t_3.shape
        t_3 = t_3[slice(None, 2, None)]
        t_3 = t_3[0]
        t_1 = t_1.view(t_3, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_4 = t_4.view(t_3, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_2 = t_2.view(t_3, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_1, t_4)
        t_4 += t_0
        t_1 = t_4.float()
        t_1 = torch.nn.functional.softmax(t_1, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_1.type_as(t_4)
        t_4 = self.l_28(t_4)
        t_2 = torch.matmul(t_4, t_2)
        t_2 = t_2.transpose(1, 2)
        t_2 = t_2.contiguous()
        t_3 = t_2.view(t_3, -1, 4096)
        t_3 = self.l_29(t_3)
        t_2 = self.l_30(t_3)
        t_2 = t_5 + t_2
        t_0 = (t_3, None, t_0)
        t_2 = (t_2,)
        t_0 = t_0[slice(1, None, None)]
        t_0 = t_2 + t_0
        t_2 = t_0[slice(None, 2, None)]
        t_3 = t_2[0]
        t_5 = self.l_31(t_3)
        t_2 = t_2[1]
        t_0 = t_0[slice(2, None, None)]
        t_5 = self.l_32(t_5)
        t_5 = torch.nn.functional.relu(t_5, inplace=False)
        t_5 = self.l_33(t_5)
        t_5 = self.l_34(t_5)
        t_5 = self.l_35(t_5)
        t_5 = t_3 + t_5
        t_2 = (t_5, t_2)
        t_0 = t_2 + t_0
        t_2 = t_0[slice(None, 2, None)]
        t_2 = t_2[0]
        t_0 = t_0[2]
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1830
        # T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1832
        return list(flatten((t_2, t_0)))

    def state_dict(self, *args, **kwargs):
        # we return the state dict of this part as it should be in the original model
        return state_dict(self, *args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return load_state_dict(self, *args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        # we return the named parameters of this part as it should be in the original model
        return named_parameters(self, *args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        # we return the named buffers of this part as it should be in the original model
        return named_buffers(self, *args, **kwargs)

    def cpu(self):
        return cpu(self)

    def cuda(self, device=None):
        return cuda(self, device=device)

    def to(self, *args, **kwargs):
        return to(self, *args, **kwargs)


class Partition7(nn.Module):
    LAYER_SCOPES = [
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5LayerNorm[final_layer_norm]',
        ]
    TENSORS = [
        ]
    def __init__(self, layers, tensors, device='cuda:7'):
        super().__init__()

        # Initialize partition layers
        for idx, layer_scope in enumerate(self.LAYER_SCOPES):
            self.add_module(f'l_{idx}' ,layers[layer_scope])

        # Initialize partition tensors (params and buffs)
        b = p = 0
        for tensor_scope in self.TENSORS:
            tensor = tensors[tensor_scope]
            if isinstance(tensor, nn.Parameter):
                self.register_parameter(f'p_{p}', tensor)
                p += 1
            else:
                self.register_buffer(f'b_{b}', tensor)
                b += 1

        self.device = torch.device(device)
        self.input_structure = [1, 1]
        self.lookup = {'l_0': 'encoder.block.21.layer.0.layer_norm',
                        'l_1': 'encoder.block.21.layer.0.SelfAttention.q',
                        'l_2': 'encoder.block.21.layer.0.SelfAttention.k',
                        'l_3': 'encoder.block.21.layer.0.SelfAttention.v',
                        'l_4': 'encoder.block.21.layer.0.SelfAttention.dropout',
                        'l_5': 'encoder.block.21.layer.0.SelfAttention.o',
                        'l_6': 'encoder.block.21.layer.0.dropout',
                        'l_7': 'encoder.block.21.layer.1.layer_norm',
                        'l_8': 'encoder.block.21.layer.1.DenseReluDense.wi',
                        'l_9': 'encoder.block.21.layer.1.DenseReluDense.dropout',
                        'l_10': 'encoder.block.21.layer.1.DenseReluDense.wo',
                        'l_11': 'encoder.block.21.layer.1.dropout',
                        'l_12': 'encoder.block.22.layer.0.layer_norm',
                        'l_13': 'encoder.block.22.layer.0.SelfAttention.q',
                        'l_14': 'encoder.block.22.layer.0.SelfAttention.k',
                        'l_15': 'encoder.block.22.layer.0.SelfAttention.v',
                        'l_16': 'encoder.block.22.layer.0.SelfAttention.dropout',
                        'l_17': 'encoder.block.22.layer.0.SelfAttention.o',
                        'l_18': 'encoder.block.22.layer.0.dropout',
                        'l_19': 'encoder.block.22.layer.1.layer_norm',
                        'l_20': 'encoder.block.22.layer.1.DenseReluDense.wi',
                        'l_21': 'encoder.block.22.layer.1.DenseReluDense.dropout',
                        'l_22': 'encoder.block.22.layer.1.DenseReluDense.wo',
                        'l_23': 'encoder.block.22.layer.1.dropout',
                        'l_24': 'encoder.block.23.layer.0.layer_norm',
                        'l_25': 'encoder.block.23.layer.0.SelfAttention.q',
                        'l_26': 'encoder.block.23.layer.0.SelfAttention.k',
                        'l_27': 'encoder.block.23.layer.0.SelfAttention.v',
                        'l_28': 'encoder.block.23.layer.0.SelfAttention.dropout',
                        'l_29': 'encoder.block.23.layer.0.SelfAttention.o',
                        'l_30': 'encoder.block.23.layer.0.dropout',
                        'l_31': 'encoder.block.23.layer.1.layer_norm',
                        'l_32': 'encoder.block.23.layer.1.DenseReluDense.wi',
                        'l_33': 'encoder.block.23.layer.1.DenseReluDense.dropout',
                        'l_34': 'encoder.block.23.layer.1.DenseReluDense.wo',
                        'l_35': 'encoder.block.23.layer.1.dropout',
                        'l_36': 'encoder.final_layer_norm'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[encoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerFF[1]/Dropout[dropout] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[encoder]/T5LayerNorm[final_layer_norm] <=> self.l_36
        # T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1830 <=> x0
        # T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___1832 <=> x1
        x0, x1 = unflatten(args, self.input_structure)
        t_0 = self.l_0(x0)
        t_1 = self.l_1(t_0)
        t_2 = self.l_2(t_0)
        t_3 = self.l_3(t_0)
        t_0 = t_0.shape
        t_0 = t_0[slice(None, 2, None)]
        t_0 = t_0[0]
        t_1 = t_1.view(t_0, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_2 = t_2.view(t_0, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_3 = t_3.view(t_0, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_2 = t_2.transpose(3, 2)
        t_2 = torch.matmul(t_1, t_2)
        t_2 += x1
        t_1 = t_2.float()
        t_1 = torch.nn.functional.softmax(t_1, dim=-1, _stacklevel=3, dtype=None)
        t_2 = t_1.type_as(t_2)
        t_2 = self.l_4(t_2)
        t_3 = torch.matmul(t_2, t_3)
        t_3 = t_3.transpose(1, 2)
        t_3 = t_3.contiguous()
        t_0 = t_3.view(t_0, -1, 4096)
        t_0 = self.l_5(t_0)
        t_3 = self.l_6(t_0)
        t_3 = x0 + t_3
        t_0 = (t_0, None, x1)
        t_3 = (t_3,)
        t_0 = t_0[slice(1, None, None)]
        t_0 = t_3 + t_0
        t_3 = t_0[slice(None, 2, None)]
        t_2 = t_3[0]
        t_1 = self.l_7(t_2)
        t_3 = t_3[1]
        t_0 = t_0[slice(2, None, None)]
        t_1 = self.l_8(t_1)
        t_1 = torch.nn.functional.relu(t_1, inplace=False)
        t_1 = self.l_9(t_1)
        t_1 = self.l_10(t_1)
        t_1 = self.l_11(t_1)
        t_1 = t_2 + t_1
        t_3 = (t_1, t_3)
        t_0 = t_3 + t_0
        t_3 = t_0[slice(None, 2, None)]
        t_3 = t_3[0]
        t_1 = self.l_12(t_3)
        t_0 = t_0[2]
        t_2 = self.l_13(t_1)
        t_4 = self.l_14(t_1)
        t_5 = self.l_15(t_1)
        t_1 = t_1.shape
        t_1 = t_1[slice(None, 2, None)]
        t_1 = t_1[0]
        t_2 = t_2.view(t_1, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_4 = t_4.view(t_1, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_5 = t_5.view(t_1, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_2, t_4)
        t_4 += t_0
        t_2 = t_4.float()
        t_2 = torch.nn.functional.softmax(t_2, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_2.type_as(t_4)
        t_4 = self.l_16(t_4)
        t_5 = torch.matmul(t_4, t_5)
        t_5 = t_5.transpose(1, 2)
        t_5 = t_5.contiguous()
        t_1 = t_5.view(t_1, -1, 4096)
        t_1 = self.l_17(t_1)
        t_5 = self.l_18(t_1)
        t_5 = t_3 + t_5
        t_0 = (t_1, None, t_0)
        t_5 = (t_5,)
        t_0 = t_0[slice(1, None, None)]
        t_0 = t_5 + t_0
        t_5 = t_0[slice(None, 2, None)]
        t_1 = t_5[0]
        t_3 = self.l_19(t_1)
        t_5 = t_5[1]
        t_0 = t_0[slice(2, None, None)]
        t_3 = self.l_20(t_3)
        t_3 = torch.nn.functional.relu(t_3, inplace=False)
        t_3 = self.l_21(t_3)
        t_3 = self.l_22(t_3)
        t_3 = self.l_23(t_3)
        t_3 = t_1 + t_3
        t_5 = (t_3, t_5)
        t_0 = t_5 + t_0
        t_5 = t_0[slice(None, 2, None)]
        t_5 = t_5[0]
        t_3 = self.l_24(t_5)
        t_0 = t_0[2]
        t_1 = self.l_25(t_3)
        t_4 = self.l_26(t_3)
        t_2 = self.l_27(t_3)
        t_3 = t_3.shape
        t_3 = t_3[slice(None, 2, None)]
        t_3 = t_3[0]
        t_1 = t_1.view(t_3, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_4 = t_4.view(t_3, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_2 = t_2.view(t_3, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_1, t_4)
        t_4 += t_0
        t_1 = t_4.float()
        t_1 = torch.nn.functional.softmax(t_1, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_1.type_as(t_4)
        t_4 = self.l_28(t_4)
        t_2 = torch.matmul(t_4, t_2)
        t_2 = t_2.transpose(1, 2)
        t_2 = t_2.contiguous()
        t_3 = t_2.view(t_3, -1, 4096)
        t_3 = self.l_29(t_3)
        t_2 = self.l_30(t_3)
        t_2 = t_5 + t_2
        t_0 = (t_3, None, t_0)
        t_2 = (t_2,)
        t_0 = t_0[slice(1, None, None)]
        t_0 = t_2 + t_0
        t_2 = t_0[slice(None, 2, None)]
        t_3 = t_2[0]
        t_5 = self.l_31(t_3)
        t_2 = t_2[1]
        t_0 = t_0[slice(2, None, None)]
        t_5 = self.l_32(t_5)
        t_5 = torch.nn.functional.relu(t_5, inplace=False)
        t_5 = self.l_33(t_5)
        t_5 = self.l_34(t_5)
        t_5 = self.l_35(t_5)
        t_5 = t_3 + t_5
        t_2 = (t_5, t_2)
        t_0 = t_2 + t_0
        t_0 = t_0[slice(None, 2, None)]
        t_0 = t_0[0]
        t_0 = self.l_36(t_0)
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/T5LayerNorm[final_layer_norm]
        return (t_0,)

    def state_dict(self, *args, **kwargs):
        # we return the state dict of this part as it should be in the original model
        return state_dict(self, *args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return load_state_dict(self, *args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        # we return the named parameters of this part as it should be in the original model
        return named_parameters(self, *args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        # we return the named buffers of this part as it should be in the original model
        return named_buffers(self, *args, **kwargs)

    def cpu(self):
        return cpu(self)

    def cuda(self, device=None):
        return cuda(self, device=device)

    def to(self, *args, **kwargs):
        return to(self, *args, **kwargs)


class Partition8(nn.Module):
    LAYER_SCOPES = [
            'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/StatelessEmbedding[embed_tokens]',
            'T5ForConditionalGeneration/T5Stack[decoder]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Embedding[relative_attention_bias]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout]',
        ]
    TENSORS = [
        ]
    def __init__(self, layers, tensors, device='cuda:8'):
        super().__init__()

        # Initialize partition layers
        for idx, layer_scope in enumerate(self.LAYER_SCOPES):
            self.add_module(f'l_{idx}' ,layers[layer_scope])

        # Initialize partition tensors (params and buffs)
        b = p = 0
        for tensor_scope in self.TENSORS:
            tensor = tensors[tensor_scope]
            if isinstance(tensor, nn.Parameter):
                self.register_parameter(f'p_{p}', tensor)
                p += 1
            else:
                self.register_buffer(f'b_{b}', tensor)
                b += 1

        self.device = torch.device(device)
        self.input_structure = [1, 1, 1, 1, 1, 1]
        self.lookup = {'l_0': 'encoder.dropout',
                        'l_1': 'decoder.embed_tokens',
                        'l_2': 'decoder.dropout',
                        'l_3': 'decoder.block.0.layer.0.layer_norm',
                        'l_4': 'decoder.block.0.layer.0.SelfAttention.q',
                        'l_5': 'decoder.block.0.layer.0.SelfAttention.k',
                        'l_6': 'decoder.block.0.layer.0.SelfAttention.v',
                        'l_7': 'decoder.block.0.layer.0.SelfAttention.relative_attention_bias',
                        'l_8': 'decoder.block.0.layer.0.SelfAttention.dropout',
                        'l_9': 'decoder.block.0.layer.0.SelfAttention.o',
                        'l_10': 'decoder.block.0.layer.0.dropout',
                        'l_11': 'decoder.block.0.layer.1.layer_norm',
                        'l_12': 'decoder.block.0.layer.1.EncDecAttention.q',
                        'l_13': 'decoder.block.0.layer.1.EncDecAttention.k',
                        'l_14': 'decoder.block.0.layer.1.EncDecAttention.v',
                        'l_15': 'decoder.block.0.layer.1.EncDecAttention.dropout',
                        'l_16': 'decoder.block.0.layer.1.EncDecAttention.o',
                        'l_17': 'decoder.block.0.layer.1.dropout',
                        'l_18': 'decoder.block.0.layer.2.layer_norm',
                        'l_19': 'decoder.block.0.layer.2.DenseReluDense.wi',
                        'l_20': 'decoder.block.0.layer.2.DenseReluDense.dropout',
                        'l_21': 'decoder.block.0.layer.2.DenseReluDense.wo',
                        'l_22': 'decoder.block.0.layer.2.dropout',
                        'l_23': 'decoder.block.1.layer.0.layer_norm',
                        'l_24': 'decoder.block.1.layer.0.SelfAttention.q',
                        'l_25': 'decoder.block.1.layer.0.SelfAttention.k',
                        'l_26': 'decoder.block.1.layer.0.SelfAttention.v',
                        'l_27': 'decoder.block.1.layer.0.SelfAttention.dropout',
                        'l_28': 'decoder.block.1.layer.0.SelfAttention.o',
                        'l_29': 'decoder.block.1.layer.0.dropout',
                        'l_30': 'decoder.block.1.layer.1.layer_norm',
                        'l_31': 'decoder.block.1.layer.1.EncDecAttention.q',
                        'l_32': 'decoder.block.1.layer.1.EncDecAttention.k',
                        'l_33': 'decoder.block.1.layer.1.EncDecAttention.v',
                        'l_34': 'decoder.block.1.layer.1.EncDecAttention.dropout',
                        'l_35': 'decoder.block.1.layer.1.EncDecAttention.o',
                        'l_36': 'decoder.block.1.layer.1.dropout',
                        'l_37': 'decoder.block.1.layer.2.layer_norm',
                        'l_38': 'decoder.block.1.layer.2.DenseReluDense.wi',
                        'l_39': 'decoder.block.1.layer.2.DenseReluDense.dropout',
                        'l_40': 'decoder.block.1.layer.2.DenseReluDense.wo',
                        'l_41': 'decoder.block.1.layer.2.dropout',
                        'l_42': 'decoder.block.2.layer.0.layer_norm',
                        'l_43': 'decoder.block.2.layer.0.SelfAttention.q',
                        'l_44': 'decoder.block.2.layer.0.SelfAttention.k',
                        'l_45': 'decoder.block.2.layer.0.SelfAttention.v',
                        'l_46': 'decoder.block.2.layer.0.SelfAttention.dropout',
                        'l_47': 'decoder.block.2.layer.0.SelfAttention.o',
                        'l_48': 'decoder.block.2.layer.0.dropout',
                        'l_49': 'decoder.block.2.layer.1.layer_norm',
                        'l_50': 'decoder.block.2.layer.1.EncDecAttention.q',
                        'l_51': 'decoder.block.2.layer.1.EncDecAttention.k',
                        'l_52': 'decoder.block.2.layer.1.EncDecAttention.v',
                        'l_53': 'decoder.block.2.layer.1.EncDecAttention.dropout',
                        'l_54': 'decoder.block.2.layer.1.EncDecAttention.o',
                        'l_55': 'decoder.block.2.layer.1.dropout',
                        'l_56': 'decoder.block.2.layer.2.layer_norm',
                        'l_57': 'decoder.block.2.layer.2.DenseReluDense.wi',
                        'l_58': 'decoder.block.2.layer.2.DenseReluDense.dropout',
                        'l_59': 'decoder.block.2.layer.2.DenseReluDense.wo',
                        'l_60': 'decoder.block.2.layer.2.dropout'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[decoder]/StatelessEmbedding[embed_tokens] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[decoder]/Dropout[dropout] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Embedding[relative_attention_bias] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[0]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_36
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_37
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_38
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_39
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_40
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[1]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout] <=> self.l_41
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_42
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_43
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_44
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_45
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_46
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_47
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_48
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_49
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_50
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_51
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_52
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_53
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_54
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_55
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_56
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_57
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_58
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_59
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[2]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout] <=> self.l_60
        # input0 <=> attention_mask
        # input2 <=> decoder_input_ids
        # T5ForConditionalGeneration/Parameter[shared_embed_weight] <=> x0
        # T5ForConditionalGeneration/T5Stack[encoder]/T5LayerNorm[final_layer_norm] <=> x1
        # T5ForConditionalGeneration/T5Stack[decoder]/Size::__getitem___2087 <=> x2
        # T5ForConditionalGeneration/T5Stack[decoder]/Tensor::__mul___2117 <=> x3
        attention_mask, decoder_input_ids, x0, x1, x2, x3 = unflatten(args, self.input_structure)
        t_0 = self.l_0(x1)
        t_1 = self.l_13(t_0)
        t_2 = self.l_14(t_0)
        t_3 = self.l_32(t_0)
        t_4 = self.l_33(t_0)
        t_5 = self.l_51(t_0)
        t_6 = self.l_52(t_0)
        t_7 = decoder_input_ids.view(-1, x2)
        t_7 = self.l_1(x0, t_7)
        t_7 = self.l_2(t_7)
        t_8 = attention_mask[(slice(None, None, None), None, None, slice(None, None, None))]
        t_8 = t_8.to(dtype=torch.float32)
        t_8 = 1.0 - t_8
        t_8 = t_8 * -1000000000.0
        t_9 = self.l_3(t_7)
        t_10 = self.l_4(t_9)
        t_11 = self.l_5(t_9)
        t_12 = self.l_6(t_9)
        t_9 = t_9.shape
        t_9 = t_9[slice(None, 2, None)]
        t_13 = t_9[0]
        t_9 = t_9[1]
        t_10 = t_10.view(t_13, -1, 32, 128)
        t_10 = t_10.transpose(1, 2)
        t_11 = t_11.view(t_13, -1, 32, 128)
        t_11 = t_11.transpose(1, 2)
        t_12 = t_12.view(t_13, -1, 32, 128)
        t_12 = t_12.transpose(1, 2)
        t_11 = t_11.transpose(3, 2)
        t_11 = torch.matmul(t_10, t_11)
        t_10 = torch.arange(t_9, dtype=torch.int64, device=self.device)
        t_10 = t_10[(slice(None, None, None), None)]
        t_9 = torch.arange(t_9, dtype=torch.int64, device=self.device)
        t_9 = t_9[(None, slice(None, None, None))]
        t_10 = t_9 - t_10
        t_9 = torch.zeros_like(t_10, device=self.device)
        t_9 = torch.min(t_10, t_9)
        t_9 = -t_9
        t_10 = t_9.float()
        t_14 = t_9 < 16
        t_10 = t_10 / 16
        t_10 = torch.log(t_10)
        t_15 = math.log(8.0)
        t_15 = t_10 / t_15
        t_15 = t_15 * 16
        t_15 = t_15.to(torch.int64)
        t_15 = 16 + t_15
        t_10 = torch.full_like(t_15, 31, device=self.device)
        t_10 = torch.min(t_15, t_10)
        t_10 = torch.where(t_14, t_9, t_10)
        t_10 = 0 + t_10
        t_10 = t_10.to(self.device)
        t_10 = self.l_7(t_10)
        t_10 = t_10.permute([2, 0, 1])
        t_10 = t_10.unsqueeze(0)
        t_10 = t_10 + x3
        t_11 += t_10
        t_9 = t_11.float()
        t_9 = torch.nn.functional.softmax(t_9, dim=-1, _stacklevel=3, dtype=None)
        t_11 = t_9.type_as(t_11)
        t_11 = self.l_8(t_11)
        t_12 = torch.matmul(t_11, t_12)
        t_12 = t_12.transpose(1, 2)
        t_12 = t_12.contiguous()
        t_13 = t_12.view(t_13, -1, 4096)
        t_13 = self.l_9(t_13)
        t_12 = self.l_10(t_13)
        t_12 = t_7 + t_12
        t_10 = (t_13, None, t_10)
        t_12 = (t_12,)
        t_10 = t_10[slice(1, None, None)]
        t_10 = t_12 + t_10
        t_12 = t_10[slice(None, 2, None)]
        t_13 = t_12[0]
        t_7 = self.l_11(t_13)
        t_12 = t_12[1]
        t_10 = t_10[slice(2, None, None)]
        t_11 = self.l_12(t_7)
        t_7 = t_7.shape
        t_7 = t_7[slice(None, 2, None)]
        t_9 = t_7[0]
        t_7 = t_7[1]
        t_14 = t_0.shape
        t_14 = t_14[1]
        t_11 = t_11.view(t_9, -1, 32, 128)
        t_11 = t_11.transpose(1, 2)
        t_1 = t_1.view(t_9, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_2 = t_2.view(t_9, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_1 = t_1.transpose(3, 2)
        t_1 = torch.matmul(t_11, t_1)
        t_11 = t_1.dtype
        t_14 = (1, 32, t_7, t_14)
        t_11 = torch.zeros(t_14, device=self.device, dtype=t_11)
        t_8 = t_11 + t_8
        t_1 += t_8
        t_11 = t_1.float()
        t_11 = torch.nn.functional.softmax(t_11, dim=-1, _stacklevel=3, dtype=None)
        t_1 = t_11.type_as(t_1)
        t_1 = self.l_15(t_1)
        t_2 = torch.matmul(t_1, t_2)
        t_2 = t_2.transpose(1, 2)
        t_2 = t_2.contiguous()
        t_9 = t_2.view(t_9, -1, 4096)
        t_9 = self.l_16(t_9)
        t_2 = self.l_17(t_9)
        t_2 = t_13 + t_2
        t_8 = (t_9, None, t_8)
        t_2 = (t_2,)
        t_8 = t_8[slice(1, None, None)]
        t_8 = t_2 + t_8
        t_2 = t_8[0]
        t_9 = self.l_18(t_2)
        t_8 = t_8[slice(2, None, None)]
        t_8 = t_10 + t_8
        t_9 = self.l_19(t_9)
        t_9 = torch.nn.functional.relu(t_9, inplace=False)
        t_9 = self.l_20(t_9)
        t_9 = self.l_21(t_9)
        t_9 = self.l_22(t_9)
        t_9 = t_2 + t_9
        t_12 = (t_9, t_12)
        t_8 = t_12 + t_8
        t_12 = t_8[slice(None, 2, None)]
        t_12 = t_12[0]
        t_9 = self.l_23(t_12)
        t_2 = t_8[2]
        t_8 = t_8[3]
        t_10 = self.l_24(t_9)
        t_13 = self.l_25(t_9)
        t_1 = self.l_26(t_9)
        t_9 = t_9.shape
        t_9 = t_9[slice(None, 2, None)]
        t_9 = t_9[0]
        t_10 = t_10.view(t_9, -1, 32, 128)
        t_10 = t_10.transpose(1, 2)
        t_13 = t_13.view(t_9, -1, 32, 128)
        t_13 = t_13.transpose(1, 2)
        t_1 = t_1.view(t_9, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_13 = t_13.transpose(3, 2)
        t_13 = torch.matmul(t_10, t_13)
        t_13 += t_2
        t_10 = t_13.float()
        t_10 = torch.nn.functional.softmax(t_10, dim=-1, _stacklevel=3, dtype=None)
        t_13 = t_10.type_as(t_13)
        t_13 = self.l_27(t_13)
        t_1 = torch.matmul(t_13, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_9 = t_1.view(t_9, -1, 4096)
        t_9 = self.l_28(t_9)
        t_1 = self.l_29(t_9)
        t_1 = t_12 + t_1
        t_2 = (t_9, None, t_2)
        t_1 = (t_1,)
        t_2 = t_2[slice(1, None, None)]
        t_2 = t_1 + t_2
        t_1 = t_2[slice(None, 2, None)]
        t_9 = t_1[0]
        t_12 = self.l_30(t_9)
        t_1 = t_1[1]
        t_2 = t_2[slice(2, None, None)]
        t_13 = self.l_31(t_12)
        t_12 = t_12.shape
        t_12 = t_12[slice(None, 2, None)]
        t_12 = t_12[0]
        t_13 = t_13.view(t_12, -1, 32, 128)
        t_13 = t_13.transpose(1, 2)
        t_3 = t_3.view(t_12, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_4 = t_4.view(t_12, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_3 = t_3.transpose(3, 2)
        t_3 = torch.matmul(t_13, t_3)
        t_3 += t_8
        t_13 = t_3.float()
        t_13 = torch.nn.functional.softmax(t_13, dim=-1, _stacklevel=3, dtype=None)
        t_3 = t_13.type_as(t_3)
        t_3 = self.l_34(t_3)
        t_4 = torch.matmul(t_3, t_4)
        t_4 = t_4.transpose(1, 2)
        t_4 = t_4.contiguous()
        t_12 = t_4.view(t_12, -1, 4096)
        t_12 = self.l_35(t_12)
        t_4 = self.l_36(t_12)
        t_4 = t_9 + t_4
        t_8 = (t_12, None, t_8)
        t_4 = (t_4,)
        t_8 = t_8[slice(1, None, None)]
        t_8 = t_4 + t_8
        t_4 = t_8[0]
        t_12 = self.l_37(t_4)
        t_8 = t_8[slice(2, None, None)]
        t_8 = t_2 + t_8
        t_12 = self.l_38(t_12)
        t_12 = torch.nn.functional.relu(t_12, inplace=False)
        t_12 = self.l_39(t_12)
        t_12 = self.l_40(t_12)
        t_12 = self.l_41(t_12)
        t_12 = t_4 + t_12
        t_1 = (t_12, t_1)
        t_8 = t_1 + t_8
        t_1 = t_8[slice(None, 2, None)]
        t_1 = t_1[0]
        t_12 = self.l_42(t_1)
        t_4 = t_8[2]
        t_8 = t_8[3]
        t_2 = self.l_43(t_12)
        t_9 = self.l_44(t_12)
        t_3 = self.l_45(t_12)
        t_12 = t_12.shape
        t_12 = t_12[slice(None, 2, None)]
        t_12 = t_12[0]
        t_2 = t_2.view(t_12, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_9 = t_9.view(t_12, -1, 32, 128)
        t_9 = t_9.transpose(1, 2)
        t_3 = t_3.view(t_12, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_9 = t_9.transpose(3, 2)
        t_9 = torch.matmul(t_2, t_9)
        t_9 += t_4
        t_2 = t_9.float()
        t_2 = torch.nn.functional.softmax(t_2, dim=-1, _stacklevel=3, dtype=None)
        t_9 = t_2.type_as(t_9)
        t_9 = self.l_46(t_9)
        t_3 = torch.matmul(t_9, t_3)
        t_3 = t_3.transpose(1, 2)
        t_3 = t_3.contiguous()
        t_12 = t_3.view(t_12, -1, 4096)
        t_12 = self.l_47(t_12)
        t_3 = self.l_48(t_12)
        t_3 = t_1 + t_3
        t_4 = (t_12, None, t_4)
        t_3 = (t_3,)
        t_4 = t_4[slice(1, None, None)]
        t_4 = t_3 + t_4
        t_3 = t_4[slice(None, 2, None)]
        t_12 = t_3[0]
        t_1 = self.l_49(t_12)
        t_3 = t_3[1]
        t_4 = t_4[slice(2, None, None)]
        t_9 = self.l_50(t_1)
        t_1 = t_1.shape
        t_1 = t_1[slice(None, 2, None)]
        t_1 = t_1[0]
        t_9 = t_9.view(t_1, -1, 32, 128)
        t_9 = t_9.transpose(1, 2)
        t_5 = t_5.view(t_1, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_6 = t_6.view(t_1, -1, 32, 128)
        t_6 = t_6.transpose(1, 2)
        t_5 = t_5.transpose(3, 2)
        t_5 = torch.matmul(t_9, t_5)
        t_5 += t_8
        t_9 = t_5.float()
        t_9 = torch.nn.functional.softmax(t_9, dim=-1, _stacklevel=3, dtype=None)
        t_5 = t_9.type_as(t_5)
        t_5 = self.l_53(t_5)
        t_6 = torch.matmul(t_5, t_6)
        t_6 = t_6.transpose(1, 2)
        t_6 = t_6.contiguous()
        t_1 = t_6.view(t_1, -1, 4096)
        t_1 = self.l_54(t_1)
        t_6 = self.l_55(t_1)
        t_6 = t_12 + t_6
        t_8 = (t_1, None, t_8)
        t_6 = (t_6,)
        t_8 = t_8[slice(1, None, None)]
        t_8 = t_6 + t_8
        t_6 = t_8[0]
        t_1 = self.l_56(t_6)
        t_8 = t_8[slice(2, None, None)]
        t_8 = t_4 + t_8
        t_1 = self.l_57(t_1)
        t_1 = torch.nn.functional.relu(t_1, inplace=False)
        t_1 = self.l_58(t_1)
        t_1 = self.l_59(t_1)
        t_1 = self.l_60(t_1)
        t_1 = t_6 + t_1
        t_3 = (t_1, t_3)
        t_8 = t_3 + t_8
        t_3 = t_8[slice(None, 2, None)]
        t_3 = t_3[0]
        t_1 = t_8[2]
        t_8 = t_8[3]
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___2632
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___2634
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___2636
        return list(flatten((t_0, t_3, t_1, t_8)))

    def state_dict(self, *args, **kwargs):
        # we return the state dict of this part as it should be in the original model
        return state_dict(self, *args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return load_state_dict(self, *args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        # we return the named parameters of this part as it should be in the original model
        return named_parameters(self, *args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        # we return the named buffers of this part as it should be in the original model
        return named_buffers(self, *args, **kwargs)

    def cpu(self):
        return cpu(self)

    def cuda(self, device=None):
        return cuda(self, device=device)

    def to(self, *args, **kwargs):
        return to(self, *args, **kwargs)


class Partition9(nn.Module):
    LAYER_SCOPES = [
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout]',
        ]
    TENSORS = [
        ]
    def __init__(self, layers, tensors, device='cuda:9'):
        super().__init__()

        # Initialize partition layers
        for idx, layer_scope in enumerate(self.LAYER_SCOPES):
            self.add_module(f'l_{idx}' ,layers[layer_scope])

        # Initialize partition tensors (params and buffs)
        b = p = 0
        for tensor_scope in self.TENSORS:
            tensor = tensors[tensor_scope]
            if isinstance(tensor, nn.Parameter):
                self.register_parameter(f'p_{p}', tensor)
                p += 1
            else:
                self.register_buffer(f'b_{b}', tensor)
                b += 1

        self.device = torch.device(device)
        self.input_structure = [1, 1, 1, 1]
        self.lookup = {'l_0': 'decoder.block.3.layer.0.layer_norm',
                        'l_1': 'decoder.block.3.layer.0.SelfAttention.q',
                        'l_2': 'decoder.block.3.layer.0.SelfAttention.k',
                        'l_3': 'decoder.block.3.layer.0.SelfAttention.v',
                        'l_4': 'decoder.block.3.layer.0.SelfAttention.dropout',
                        'l_5': 'decoder.block.3.layer.0.SelfAttention.o',
                        'l_6': 'decoder.block.3.layer.0.dropout',
                        'l_7': 'decoder.block.3.layer.1.layer_norm',
                        'l_8': 'decoder.block.3.layer.1.EncDecAttention.q',
                        'l_9': 'decoder.block.3.layer.1.EncDecAttention.k',
                        'l_10': 'decoder.block.3.layer.1.EncDecAttention.v',
                        'l_11': 'decoder.block.3.layer.1.EncDecAttention.dropout',
                        'l_12': 'decoder.block.3.layer.1.EncDecAttention.o',
                        'l_13': 'decoder.block.3.layer.1.dropout',
                        'l_14': 'decoder.block.3.layer.2.layer_norm',
                        'l_15': 'decoder.block.3.layer.2.DenseReluDense.wi',
                        'l_16': 'decoder.block.3.layer.2.DenseReluDense.dropout',
                        'l_17': 'decoder.block.3.layer.2.DenseReluDense.wo',
                        'l_18': 'decoder.block.3.layer.2.dropout',
                        'l_19': 'decoder.block.4.layer.0.layer_norm',
                        'l_20': 'decoder.block.4.layer.0.SelfAttention.q',
                        'l_21': 'decoder.block.4.layer.0.SelfAttention.k',
                        'l_22': 'decoder.block.4.layer.0.SelfAttention.v',
                        'l_23': 'decoder.block.4.layer.0.SelfAttention.dropout',
                        'l_24': 'decoder.block.4.layer.0.SelfAttention.o',
                        'l_25': 'decoder.block.4.layer.0.dropout',
                        'l_26': 'decoder.block.4.layer.1.layer_norm',
                        'l_27': 'decoder.block.4.layer.1.EncDecAttention.q',
                        'l_28': 'decoder.block.4.layer.1.EncDecAttention.k',
                        'l_29': 'decoder.block.4.layer.1.EncDecAttention.v',
                        'l_30': 'decoder.block.4.layer.1.EncDecAttention.dropout',
                        'l_31': 'decoder.block.4.layer.1.EncDecAttention.o',
                        'l_32': 'decoder.block.4.layer.1.dropout',
                        'l_33': 'decoder.block.4.layer.2.layer_norm',
                        'l_34': 'decoder.block.4.layer.2.DenseReluDense.wi',
                        'l_35': 'decoder.block.4.layer.2.DenseReluDense.dropout',
                        'l_36': 'decoder.block.4.layer.2.DenseReluDense.wo',
                        'l_37': 'decoder.block.4.layer.2.dropout',
                        'l_38': 'decoder.block.5.layer.0.layer_norm',
                        'l_39': 'decoder.block.5.layer.0.SelfAttention.q',
                        'l_40': 'decoder.block.5.layer.0.SelfAttention.k',
                        'l_41': 'decoder.block.5.layer.0.SelfAttention.v',
                        'l_42': 'decoder.block.5.layer.0.SelfAttention.dropout',
                        'l_43': 'decoder.block.5.layer.0.SelfAttention.o',
                        'l_44': 'decoder.block.5.layer.0.dropout',
                        'l_45': 'decoder.block.5.layer.1.layer_norm',
                        'l_46': 'decoder.block.5.layer.1.EncDecAttention.q',
                        'l_47': 'decoder.block.5.layer.1.EncDecAttention.k',
                        'l_48': 'decoder.block.5.layer.1.EncDecAttention.v',
                        'l_49': 'decoder.block.5.layer.1.EncDecAttention.dropout',
                        'l_50': 'decoder.block.5.layer.1.EncDecAttention.o',
                        'l_51': 'decoder.block.5.layer.1.dropout',
                        'l_52': 'decoder.block.5.layer.2.layer_norm',
                        'l_53': 'decoder.block.5.layer.2.DenseReluDense.wi',
                        'l_54': 'decoder.block.5.layer.2.DenseReluDense.dropout',
                        'l_55': 'decoder.block.5.layer.2.DenseReluDense.wo',
                        'l_56': 'decoder.block.5.layer.2.dropout'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[3]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_36
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[4]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout] <=> self.l_37
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_38
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_39
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_40
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_41
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_42
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_43
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_44
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_45
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_46
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_47
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_48
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_49
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_50
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_51
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_52
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_53
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_54
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_55
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[5]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout] <=> self.l_56
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout] <=> x0
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___2632 <=> x1
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___2634 <=> x2
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___2636 <=> x3
        x0, x1, x2, x3 = unflatten(args, self.input_structure)
        t_0 = self.l_9(x0)
        t_1 = self.l_10(x0)
        t_2 = self.l_28(x0)
        t_3 = self.l_29(x0)
        t_4 = self.l_47(x0)
        t_5 = self.l_48(x0)
        t_6 = self.l_0(x1)
        t_7 = self.l_1(t_6)
        t_8 = self.l_2(t_6)
        t_9 = self.l_3(t_6)
        t_6 = t_6.shape
        t_6 = t_6[slice(None, 2, None)]
        t_6 = t_6[0]
        t_7 = t_7.view(t_6, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_8 = t_8.view(t_6, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_9 = t_9.view(t_6, -1, 32, 128)
        t_9 = t_9.transpose(1, 2)
        t_8 = t_8.transpose(3, 2)
        t_8 = torch.matmul(t_7, t_8)
        t_8 += x2
        t_7 = t_8.float()
        t_7 = torch.nn.functional.softmax(t_7, dim=-1, _stacklevel=3, dtype=None)
        t_8 = t_7.type_as(t_8)
        t_8 = self.l_4(t_8)
        t_9 = torch.matmul(t_8, t_9)
        t_9 = t_9.transpose(1, 2)
        t_9 = t_9.contiguous()
        t_6 = t_9.view(t_6, -1, 4096)
        t_6 = self.l_5(t_6)
        t_9 = self.l_6(t_6)
        t_9 = x1 + t_9
        t_6 = (t_6, None, x2)
        t_9 = (t_9,)
        t_6 = t_6[slice(1, None, None)]
        t_6 = t_9 + t_6
        t_9 = t_6[slice(None, 2, None)]
        t_8 = t_9[0]
        t_7 = self.l_7(t_8)
        t_9 = t_9[1]
        t_6 = t_6[slice(2, None, None)]
        t_10 = self.l_8(t_7)
        t_7 = t_7.shape
        t_7 = t_7[slice(None, 2, None)]
        t_7 = t_7[0]
        t_10 = t_10.view(t_7, -1, 32, 128)
        t_10 = t_10.transpose(1, 2)
        t_0 = t_0.view(t_7, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_1 = t_1.view(t_7, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_0 = t_0.transpose(3, 2)
        t_0 = torch.matmul(t_10, t_0)
        t_0 += x3
        t_10 = t_0.float()
        t_10 = torch.nn.functional.softmax(t_10, dim=-1, _stacklevel=3, dtype=None)
        t_0 = t_10.type_as(t_0)
        t_0 = self.l_11(t_0)
        t_1 = torch.matmul(t_0, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_7 = t_1.view(t_7, -1, 4096)
        t_7 = self.l_12(t_7)
        t_1 = self.l_13(t_7)
        t_1 = t_8 + t_1
        t_7 = (t_7, None, x3)
        t_1 = (t_1,)
        t_7 = t_7[slice(1, None, None)]
        t_7 = t_1 + t_7
        t_1 = t_7[0]
        t_8 = self.l_14(t_1)
        t_7 = t_7[slice(2, None, None)]
        t_7 = t_6 + t_7
        t_8 = self.l_15(t_8)
        t_8 = torch.nn.functional.relu(t_8, inplace=False)
        t_8 = self.l_16(t_8)
        t_8 = self.l_17(t_8)
        t_8 = self.l_18(t_8)
        t_8 = t_1 + t_8
        t_9 = (t_8, t_9)
        t_7 = t_9 + t_7
        t_9 = t_7[slice(None, 2, None)]
        t_9 = t_9[0]
        t_8 = self.l_19(t_9)
        t_1 = t_7[2]
        t_7 = t_7[3]
        t_6 = self.l_20(t_8)
        t_0 = self.l_21(t_8)
        t_10 = self.l_22(t_8)
        t_8 = t_8.shape
        t_8 = t_8[slice(None, 2, None)]
        t_8 = t_8[0]
        t_6 = t_6.view(t_8, -1, 32, 128)
        t_6 = t_6.transpose(1, 2)
        t_0 = t_0.view(t_8, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_10 = t_10.view(t_8, -1, 32, 128)
        t_10 = t_10.transpose(1, 2)
        t_0 = t_0.transpose(3, 2)
        t_0 = torch.matmul(t_6, t_0)
        t_0 += t_1
        t_6 = t_0.float()
        t_6 = torch.nn.functional.softmax(t_6, dim=-1, _stacklevel=3, dtype=None)
        t_0 = t_6.type_as(t_0)
        t_0 = self.l_23(t_0)
        t_10 = torch.matmul(t_0, t_10)
        t_10 = t_10.transpose(1, 2)
        t_10 = t_10.contiguous()
        t_8 = t_10.view(t_8, -1, 4096)
        t_8 = self.l_24(t_8)
        t_10 = self.l_25(t_8)
        t_10 = t_9 + t_10
        t_1 = (t_8, None, t_1)
        t_10 = (t_10,)
        t_1 = t_1[slice(1, None, None)]
        t_1 = t_10 + t_1
        t_10 = t_1[slice(None, 2, None)]
        t_8 = t_10[0]
        t_9 = self.l_26(t_8)
        t_10 = t_10[1]
        t_1 = t_1[slice(2, None, None)]
        t_0 = self.l_27(t_9)
        t_9 = t_9.shape
        t_9 = t_9[slice(None, 2, None)]
        t_9 = t_9[0]
        t_0 = t_0.view(t_9, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_2 = t_2.view(t_9, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_3 = t_3.view(t_9, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_2 = t_2.transpose(3, 2)
        t_2 = torch.matmul(t_0, t_2)
        t_2 += t_7
        t_0 = t_2.float()
        t_0 = torch.nn.functional.softmax(t_0, dim=-1, _stacklevel=3, dtype=None)
        t_2 = t_0.type_as(t_2)
        t_2 = self.l_30(t_2)
        t_3 = torch.matmul(t_2, t_3)
        t_3 = t_3.transpose(1, 2)
        t_3 = t_3.contiguous()
        t_9 = t_3.view(t_9, -1, 4096)
        t_9 = self.l_31(t_9)
        t_3 = self.l_32(t_9)
        t_3 = t_8 + t_3
        t_7 = (t_9, None, t_7)
        t_3 = (t_3,)
        t_7 = t_7[slice(1, None, None)]
        t_7 = t_3 + t_7
        t_3 = t_7[0]
        t_9 = self.l_33(t_3)
        t_7 = t_7[slice(2, None, None)]
        t_7 = t_1 + t_7
        t_9 = self.l_34(t_9)
        t_9 = torch.nn.functional.relu(t_9, inplace=False)
        t_9 = self.l_35(t_9)
        t_9 = self.l_36(t_9)
        t_9 = self.l_37(t_9)
        t_9 = t_3 + t_9
        t_10 = (t_9, t_10)
        t_7 = t_10 + t_7
        t_10 = t_7[slice(None, 2, None)]
        t_10 = t_10[0]
        t_9 = self.l_38(t_10)
        t_3 = t_7[2]
        t_7 = t_7[3]
        t_1 = self.l_39(t_9)
        t_8 = self.l_40(t_9)
        t_2 = self.l_41(t_9)
        t_9 = t_9.shape
        t_9 = t_9[slice(None, 2, None)]
        t_9 = t_9[0]
        t_1 = t_1.view(t_9, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_8 = t_8.view(t_9, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_2 = t_2.view(t_9, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_8 = t_8.transpose(3, 2)
        t_8 = torch.matmul(t_1, t_8)
        t_8 += t_3
        t_1 = t_8.float()
        t_1 = torch.nn.functional.softmax(t_1, dim=-1, _stacklevel=3, dtype=None)
        t_8 = t_1.type_as(t_8)
        t_8 = self.l_42(t_8)
        t_2 = torch.matmul(t_8, t_2)
        t_2 = t_2.transpose(1, 2)
        t_2 = t_2.contiguous()
        t_9 = t_2.view(t_9, -1, 4096)
        t_9 = self.l_43(t_9)
        t_2 = self.l_44(t_9)
        t_2 = t_10 + t_2
        t_3 = (t_9, None, t_3)
        t_2 = (t_2,)
        t_3 = t_3[slice(1, None, None)]
        t_3 = t_2 + t_3
        t_2 = t_3[slice(None, 2, None)]
        t_9 = t_2[0]
        t_10 = self.l_45(t_9)
        t_2 = t_2[1]
        t_3 = t_3[slice(2, None, None)]
        t_8 = self.l_46(t_10)
        t_10 = t_10.shape
        t_10 = t_10[slice(None, 2, None)]
        t_10 = t_10[0]
        t_8 = t_8.view(t_10, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_4 = t_4.view(t_10, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_5 = t_5.view(t_10, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_8, t_4)
        t_4 += t_7
        t_8 = t_4.float()
        t_8 = torch.nn.functional.softmax(t_8, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_8.type_as(t_4)
        t_4 = self.l_49(t_4)
        t_5 = torch.matmul(t_4, t_5)
        t_5 = t_5.transpose(1, 2)
        t_5 = t_5.contiguous()
        t_10 = t_5.view(t_10, -1, 4096)
        t_10 = self.l_50(t_10)
        t_5 = self.l_51(t_10)
        t_5 = t_9 + t_5
        t_7 = (t_10, None, t_7)
        t_5 = (t_5,)
        t_7 = t_7[slice(1, None, None)]
        t_7 = t_5 + t_7
        t_5 = t_7[0]
        t_10 = self.l_52(t_5)
        t_7 = t_7[slice(2, None, None)]
        t_7 = t_3 + t_7
        t_10 = self.l_53(t_10)
        t_10 = torch.nn.functional.relu(t_10, inplace=False)
        t_10 = self.l_54(t_10)
        t_10 = self.l_55(t_10)
        t_10 = self.l_56(t_10)
        t_10 = t_5 + t_10
        t_2 = (t_10, t_2)
        t_7 = t_2 + t_7
        t_2 = t_7[slice(None, 2, None)]
        t_2 = t_2[0]
        t_10 = t_7[2]
        t_7 = t_7[3]
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3085
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3087
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3089
        return list(flatten((x0, t_2, t_10, t_7)))

    def state_dict(self, *args, **kwargs):
        # we return the state dict of this part as it should be in the original model
        return state_dict(self, *args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return load_state_dict(self, *args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        # we return the named parameters of this part as it should be in the original model
        return named_parameters(self, *args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        # we return the named buffers of this part as it should be in the original model
        return named_buffers(self, *args, **kwargs)

    def cpu(self):
        return cpu(self)

    def cuda(self, device=None):
        return cuda(self, device=device)

    def to(self, *args, **kwargs):
        return to(self, *args, **kwargs)


class Partition10(nn.Module):
    LAYER_SCOPES = [
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout]',
        ]
    TENSORS = [
        ]
    def __init__(self, layers, tensors, device='cuda:10'):
        super().__init__()

        # Initialize partition layers
        for idx, layer_scope in enumerate(self.LAYER_SCOPES):
            self.add_module(f'l_{idx}' ,layers[layer_scope])

        # Initialize partition tensors (params and buffs)
        b = p = 0
        for tensor_scope in self.TENSORS:
            tensor = tensors[tensor_scope]
            if isinstance(tensor, nn.Parameter):
                self.register_parameter(f'p_{p}', tensor)
                p += 1
            else:
                self.register_buffer(f'b_{b}', tensor)
                b += 1

        self.device = torch.device(device)
        self.input_structure = [1, 1, 1, 1]
        self.lookup = {'l_0': 'decoder.block.6.layer.0.layer_norm',
                        'l_1': 'decoder.block.6.layer.0.SelfAttention.q',
                        'l_2': 'decoder.block.6.layer.0.SelfAttention.k',
                        'l_3': 'decoder.block.6.layer.0.SelfAttention.v',
                        'l_4': 'decoder.block.6.layer.0.SelfAttention.dropout',
                        'l_5': 'decoder.block.6.layer.0.SelfAttention.o',
                        'l_6': 'decoder.block.6.layer.0.dropout',
                        'l_7': 'decoder.block.6.layer.1.layer_norm',
                        'l_8': 'decoder.block.6.layer.1.EncDecAttention.q',
                        'l_9': 'decoder.block.6.layer.1.EncDecAttention.k',
                        'l_10': 'decoder.block.6.layer.1.EncDecAttention.v',
                        'l_11': 'decoder.block.6.layer.1.EncDecAttention.dropout',
                        'l_12': 'decoder.block.6.layer.1.EncDecAttention.o',
                        'l_13': 'decoder.block.6.layer.1.dropout',
                        'l_14': 'decoder.block.6.layer.2.layer_norm',
                        'l_15': 'decoder.block.6.layer.2.DenseReluDense.wi',
                        'l_16': 'decoder.block.6.layer.2.DenseReluDense.dropout',
                        'l_17': 'decoder.block.6.layer.2.DenseReluDense.wo',
                        'l_18': 'decoder.block.6.layer.2.dropout',
                        'l_19': 'decoder.block.7.layer.0.layer_norm',
                        'l_20': 'decoder.block.7.layer.0.SelfAttention.q',
                        'l_21': 'decoder.block.7.layer.0.SelfAttention.k',
                        'l_22': 'decoder.block.7.layer.0.SelfAttention.v',
                        'l_23': 'decoder.block.7.layer.0.SelfAttention.dropout',
                        'l_24': 'decoder.block.7.layer.0.SelfAttention.o',
                        'l_25': 'decoder.block.7.layer.0.dropout',
                        'l_26': 'decoder.block.7.layer.1.layer_norm',
                        'l_27': 'decoder.block.7.layer.1.EncDecAttention.q',
                        'l_28': 'decoder.block.7.layer.1.EncDecAttention.k',
                        'l_29': 'decoder.block.7.layer.1.EncDecAttention.v',
                        'l_30': 'decoder.block.7.layer.1.EncDecAttention.dropout',
                        'l_31': 'decoder.block.7.layer.1.EncDecAttention.o',
                        'l_32': 'decoder.block.7.layer.1.dropout',
                        'l_33': 'decoder.block.7.layer.2.layer_norm',
                        'l_34': 'decoder.block.7.layer.2.DenseReluDense.wi',
                        'l_35': 'decoder.block.7.layer.2.DenseReluDense.dropout',
                        'l_36': 'decoder.block.7.layer.2.DenseReluDense.wo',
                        'l_37': 'decoder.block.7.layer.2.dropout',
                        'l_38': 'decoder.block.8.layer.0.layer_norm',
                        'l_39': 'decoder.block.8.layer.0.SelfAttention.q',
                        'l_40': 'decoder.block.8.layer.0.SelfAttention.k',
                        'l_41': 'decoder.block.8.layer.0.SelfAttention.v',
                        'l_42': 'decoder.block.8.layer.0.SelfAttention.dropout',
                        'l_43': 'decoder.block.8.layer.0.SelfAttention.o',
                        'l_44': 'decoder.block.8.layer.0.dropout',
                        'l_45': 'decoder.block.8.layer.1.layer_norm',
                        'l_46': 'decoder.block.8.layer.1.EncDecAttention.q',
                        'l_47': 'decoder.block.8.layer.1.EncDecAttention.k',
                        'l_48': 'decoder.block.8.layer.1.EncDecAttention.v',
                        'l_49': 'decoder.block.8.layer.1.EncDecAttention.dropout',
                        'l_50': 'decoder.block.8.layer.1.EncDecAttention.o',
                        'l_51': 'decoder.block.8.layer.1.dropout',
                        'l_52': 'decoder.block.8.layer.2.layer_norm',
                        'l_53': 'decoder.block.8.layer.2.DenseReluDense.wi',
                        'l_54': 'decoder.block.8.layer.2.DenseReluDense.dropout',
                        'l_55': 'decoder.block.8.layer.2.DenseReluDense.wo',
                        'l_56': 'decoder.block.8.layer.2.dropout'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[6]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_36
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[7]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout] <=> self.l_37
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_38
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_39
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_40
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_41
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_42
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_43
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_44
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_45
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_46
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_47
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_48
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_49
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_50
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_51
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_52
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_53
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_54
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_55
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[8]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout] <=> self.l_56
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout] <=> x0
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3085 <=> x1
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3087 <=> x2
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3089 <=> x3
        x0, x1, x2, x3 = unflatten(args, self.input_structure)
        t_0 = self.l_9(x0)
        t_1 = self.l_10(x0)
        t_2 = self.l_28(x0)
        t_3 = self.l_29(x0)
        t_4 = self.l_47(x0)
        t_5 = self.l_48(x0)
        t_6 = self.l_0(x1)
        t_7 = self.l_1(t_6)
        t_8 = self.l_2(t_6)
        t_9 = self.l_3(t_6)
        t_6 = t_6.shape
        t_6 = t_6[slice(None, 2, None)]
        t_6 = t_6[0]
        t_7 = t_7.view(t_6, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_8 = t_8.view(t_6, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_9 = t_9.view(t_6, -1, 32, 128)
        t_9 = t_9.transpose(1, 2)
        t_8 = t_8.transpose(3, 2)
        t_8 = torch.matmul(t_7, t_8)
        t_8 += x2
        t_7 = t_8.float()
        t_7 = torch.nn.functional.softmax(t_7, dim=-1, _stacklevel=3, dtype=None)
        t_8 = t_7.type_as(t_8)
        t_8 = self.l_4(t_8)
        t_9 = torch.matmul(t_8, t_9)
        t_9 = t_9.transpose(1, 2)
        t_9 = t_9.contiguous()
        t_6 = t_9.view(t_6, -1, 4096)
        t_6 = self.l_5(t_6)
        t_9 = self.l_6(t_6)
        t_9 = x1 + t_9
        t_6 = (t_6, None, x2)
        t_9 = (t_9,)
        t_6 = t_6[slice(1, None, None)]
        t_6 = t_9 + t_6
        t_9 = t_6[slice(None, 2, None)]
        t_8 = t_9[0]
        t_7 = self.l_7(t_8)
        t_9 = t_9[1]
        t_6 = t_6[slice(2, None, None)]
        t_10 = self.l_8(t_7)
        t_7 = t_7.shape
        t_7 = t_7[slice(None, 2, None)]
        t_7 = t_7[0]
        t_10 = t_10.view(t_7, -1, 32, 128)
        t_10 = t_10.transpose(1, 2)
        t_0 = t_0.view(t_7, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_1 = t_1.view(t_7, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_0 = t_0.transpose(3, 2)
        t_0 = torch.matmul(t_10, t_0)
        t_0 += x3
        t_10 = t_0.float()
        t_10 = torch.nn.functional.softmax(t_10, dim=-1, _stacklevel=3, dtype=None)
        t_0 = t_10.type_as(t_0)
        t_0 = self.l_11(t_0)
        t_1 = torch.matmul(t_0, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_7 = t_1.view(t_7, -1, 4096)
        t_7 = self.l_12(t_7)
        t_1 = self.l_13(t_7)
        t_1 = t_8 + t_1
        t_7 = (t_7, None, x3)
        t_1 = (t_1,)
        t_7 = t_7[slice(1, None, None)]
        t_7 = t_1 + t_7
        t_1 = t_7[0]
        t_8 = self.l_14(t_1)
        t_7 = t_7[slice(2, None, None)]
        t_7 = t_6 + t_7
        t_8 = self.l_15(t_8)
        t_8 = torch.nn.functional.relu(t_8, inplace=False)
        t_8 = self.l_16(t_8)
        t_8 = self.l_17(t_8)
        t_8 = self.l_18(t_8)
        t_8 = t_1 + t_8
        t_9 = (t_8, t_9)
        t_7 = t_9 + t_7
        t_9 = t_7[slice(None, 2, None)]
        t_9 = t_9[0]
        t_8 = self.l_19(t_9)
        t_1 = t_7[2]
        t_7 = t_7[3]
        t_6 = self.l_20(t_8)
        t_0 = self.l_21(t_8)
        t_10 = self.l_22(t_8)
        t_8 = t_8.shape
        t_8 = t_8[slice(None, 2, None)]
        t_8 = t_8[0]
        t_6 = t_6.view(t_8, -1, 32, 128)
        t_6 = t_6.transpose(1, 2)
        t_0 = t_0.view(t_8, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_10 = t_10.view(t_8, -1, 32, 128)
        t_10 = t_10.transpose(1, 2)
        t_0 = t_0.transpose(3, 2)
        t_0 = torch.matmul(t_6, t_0)
        t_0 += t_1
        t_6 = t_0.float()
        t_6 = torch.nn.functional.softmax(t_6, dim=-1, _stacklevel=3, dtype=None)
        t_0 = t_6.type_as(t_0)
        t_0 = self.l_23(t_0)
        t_10 = torch.matmul(t_0, t_10)
        t_10 = t_10.transpose(1, 2)
        t_10 = t_10.contiguous()
        t_8 = t_10.view(t_8, -1, 4096)
        t_8 = self.l_24(t_8)
        t_10 = self.l_25(t_8)
        t_10 = t_9 + t_10
        t_1 = (t_8, None, t_1)
        t_10 = (t_10,)
        t_1 = t_1[slice(1, None, None)]
        t_1 = t_10 + t_1
        t_10 = t_1[slice(None, 2, None)]
        t_8 = t_10[0]
        t_9 = self.l_26(t_8)
        t_10 = t_10[1]
        t_1 = t_1[slice(2, None, None)]
        t_0 = self.l_27(t_9)
        t_9 = t_9.shape
        t_9 = t_9[slice(None, 2, None)]
        t_9 = t_9[0]
        t_0 = t_0.view(t_9, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_2 = t_2.view(t_9, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_3 = t_3.view(t_9, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_2 = t_2.transpose(3, 2)
        t_2 = torch.matmul(t_0, t_2)
        t_2 += t_7
        t_0 = t_2.float()
        t_0 = torch.nn.functional.softmax(t_0, dim=-1, _stacklevel=3, dtype=None)
        t_2 = t_0.type_as(t_2)
        t_2 = self.l_30(t_2)
        t_3 = torch.matmul(t_2, t_3)
        t_3 = t_3.transpose(1, 2)
        t_3 = t_3.contiguous()
        t_9 = t_3.view(t_9, -1, 4096)
        t_9 = self.l_31(t_9)
        t_3 = self.l_32(t_9)
        t_3 = t_8 + t_3
        t_7 = (t_9, None, t_7)
        t_3 = (t_3,)
        t_7 = t_7[slice(1, None, None)]
        t_7 = t_3 + t_7
        t_3 = t_7[0]
        t_9 = self.l_33(t_3)
        t_7 = t_7[slice(2, None, None)]
        t_7 = t_1 + t_7
        t_9 = self.l_34(t_9)
        t_9 = torch.nn.functional.relu(t_9, inplace=False)
        t_9 = self.l_35(t_9)
        t_9 = self.l_36(t_9)
        t_9 = self.l_37(t_9)
        t_9 = t_3 + t_9
        t_10 = (t_9, t_10)
        t_7 = t_10 + t_7
        t_10 = t_7[slice(None, 2, None)]
        t_10 = t_10[0]
        t_9 = self.l_38(t_10)
        t_3 = t_7[2]
        t_7 = t_7[3]
        t_1 = self.l_39(t_9)
        t_8 = self.l_40(t_9)
        t_2 = self.l_41(t_9)
        t_9 = t_9.shape
        t_9 = t_9[slice(None, 2, None)]
        t_9 = t_9[0]
        t_1 = t_1.view(t_9, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_8 = t_8.view(t_9, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_2 = t_2.view(t_9, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_8 = t_8.transpose(3, 2)
        t_8 = torch.matmul(t_1, t_8)
        t_8 += t_3
        t_1 = t_8.float()
        t_1 = torch.nn.functional.softmax(t_1, dim=-1, _stacklevel=3, dtype=None)
        t_8 = t_1.type_as(t_8)
        t_8 = self.l_42(t_8)
        t_2 = torch.matmul(t_8, t_2)
        t_2 = t_2.transpose(1, 2)
        t_2 = t_2.contiguous()
        t_9 = t_2.view(t_9, -1, 4096)
        t_9 = self.l_43(t_9)
        t_2 = self.l_44(t_9)
        t_2 = t_10 + t_2
        t_3 = (t_9, None, t_3)
        t_2 = (t_2,)
        t_3 = t_3[slice(1, None, None)]
        t_3 = t_2 + t_3
        t_2 = t_3[slice(None, 2, None)]
        t_9 = t_2[0]
        t_10 = self.l_45(t_9)
        t_2 = t_2[1]
        t_3 = t_3[slice(2, None, None)]
        t_8 = self.l_46(t_10)
        t_10 = t_10.shape
        t_10 = t_10[slice(None, 2, None)]
        t_10 = t_10[0]
        t_8 = t_8.view(t_10, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_4 = t_4.view(t_10, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_5 = t_5.view(t_10, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_8, t_4)
        t_4 += t_7
        t_8 = t_4.float()
        t_8 = torch.nn.functional.softmax(t_8, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_8.type_as(t_4)
        t_4 = self.l_49(t_4)
        t_5 = torch.matmul(t_4, t_5)
        t_5 = t_5.transpose(1, 2)
        t_5 = t_5.contiguous()
        t_10 = t_5.view(t_10, -1, 4096)
        t_10 = self.l_50(t_10)
        t_5 = self.l_51(t_10)
        t_5 = t_9 + t_5
        t_7 = (t_10, None, t_7)
        t_5 = (t_5,)
        t_7 = t_7[slice(1, None, None)]
        t_7 = t_5 + t_7
        t_5 = t_7[0]
        t_10 = self.l_52(t_5)
        t_7 = t_7[slice(2, None, None)]
        t_7 = t_3 + t_7
        t_10 = self.l_53(t_10)
        t_10 = torch.nn.functional.relu(t_10, inplace=False)
        t_10 = self.l_54(t_10)
        t_10 = self.l_55(t_10)
        t_10 = self.l_56(t_10)
        t_10 = t_5 + t_10
        t_2 = (t_10, t_2)
        t_7 = t_2 + t_7
        t_2 = t_7[slice(None, 2, None)]
        t_2 = t_2[0]
        t_10 = t_7[2]
        t_7 = t_7[3]
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3538
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3540
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3542
        return list(flatten((x0, t_2, t_10, t_7)))

    def state_dict(self, *args, **kwargs):
        # we return the state dict of this part as it should be in the original model
        return state_dict(self, *args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return load_state_dict(self, *args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        # we return the named parameters of this part as it should be in the original model
        return named_parameters(self, *args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        # we return the named buffers of this part as it should be in the original model
        return named_buffers(self, *args, **kwargs)

    def cpu(self):
        return cpu(self)

    def cuda(self, device=None):
        return cuda(self, device=device)

    def to(self, *args, **kwargs):
        return to(self, *args, **kwargs)


class Partition11(nn.Module):
    LAYER_SCOPES = [
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout]',
        ]
    TENSORS = [
        ]
    def __init__(self, layers, tensors, device='cuda:11'):
        super().__init__()

        # Initialize partition layers
        for idx, layer_scope in enumerate(self.LAYER_SCOPES):
            self.add_module(f'l_{idx}' ,layers[layer_scope])

        # Initialize partition tensors (params and buffs)
        b = p = 0
        for tensor_scope in self.TENSORS:
            tensor = tensors[tensor_scope]
            if isinstance(tensor, nn.Parameter):
                self.register_parameter(f'p_{p}', tensor)
                p += 1
            else:
                self.register_buffer(f'b_{b}', tensor)
                b += 1

        self.device = torch.device(device)
        self.input_structure = [1, 1, 1, 1]
        self.lookup = {'l_0': 'decoder.block.9.layer.0.layer_norm',
                        'l_1': 'decoder.block.9.layer.0.SelfAttention.q',
                        'l_2': 'decoder.block.9.layer.0.SelfAttention.k',
                        'l_3': 'decoder.block.9.layer.0.SelfAttention.v',
                        'l_4': 'decoder.block.9.layer.0.SelfAttention.dropout',
                        'l_5': 'decoder.block.9.layer.0.SelfAttention.o',
                        'l_6': 'decoder.block.9.layer.0.dropout',
                        'l_7': 'decoder.block.9.layer.1.layer_norm',
                        'l_8': 'decoder.block.9.layer.1.EncDecAttention.q',
                        'l_9': 'decoder.block.9.layer.1.EncDecAttention.k',
                        'l_10': 'decoder.block.9.layer.1.EncDecAttention.v',
                        'l_11': 'decoder.block.9.layer.1.EncDecAttention.dropout',
                        'l_12': 'decoder.block.9.layer.1.EncDecAttention.o',
                        'l_13': 'decoder.block.9.layer.1.dropout',
                        'l_14': 'decoder.block.9.layer.2.layer_norm',
                        'l_15': 'decoder.block.9.layer.2.DenseReluDense.wi',
                        'l_16': 'decoder.block.9.layer.2.DenseReluDense.dropout',
                        'l_17': 'decoder.block.9.layer.2.DenseReluDense.wo',
                        'l_18': 'decoder.block.9.layer.2.dropout',
                        'l_19': 'decoder.block.10.layer.0.layer_norm',
                        'l_20': 'decoder.block.10.layer.0.SelfAttention.q',
                        'l_21': 'decoder.block.10.layer.0.SelfAttention.k',
                        'l_22': 'decoder.block.10.layer.0.SelfAttention.v',
                        'l_23': 'decoder.block.10.layer.0.SelfAttention.dropout',
                        'l_24': 'decoder.block.10.layer.0.SelfAttention.o',
                        'l_25': 'decoder.block.10.layer.0.dropout',
                        'l_26': 'decoder.block.10.layer.1.layer_norm',
                        'l_27': 'decoder.block.10.layer.1.EncDecAttention.q',
                        'l_28': 'decoder.block.10.layer.1.EncDecAttention.k',
                        'l_29': 'decoder.block.10.layer.1.EncDecAttention.v',
                        'l_30': 'decoder.block.10.layer.1.EncDecAttention.dropout',
                        'l_31': 'decoder.block.10.layer.1.EncDecAttention.o',
                        'l_32': 'decoder.block.10.layer.1.dropout',
                        'l_33': 'decoder.block.10.layer.2.layer_norm',
                        'l_34': 'decoder.block.10.layer.2.DenseReluDense.wi',
                        'l_35': 'decoder.block.10.layer.2.DenseReluDense.dropout',
                        'l_36': 'decoder.block.10.layer.2.DenseReluDense.wo',
                        'l_37': 'decoder.block.10.layer.2.dropout',
                        'l_38': 'decoder.block.11.layer.0.layer_norm',
                        'l_39': 'decoder.block.11.layer.0.SelfAttention.q',
                        'l_40': 'decoder.block.11.layer.0.SelfAttention.k',
                        'l_41': 'decoder.block.11.layer.0.SelfAttention.v',
                        'l_42': 'decoder.block.11.layer.0.SelfAttention.dropout',
                        'l_43': 'decoder.block.11.layer.0.SelfAttention.o',
                        'l_44': 'decoder.block.11.layer.0.dropout',
                        'l_45': 'decoder.block.11.layer.1.layer_norm',
                        'l_46': 'decoder.block.11.layer.1.EncDecAttention.q',
                        'l_47': 'decoder.block.11.layer.1.EncDecAttention.k',
                        'l_48': 'decoder.block.11.layer.1.EncDecAttention.v',
                        'l_49': 'decoder.block.11.layer.1.EncDecAttention.dropout',
                        'l_50': 'decoder.block.11.layer.1.EncDecAttention.o',
                        'l_51': 'decoder.block.11.layer.1.dropout',
                        'l_52': 'decoder.block.11.layer.2.layer_norm',
                        'l_53': 'decoder.block.11.layer.2.DenseReluDense.wi',
                        'l_54': 'decoder.block.11.layer.2.DenseReluDense.dropout',
                        'l_55': 'decoder.block.11.layer.2.DenseReluDense.wo',
                        'l_56': 'decoder.block.11.layer.2.dropout'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[9]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_36
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[10]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout] <=> self.l_37
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_38
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_39
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_40
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_41
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_42
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_43
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_44
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_45
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_46
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_47
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_48
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_49
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_50
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_51
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_52
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_53
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_54
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_55
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[11]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout] <=> self.l_56
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout] <=> x0
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3538 <=> x1
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3540 <=> x2
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3542 <=> x3
        x0, x1, x2, x3 = unflatten(args, self.input_structure)
        t_0 = self.l_9(x0)
        t_1 = self.l_10(x0)
        t_2 = self.l_28(x0)
        t_3 = self.l_29(x0)
        t_4 = self.l_47(x0)
        t_5 = self.l_48(x0)
        t_6 = self.l_0(x1)
        t_7 = self.l_1(t_6)
        t_8 = self.l_2(t_6)
        t_9 = self.l_3(t_6)
        t_6 = t_6.shape
        t_6 = t_6[slice(None, 2, None)]
        t_6 = t_6[0]
        t_7 = t_7.view(t_6, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_8 = t_8.view(t_6, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_9 = t_9.view(t_6, -1, 32, 128)
        t_9 = t_9.transpose(1, 2)
        t_8 = t_8.transpose(3, 2)
        t_8 = torch.matmul(t_7, t_8)
        t_8 += x2
        t_7 = t_8.float()
        t_7 = torch.nn.functional.softmax(t_7, dim=-1, _stacklevel=3, dtype=None)
        t_8 = t_7.type_as(t_8)
        t_8 = self.l_4(t_8)
        t_9 = torch.matmul(t_8, t_9)
        t_9 = t_9.transpose(1, 2)
        t_9 = t_9.contiguous()
        t_6 = t_9.view(t_6, -1, 4096)
        t_6 = self.l_5(t_6)
        t_9 = self.l_6(t_6)
        t_9 = x1 + t_9
        t_6 = (t_6, None, x2)
        t_9 = (t_9,)
        t_6 = t_6[slice(1, None, None)]
        t_6 = t_9 + t_6
        t_9 = t_6[slice(None, 2, None)]
        t_8 = t_9[0]
        t_7 = self.l_7(t_8)
        t_9 = t_9[1]
        t_6 = t_6[slice(2, None, None)]
        t_10 = self.l_8(t_7)
        t_7 = t_7.shape
        t_7 = t_7[slice(None, 2, None)]
        t_7 = t_7[0]
        t_10 = t_10.view(t_7, -1, 32, 128)
        t_10 = t_10.transpose(1, 2)
        t_0 = t_0.view(t_7, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_1 = t_1.view(t_7, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_0 = t_0.transpose(3, 2)
        t_0 = torch.matmul(t_10, t_0)
        t_0 += x3
        t_10 = t_0.float()
        t_10 = torch.nn.functional.softmax(t_10, dim=-1, _stacklevel=3, dtype=None)
        t_0 = t_10.type_as(t_0)
        t_0 = self.l_11(t_0)
        t_1 = torch.matmul(t_0, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_7 = t_1.view(t_7, -1, 4096)
        t_7 = self.l_12(t_7)
        t_1 = self.l_13(t_7)
        t_1 = t_8 + t_1
        t_7 = (t_7, None, x3)
        t_1 = (t_1,)
        t_7 = t_7[slice(1, None, None)]
        t_7 = t_1 + t_7
        t_1 = t_7[0]
        t_8 = self.l_14(t_1)
        t_7 = t_7[slice(2, None, None)]
        t_7 = t_6 + t_7
        t_8 = self.l_15(t_8)
        t_8 = torch.nn.functional.relu(t_8, inplace=False)
        t_8 = self.l_16(t_8)
        t_8 = self.l_17(t_8)
        t_8 = self.l_18(t_8)
        t_8 = t_1 + t_8
        t_9 = (t_8, t_9)
        t_7 = t_9 + t_7
        t_9 = t_7[slice(None, 2, None)]
        t_9 = t_9[0]
        t_8 = self.l_19(t_9)
        t_1 = t_7[2]
        t_7 = t_7[3]
        t_6 = self.l_20(t_8)
        t_0 = self.l_21(t_8)
        t_10 = self.l_22(t_8)
        t_8 = t_8.shape
        t_8 = t_8[slice(None, 2, None)]
        t_8 = t_8[0]
        t_6 = t_6.view(t_8, -1, 32, 128)
        t_6 = t_6.transpose(1, 2)
        t_0 = t_0.view(t_8, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_10 = t_10.view(t_8, -1, 32, 128)
        t_10 = t_10.transpose(1, 2)
        t_0 = t_0.transpose(3, 2)
        t_0 = torch.matmul(t_6, t_0)
        t_0 += t_1
        t_6 = t_0.float()
        t_6 = torch.nn.functional.softmax(t_6, dim=-1, _stacklevel=3, dtype=None)
        t_0 = t_6.type_as(t_0)
        t_0 = self.l_23(t_0)
        t_10 = torch.matmul(t_0, t_10)
        t_10 = t_10.transpose(1, 2)
        t_10 = t_10.contiguous()
        t_8 = t_10.view(t_8, -1, 4096)
        t_8 = self.l_24(t_8)
        t_10 = self.l_25(t_8)
        t_10 = t_9 + t_10
        t_1 = (t_8, None, t_1)
        t_10 = (t_10,)
        t_1 = t_1[slice(1, None, None)]
        t_1 = t_10 + t_1
        t_10 = t_1[slice(None, 2, None)]
        t_8 = t_10[0]
        t_9 = self.l_26(t_8)
        t_10 = t_10[1]
        t_1 = t_1[slice(2, None, None)]
        t_0 = self.l_27(t_9)
        t_9 = t_9.shape
        t_9 = t_9[slice(None, 2, None)]
        t_9 = t_9[0]
        t_0 = t_0.view(t_9, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_2 = t_2.view(t_9, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_3 = t_3.view(t_9, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_2 = t_2.transpose(3, 2)
        t_2 = torch.matmul(t_0, t_2)
        t_2 += t_7
        t_0 = t_2.float()
        t_0 = torch.nn.functional.softmax(t_0, dim=-1, _stacklevel=3, dtype=None)
        t_2 = t_0.type_as(t_2)
        t_2 = self.l_30(t_2)
        t_3 = torch.matmul(t_2, t_3)
        t_3 = t_3.transpose(1, 2)
        t_3 = t_3.contiguous()
        t_9 = t_3.view(t_9, -1, 4096)
        t_9 = self.l_31(t_9)
        t_3 = self.l_32(t_9)
        t_3 = t_8 + t_3
        t_7 = (t_9, None, t_7)
        t_3 = (t_3,)
        t_7 = t_7[slice(1, None, None)]
        t_7 = t_3 + t_7
        t_3 = t_7[0]
        t_9 = self.l_33(t_3)
        t_7 = t_7[slice(2, None, None)]
        t_7 = t_1 + t_7
        t_9 = self.l_34(t_9)
        t_9 = torch.nn.functional.relu(t_9, inplace=False)
        t_9 = self.l_35(t_9)
        t_9 = self.l_36(t_9)
        t_9 = self.l_37(t_9)
        t_9 = t_3 + t_9
        t_10 = (t_9, t_10)
        t_7 = t_10 + t_7
        t_10 = t_7[slice(None, 2, None)]
        t_10 = t_10[0]
        t_9 = self.l_38(t_10)
        t_3 = t_7[2]
        t_7 = t_7[3]
        t_1 = self.l_39(t_9)
        t_8 = self.l_40(t_9)
        t_2 = self.l_41(t_9)
        t_9 = t_9.shape
        t_9 = t_9[slice(None, 2, None)]
        t_9 = t_9[0]
        t_1 = t_1.view(t_9, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_8 = t_8.view(t_9, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_2 = t_2.view(t_9, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_8 = t_8.transpose(3, 2)
        t_8 = torch.matmul(t_1, t_8)
        t_8 += t_3
        t_1 = t_8.float()
        t_1 = torch.nn.functional.softmax(t_1, dim=-1, _stacklevel=3, dtype=None)
        t_8 = t_1.type_as(t_8)
        t_8 = self.l_42(t_8)
        t_2 = torch.matmul(t_8, t_2)
        t_2 = t_2.transpose(1, 2)
        t_2 = t_2.contiguous()
        t_9 = t_2.view(t_9, -1, 4096)
        t_9 = self.l_43(t_9)
        t_2 = self.l_44(t_9)
        t_2 = t_10 + t_2
        t_3 = (t_9, None, t_3)
        t_2 = (t_2,)
        t_3 = t_3[slice(1, None, None)]
        t_3 = t_2 + t_3
        t_2 = t_3[slice(None, 2, None)]
        t_9 = t_2[0]
        t_10 = self.l_45(t_9)
        t_2 = t_2[1]
        t_3 = t_3[slice(2, None, None)]
        t_8 = self.l_46(t_10)
        t_10 = t_10.shape
        t_10 = t_10[slice(None, 2, None)]
        t_10 = t_10[0]
        t_8 = t_8.view(t_10, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_4 = t_4.view(t_10, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_5 = t_5.view(t_10, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_8, t_4)
        t_4 += t_7
        t_8 = t_4.float()
        t_8 = torch.nn.functional.softmax(t_8, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_8.type_as(t_4)
        t_4 = self.l_49(t_4)
        t_5 = torch.matmul(t_4, t_5)
        t_5 = t_5.transpose(1, 2)
        t_5 = t_5.contiguous()
        t_10 = t_5.view(t_10, -1, 4096)
        t_10 = self.l_50(t_10)
        t_5 = self.l_51(t_10)
        t_5 = t_9 + t_5
        t_7 = (t_10, None, t_7)
        t_5 = (t_5,)
        t_7 = t_7[slice(1, None, None)]
        t_7 = t_5 + t_7
        t_5 = t_7[0]
        t_10 = self.l_52(t_5)
        t_7 = t_7[slice(2, None, None)]
        t_7 = t_3 + t_7
        t_10 = self.l_53(t_10)
        t_10 = torch.nn.functional.relu(t_10, inplace=False)
        t_10 = self.l_54(t_10)
        t_10 = self.l_55(t_10)
        t_10 = self.l_56(t_10)
        t_10 = t_5 + t_10
        t_2 = (t_10, t_2)
        t_7 = t_2 + t_7
        t_2 = t_7[slice(None, 2, None)]
        t_2 = t_2[0]
        t_10 = t_7[2]
        t_7 = t_7[3]
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3991
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3993
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3995
        return list(flatten((x0, t_2, t_10, t_7)))

    def state_dict(self, *args, **kwargs):
        # we return the state dict of this part as it should be in the original model
        return state_dict(self, *args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return load_state_dict(self, *args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        # we return the named parameters of this part as it should be in the original model
        return named_parameters(self, *args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        # we return the named buffers of this part as it should be in the original model
        return named_buffers(self, *args, **kwargs)

    def cpu(self):
        return cpu(self)

    def cuda(self, device=None):
        return cuda(self, device=device)

    def to(self, *args, **kwargs):
        return to(self, *args, **kwargs)


class Partition12(nn.Module):
    LAYER_SCOPES = [
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout]',
        ]
    TENSORS = [
        ]
    def __init__(self, layers, tensors, device='cuda:12'):
        super().__init__()

        # Initialize partition layers
        for idx, layer_scope in enumerate(self.LAYER_SCOPES):
            self.add_module(f'l_{idx}' ,layers[layer_scope])

        # Initialize partition tensors (params and buffs)
        b = p = 0
        for tensor_scope in self.TENSORS:
            tensor = tensors[tensor_scope]
            if isinstance(tensor, nn.Parameter):
                self.register_parameter(f'p_{p}', tensor)
                p += 1
            else:
                self.register_buffer(f'b_{b}', tensor)
                b += 1

        self.device = torch.device(device)
        self.input_structure = [1, 1, 1, 1]
        self.lookup = {'l_0': 'decoder.block.12.layer.0.layer_norm',
                        'l_1': 'decoder.block.12.layer.0.SelfAttention.q',
                        'l_2': 'decoder.block.12.layer.0.SelfAttention.k',
                        'l_3': 'decoder.block.12.layer.0.SelfAttention.v',
                        'l_4': 'decoder.block.12.layer.0.SelfAttention.dropout',
                        'l_5': 'decoder.block.12.layer.0.SelfAttention.o',
                        'l_6': 'decoder.block.12.layer.0.dropout',
                        'l_7': 'decoder.block.12.layer.1.layer_norm',
                        'l_8': 'decoder.block.12.layer.1.EncDecAttention.q',
                        'l_9': 'decoder.block.12.layer.1.EncDecAttention.k',
                        'l_10': 'decoder.block.12.layer.1.EncDecAttention.v',
                        'l_11': 'decoder.block.12.layer.1.EncDecAttention.dropout',
                        'l_12': 'decoder.block.12.layer.1.EncDecAttention.o',
                        'l_13': 'decoder.block.12.layer.1.dropout',
                        'l_14': 'decoder.block.12.layer.2.layer_norm',
                        'l_15': 'decoder.block.12.layer.2.DenseReluDense.wi',
                        'l_16': 'decoder.block.12.layer.2.DenseReluDense.dropout',
                        'l_17': 'decoder.block.12.layer.2.DenseReluDense.wo',
                        'l_18': 'decoder.block.12.layer.2.dropout',
                        'l_19': 'decoder.block.13.layer.0.layer_norm',
                        'l_20': 'decoder.block.13.layer.0.SelfAttention.q',
                        'l_21': 'decoder.block.13.layer.0.SelfAttention.k',
                        'l_22': 'decoder.block.13.layer.0.SelfAttention.v',
                        'l_23': 'decoder.block.13.layer.0.SelfAttention.dropout',
                        'l_24': 'decoder.block.13.layer.0.SelfAttention.o',
                        'l_25': 'decoder.block.13.layer.0.dropout',
                        'l_26': 'decoder.block.13.layer.1.layer_norm',
                        'l_27': 'decoder.block.13.layer.1.EncDecAttention.q',
                        'l_28': 'decoder.block.13.layer.1.EncDecAttention.k',
                        'l_29': 'decoder.block.13.layer.1.EncDecAttention.v',
                        'l_30': 'decoder.block.13.layer.1.EncDecAttention.dropout',
                        'l_31': 'decoder.block.13.layer.1.EncDecAttention.o',
                        'l_32': 'decoder.block.13.layer.1.dropout',
                        'l_33': 'decoder.block.13.layer.2.layer_norm',
                        'l_34': 'decoder.block.13.layer.2.DenseReluDense.wi',
                        'l_35': 'decoder.block.13.layer.2.DenseReluDense.dropout',
                        'l_36': 'decoder.block.13.layer.2.DenseReluDense.wo',
                        'l_37': 'decoder.block.13.layer.2.dropout',
                        'l_38': 'decoder.block.14.layer.0.layer_norm',
                        'l_39': 'decoder.block.14.layer.0.SelfAttention.q',
                        'l_40': 'decoder.block.14.layer.0.SelfAttention.k',
                        'l_41': 'decoder.block.14.layer.0.SelfAttention.v',
                        'l_42': 'decoder.block.14.layer.0.SelfAttention.dropout',
                        'l_43': 'decoder.block.14.layer.0.SelfAttention.o',
                        'l_44': 'decoder.block.14.layer.0.dropout',
                        'l_45': 'decoder.block.14.layer.1.layer_norm',
                        'l_46': 'decoder.block.14.layer.1.EncDecAttention.q',
                        'l_47': 'decoder.block.14.layer.1.EncDecAttention.k',
                        'l_48': 'decoder.block.14.layer.1.EncDecAttention.v',
                        'l_49': 'decoder.block.14.layer.1.EncDecAttention.dropout',
                        'l_50': 'decoder.block.14.layer.1.EncDecAttention.o',
                        'l_51': 'decoder.block.14.layer.1.dropout',
                        'l_52': 'decoder.block.14.layer.2.layer_norm',
                        'l_53': 'decoder.block.14.layer.2.DenseReluDense.wi',
                        'l_54': 'decoder.block.14.layer.2.DenseReluDense.dropout',
                        'l_55': 'decoder.block.14.layer.2.DenseReluDense.wo',
                        'l_56': 'decoder.block.14.layer.2.dropout'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[12]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_36
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[13]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout] <=> self.l_37
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_38
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_39
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_40
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_41
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_42
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_43
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_44
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_45
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_46
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_47
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_48
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_49
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_50
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_51
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_52
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_53
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_54
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_55
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[14]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout] <=> self.l_56
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout] <=> x0
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3991 <=> x1
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3993 <=> x2
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___3995 <=> x3
        x0, x1, x2, x3 = unflatten(args, self.input_structure)
        t_0 = self.l_9(x0)
        t_1 = self.l_10(x0)
        t_2 = self.l_28(x0)
        t_3 = self.l_29(x0)
        t_4 = self.l_47(x0)
        t_5 = self.l_48(x0)
        t_6 = self.l_0(x1)
        t_7 = self.l_1(t_6)
        t_8 = self.l_2(t_6)
        t_9 = self.l_3(t_6)
        t_6 = t_6.shape
        t_6 = t_6[slice(None, 2, None)]
        t_6 = t_6[0]
        t_7 = t_7.view(t_6, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_8 = t_8.view(t_6, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_9 = t_9.view(t_6, -1, 32, 128)
        t_9 = t_9.transpose(1, 2)
        t_8 = t_8.transpose(3, 2)
        t_8 = torch.matmul(t_7, t_8)
        t_8 += x2
        t_7 = t_8.float()
        t_7 = torch.nn.functional.softmax(t_7, dim=-1, _stacklevel=3, dtype=None)
        t_8 = t_7.type_as(t_8)
        t_8 = self.l_4(t_8)
        t_9 = torch.matmul(t_8, t_9)
        t_9 = t_9.transpose(1, 2)
        t_9 = t_9.contiguous()
        t_6 = t_9.view(t_6, -1, 4096)
        t_6 = self.l_5(t_6)
        t_9 = self.l_6(t_6)
        t_9 = x1 + t_9
        t_6 = (t_6, None, x2)
        t_9 = (t_9,)
        t_6 = t_6[slice(1, None, None)]
        t_6 = t_9 + t_6
        t_9 = t_6[slice(None, 2, None)]
        t_8 = t_9[0]
        t_7 = self.l_7(t_8)
        t_9 = t_9[1]
        t_6 = t_6[slice(2, None, None)]
        t_10 = self.l_8(t_7)
        t_7 = t_7.shape
        t_7 = t_7[slice(None, 2, None)]
        t_7 = t_7[0]
        t_10 = t_10.view(t_7, -1, 32, 128)
        t_10 = t_10.transpose(1, 2)
        t_0 = t_0.view(t_7, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_1 = t_1.view(t_7, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_0 = t_0.transpose(3, 2)
        t_0 = torch.matmul(t_10, t_0)
        t_0 += x3
        t_10 = t_0.float()
        t_10 = torch.nn.functional.softmax(t_10, dim=-1, _stacklevel=3, dtype=None)
        t_0 = t_10.type_as(t_0)
        t_0 = self.l_11(t_0)
        t_1 = torch.matmul(t_0, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_7 = t_1.view(t_7, -1, 4096)
        t_7 = self.l_12(t_7)
        t_1 = self.l_13(t_7)
        t_1 = t_8 + t_1
        t_7 = (t_7, None, x3)
        t_1 = (t_1,)
        t_7 = t_7[slice(1, None, None)]
        t_7 = t_1 + t_7
        t_1 = t_7[0]
        t_8 = self.l_14(t_1)
        t_7 = t_7[slice(2, None, None)]
        t_7 = t_6 + t_7
        t_8 = self.l_15(t_8)
        t_8 = torch.nn.functional.relu(t_8, inplace=False)
        t_8 = self.l_16(t_8)
        t_8 = self.l_17(t_8)
        t_8 = self.l_18(t_8)
        t_8 = t_1 + t_8
        t_9 = (t_8, t_9)
        t_7 = t_9 + t_7
        t_9 = t_7[slice(None, 2, None)]
        t_9 = t_9[0]
        t_8 = self.l_19(t_9)
        t_1 = t_7[2]
        t_7 = t_7[3]
        t_6 = self.l_20(t_8)
        t_0 = self.l_21(t_8)
        t_10 = self.l_22(t_8)
        t_8 = t_8.shape
        t_8 = t_8[slice(None, 2, None)]
        t_8 = t_8[0]
        t_6 = t_6.view(t_8, -1, 32, 128)
        t_6 = t_6.transpose(1, 2)
        t_0 = t_0.view(t_8, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_10 = t_10.view(t_8, -1, 32, 128)
        t_10 = t_10.transpose(1, 2)
        t_0 = t_0.transpose(3, 2)
        t_0 = torch.matmul(t_6, t_0)
        t_0 += t_1
        t_6 = t_0.float()
        t_6 = torch.nn.functional.softmax(t_6, dim=-1, _stacklevel=3, dtype=None)
        t_0 = t_6.type_as(t_0)
        t_0 = self.l_23(t_0)
        t_10 = torch.matmul(t_0, t_10)
        t_10 = t_10.transpose(1, 2)
        t_10 = t_10.contiguous()
        t_8 = t_10.view(t_8, -1, 4096)
        t_8 = self.l_24(t_8)
        t_10 = self.l_25(t_8)
        t_10 = t_9 + t_10
        t_1 = (t_8, None, t_1)
        t_10 = (t_10,)
        t_1 = t_1[slice(1, None, None)]
        t_1 = t_10 + t_1
        t_10 = t_1[slice(None, 2, None)]
        t_8 = t_10[0]
        t_9 = self.l_26(t_8)
        t_10 = t_10[1]
        t_1 = t_1[slice(2, None, None)]
        t_0 = self.l_27(t_9)
        t_9 = t_9.shape
        t_9 = t_9[slice(None, 2, None)]
        t_9 = t_9[0]
        t_0 = t_0.view(t_9, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_2 = t_2.view(t_9, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_3 = t_3.view(t_9, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_2 = t_2.transpose(3, 2)
        t_2 = torch.matmul(t_0, t_2)
        t_2 += t_7
        t_0 = t_2.float()
        t_0 = torch.nn.functional.softmax(t_0, dim=-1, _stacklevel=3, dtype=None)
        t_2 = t_0.type_as(t_2)
        t_2 = self.l_30(t_2)
        t_3 = torch.matmul(t_2, t_3)
        t_3 = t_3.transpose(1, 2)
        t_3 = t_3.contiguous()
        t_9 = t_3.view(t_9, -1, 4096)
        t_9 = self.l_31(t_9)
        t_3 = self.l_32(t_9)
        t_3 = t_8 + t_3
        t_7 = (t_9, None, t_7)
        t_3 = (t_3,)
        t_7 = t_7[slice(1, None, None)]
        t_7 = t_3 + t_7
        t_3 = t_7[0]
        t_9 = self.l_33(t_3)
        t_7 = t_7[slice(2, None, None)]
        t_7 = t_1 + t_7
        t_9 = self.l_34(t_9)
        t_9 = torch.nn.functional.relu(t_9, inplace=False)
        t_9 = self.l_35(t_9)
        t_9 = self.l_36(t_9)
        t_9 = self.l_37(t_9)
        t_9 = t_3 + t_9
        t_10 = (t_9, t_10)
        t_7 = t_10 + t_7
        t_10 = t_7[slice(None, 2, None)]
        t_10 = t_10[0]
        t_9 = self.l_38(t_10)
        t_3 = t_7[2]
        t_7 = t_7[3]
        t_1 = self.l_39(t_9)
        t_8 = self.l_40(t_9)
        t_2 = self.l_41(t_9)
        t_9 = t_9.shape
        t_9 = t_9[slice(None, 2, None)]
        t_9 = t_9[0]
        t_1 = t_1.view(t_9, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_8 = t_8.view(t_9, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_2 = t_2.view(t_9, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_8 = t_8.transpose(3, 2)
        t_8 = torch.matmul(t_1, t_8)
        t_8 += t_3
        t_1 = t_8.float()
        t_1 = torch.nn.functional.softmax(t_1, dim=-1, _stacklevel=3, dtype=None)
        t_8 = t_1.type_as(t_8)
        t_8 = self.l_42(t_8)
        t_2 = torch.matmul(t_8, t_2)
        t_2 = t_2.transpose(1, 2)
        t_2 = t_2.contiguous()
        t_9 = t_2.view(t_9, -1, 4096)
        t_9 = self.l_43(t_9)
        t_2 = self.l_44(t_9)
        t_2 = t_10 + t_2
        t_3 = (t_9, None, t_3)
        t_2 = (t_2,)
        t_3 = t_3[slice(1, None, None)]
        t_3 = t_2 + t_3
        t_2 = t_3[slice(None, 2, None)]
        t_9 = t_2[0]
        t_10 = self.l_45(t_9)
        t_2 = t_2[1]
        t_3 = t_3[slice(2, None, None)]
        t_8 = self.l_46(t_10)
        t_10 = t_10.shape
        t_10 = t_10[slice(None, 2, None)]
        t_10 = t_10[0]
        t_8 = t_8.view(t_10, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_4 = t_4.view(t_10, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_5 = t_5.view(t_10, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_8, t_4)
        t_4 += t_7
        t_8 = t_4.float()
        t_8 = torch.nn.functional.softmax(t_8, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_8.type_as(t_4)
        t_4 = self.l_49(t_4)
        t_5 = torch.matmul(t_4, t_5)
        t_5 = t_5.transpose(1, 2)
        t_5 = t_5.contiguous()
        t_10 = t_5.view(t_10, -1, 4096)
        t_10 = self.l_50(t_10)
        t_5 = self.l_51(t_10)
        t_5 = t_9 + t_5
        t_7 = (t_10, None, t_7)
        t_5 = (t_5,)
        t_7 = t_7[slice(1, None, None)]
        t_7 = t_5 + t_7
        t_5 = t_7[0]
        t_10 = self.l_52(t_5)
        t_7 = t_7[slice(2, None, None)]
        t_7 = t_3 + t_7
        t_10 = self.l_53(t_10)
        t_10 = torch.nn.functional.relu(t_10, inplace=False)
        t_10 = self.l_54(t_10)
        t_10 = self.l_55(t_10)
        t_10 = self.l_56(t_10)
        t_10 = t_5 + t_10
        t_2 = (t_10, t_2)
        t_7 = t_2 + t_7
        t_2 = t_7[slice(None, 2, None)]
        t_2 = t_2[0]
        t_10 = t_7[2]
        t_7 = t_7[3]
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___4444
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___4446
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___4448
        return list(flatten((x0, t_2, t_10, t_7)))

    def state_dict(self, *args, **kwargs):
        # we return the state dict of this part as it should be in the original model
        return state_dict(self, *args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return load_state_dict(self, *args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        # we return the named parameters of this part as it should be in the original model
        return named_parameters(self, *args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        # we return the named buffers of this part as it should be in the original model
        return named_buffers(self, *args, **kwargs)

    def cpu(self):
        return cpu(self)

    def cuda(self, device=None):
        return cuda(self, device=device)

    def to(self, *args, **kwargs):
        return to(self, *args, **kwargs)


class Partition13(nn.Module):
    LAYER_SCOPES = [
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout]',
        ]
    TENSORS = [
        ]
    def __init__(self, layers, tensors, device='cuda:13'):
        super().__init__()

        # Initialize partition layers
        for idx, layer_scope in enumerate(self.LAYER_SCOPES):
            self.add_module(f'l_{idx}' ,layers[layer_scope])

        # Initialize partition tensors (params and buffs)
        b = p = 0
        for tensor_scope in self.TENSORS:
            tensor = tensors[tensor_scope]
            if isinstance(tensor, nn.Parameter):
                self.register_parameter(f'p_{p}', tensor)
                p += 1
            else:
                self.register_buffer(f'b_{b}', tensor)
                b += 1

        self.device = torch.device(device)
        self.input_structure = [1, 1, 1, 1]
        self.lookup = {'l_0': 'decoder.block.15.layer.0.layer_norm',
                        'l_1': 'decoder.block.15.layer.0.SelfAttention.q',
                        'l_2': 'decoder.block.15.layer.0.SelfAttention.k',
                        'l_3': 'decoder.block.15.layer.0.SelfAttention.v',
                        'l_4': 'decoder.block.15.layer.0.SelfAttention.dropout',
                        'l_5': 'decoder.block.15.layer.0.SelfAttention.o',
                        'l_6': 'decoder.block.15.layer.0.dropout',
                        'l_7': 'decoder.block.15.layer.1.layer_norm',
                        'l_8': 'decoder.block.15.layer.1.EncDecAttention.q',
                        'l_9': 'decoder.block.15.layer.1.EncDecAttention.k',
                        'l_10': 'decoder.block.15.layer.1.EncDecAttention.v',
                        'l_11': 'decoder.block.15.layer.1.EncDecAttention.dropout',
                        'l_12': 'decoder.block.15.layer.1.EncDecAttention.o',
                        'l_13': 'decoder.block.15.layer.1.dropout',
                        'l_14': 'decoder.block.15.layer.2.layer_norm',
                        'l_15': 'decoder.block.15.layer.2.DenseReluDense.wi',
                        'l_16': 'decoder.block.15.layer.2.DenseReluDense.dropout',
                        'l_17': 'decoder.block.15.layer.2.DenseReluDense.wo',
                        'l_18': 'decoder.block.15.layer.2.dropout',
                        'l_19': 'decoder.block.16.layer.0.layer_norm',
                        'l_20': 'decoder.block.16.layer.0.SelfAttention.q',
                        'l_21': 'decoder.block.16.layer.0.SelfAttention.k',
                        'l_22': 'decoder.block.16.layer.0.SelfAttention.v',
                        'l_23': 'decoder.block.16.layer.0.SelfAttention.dropout',
                        'l_24': 'decoder.block.16.layer.0.SelfAttention.o',
                        'l_25': 'decoder.block.16.layer.0.dropout',
                        'l_26': 'decoder.block.16.layer.1.layer_norm',
                        'l_27': 'decoder.block.16.layer.1.EncDecAttention.q',
                        'l_28': 'decoder.block.16.layer.1.EncDecAttention.k',
                        'l_29': 'decoder.block.16.layer.1.EncDecAttention.v',
                        'l_30': 'decoder.block.16.layer.1.EncDecAttention.dropout',
                        'l_31': 'decoder.block.16.layer.1.EncDecAttention.o',
                        'l_32': 'decoder.block.16.layer.1.dropout',
                        'l_33': 'decoder.block.16.layer.2.layer_norm',
                        'l_34': 'decoder.block.16.layer.2.DenseReluDense.wi',
                        'l_35': 'decoder.block.16.layer.2.DenseReluDense.dropout',
                        'l_36': 'decoder.block.16.layer.2.DenseReluDense.wo',
                        'l_37': 'decoder.block.16.layer.2.dropout',
                        'l_38': 'decoder.block.17.layer.0.layer_norm',
                        'l_39': 'decoder.block.17.layer.0.SelfAttention.q',
                        'l_40': 'decoder.block.17.layer.0.SelfAttention.k',
                        'l_41': 'decoder.block.17.layer.0.SelfAttention.v',
                        'l_42': 'decoder.block.17.layer.0.SelfAttention.dropout',
                        'l_43': 'decoder.block.17.layer.0.SelfAttention.o',
                        'l_44': 'decoder.block.17.layer.0.dropout',
                        'l_45': 'decoder.block.17.layer.1.layer_norm',
                        'l_46': 'decoder.block.17.layer.1.EncDecAttention.q',
                        'l_47': 'decoder.block.17.layer.1.EncDecAttention.k',
                        'l_48': 'decoder.block.17.layer.1.EncDecAttention.v',
                        'l_49': 'decoder.block.17.layer.1.EncDecAttention.dropout',
                        'l_50': 'decoder.block.17.layer.1.EncDecAttention.o',
                        'l_51': 'decoder.block.17.layer.1.dropout',
                        'l_52': 'decoder.block.17.layer.2.layer_norm',
                        'l_53': 'decoder.block.17.layer.2.DenseReluDense.wi',
                        'l_54': 'decoder.block.17.layer.2.DenseReluDense.dropout',
                        'l_55': 'decoder.block.17.layer.2.DenseReluDense.wo',
                        'l_56': 'decoder.block.17.layer.2.dropout'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[15]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_36
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[16]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout] <=> self.l_37
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_38
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_39
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_40
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_41
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_42
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_43
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_44
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_45
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_46
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_47
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_48
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_49
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_50
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_51
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_52
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_53
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_54
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_55
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[17]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout] <=> self.l_56
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout] <=> x0
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___4444 <=> x1
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___4446 <=> x2
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___4448 <=> x3
        x0, x1, x2, x3 = unflatten(args, self.input_structure)
        t_0 = self.l_9(x0)
        t_1 = self.l_10(x0)
        t_2 = self.l_28(x0)
        t_3 = self.l_29(x0)
        t_4 = self.l_47(x0)
        t_5 = self.l_48(x0)
        t_6 = self.l_0(x1)
        t_7 = self.l_1(t_6)
        t_8 = self.l_2(t_6)
        t_9 = self.l_3(t_6)
        t_6 = t_6.shape
        t_6 = t_6[slice(None, 2, None)]
        t_6 = t_6[0]
        t_7 = t_7.view(t_6, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_8 = t_8.view(t_6, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_9 = t_9.view(t_6, -1, 32, 128)
        t_9 = t_9.transpose(1, 2)
        t_8 = t_8.transpose(3, 2)
        t_8 = torch.matmul(t_7, t_8)
        t_8 += x2
        t_7 = t_8.float()
        t_7 = torch.nn.functional.softmax(t_7, dim=-1, _stacklevel=3, dtype=None)
        t_8 = t_7.type_as(t_8)
        t_8 = self.l_4(t_8)
        t_9 = torch.matmul(t_8, t_9)
        t_9 = t_9.transpose(1, 2)
        t_9 = t_9.contiguous()
        t_6 = t_9.view(t_6, -1, 4096)
        t_6 = self.l_5(t_6)
        t_9 = self.l_6(t_6)
        t_9 = x1 + t_9
        t_6 = (t_6, None, x2)
        t_9 = (t_9,)
        t_6 = t_6[slice(1, None, None)]
        t_6 = t_9 + t_6
        t_9 = t_6[slice(None, 2, None)]
        t_8 = t_9[0]
        t_7 = self.l_7(t_8)
        t_9 = t_9[1]
        t_6 = t_6[slice(2, None, None)]
        t_10 = self.l_8(t_7)
        t_7 = t_7.shape
        t_7 = t_7[slice(None, 2, None)]
        t_7 = t_7[0]
        t_10 = t_10.view(t_7, -1, 32, 128)
        t_10 = t_10.transpose(1, 2)
        t_0 = t_0.view(t_7, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_1 = t_1.view(t_7, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_0 = t_0.transpose(3, 2)
        t_0 = torch.matmul(t_10, t_0)
        t_0 += x3
        t_10 = t_0.float()
        t_10 = torch.nn.functional.softmax(t_10, dim=-1, _stacklevel=3, dtype=None)
        t_0 = t_10.type_as(t_0)
        t_0 = self.l_11(t_0)
        t_1 = torch.matmul(t_0, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_7 = t_1.view(t_7, -1, 4096)
        t_7 = self.l_12(t_7)
        t_1 = self.l_13(t_7)
        t_1 = t_8 + t_1
        t_7 = (t_7, None, x3)
        t_1 = (t_1,)
        t_7 = t_7[slice(1, None, None)]
        t_7 = t_1 + t_7
        t_1 = t_7[0]
        t_8 = self.l_14(t_1)
        t_7 = t_7[slice(2, None, None)]
        t_7 = t_6 + t_7
        t_8 = self.l_15(t_8)
        t_8 = torch.nn.functional.relu(t_8, inplace=False)
        t_8 = self.l_16(t_8)
        t_8 = self.l_17(t_8)
        t_8 = self.l_18(t_8)
        t_8 = t_1 + t_8
        t_9 = (t_8, t_9)
        t_7 = t_9 + t_7
        t_9 = t_7[slice(None, 2, None)]
        t_9 = t_9[0]
        t_8 = self.l_19(t_9)
        t_1 = t_7[2]
        t_7 = t_7[3]
        t_6 = self.l_20(t_8)
        t_0 = self.l_21(t_8)
        t_10 = self.l_22(t_8)
        t_8 = t_8.shape
        t_8 = t_8[slice(None, 2, None)]
        t_8 = t_8[0]
        t_6 = t_6.view(t_8, -1, 32, 128)
        t_6 = t_6.transpose(1, 2)
        t_0 = t_0.view(t_8, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_10 = t_10.view(t_8, -1, 32, 128)
        t_10 = t_10.transpose(1, 2)
        t_0 = t_0.transpose(3, 2)
        t_0 = torch.matmul(t_6, t_0)
        t_0 += t_1
        t_6 = t_0.float()
        t_6 = torch.nn.functional.softmax(t_6, dim=-1, _stacklevel=3, dtype=None)
        t_0 = t_6.type_as(t_0)
        t_0 = self.l_23(t_0)
        t_10 = torch.matmul(t_0, t_10)
        t_10 = t_10.transpose(1, 2)
        t_10 = t_10.contiguous()
        t_8 = t_10.view(t_8, -1, 4096)
        t_8 = self.l_24(t_8)
        t_10 = self.l_25(t_8)
        t_10 = t_9 + t_10
        t_1 = (t_8, None, t_1)
        t_10 = (t_10,)
        t_1 = t_1[slice(1, None, None)]
        t_1 = t_10 + t_1
        t_10 = t_1[slice(None, 2, None)]
        t_8 = t_10[0]
        t_9 = self.l_26(t_8)
        t_10 = t_10[1]
        t_1 = t_1[slice(2, None, None)]
        t_0 = self.l_27(t_9)
        t_9 = t_9.shape
        t_9 = t_9[slice(None, 2, None)]
        t_9 = t_9[0]
        t_0 = t_0.view(t_9, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_2 = t_2.view(t_9, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_3 = t_3.view(t_9, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_2 = t_2.transpose(3, 2)
        t_2 = torch.matmul(t_0, t_2)
        t_2 += t_7
        t_0 = t_2.float()
        t_0 = torch.nn.functional.softmax(t_0, dim=-1, _stacklevel=3, dtype=None)
        t_2 = t_0.type_as(t_2)
        t_2 = self.l_30(t_2)
        t_3 = torch.matmul(t_2, t_3)
        t_3 = t_3.transpose(1, 2)
        t_3 = t_3.contiguous()
        t_9 = t_3.view(t_9, -1, 4096)
        t_9 = self.l_31(t_9)
        t_3 = self.l_32(t_9)
        t_3 = t_8 + t_3
        t_7 = (t_9, None, t_7)
        t_3 = (t_3,)
        t_7 = t_7[slice(1, None, None)]
        t_7 = t_3 + t_7
        t_3 = t_7[0]
        t_9 = self.l_33(t_3)
        t_7 = t_7[slice(2, None, None)]
        t_7 = t_1 + t_7
        t_9 = self.l_34(t_9)
        t_9 = torch.nn.functional.relu(t_9, inplace=False)
        t_9 = self.l_35(t_9)
        t_9 = self.l_36(t_9)
        t_9 = self.l_37(t_9)
        t_9 = t_3 + t_9
        t_10 = (t_9, t_10)
        t_7 = t_10 + t_7
        t_10 = t_7[slice(None, 2, None)]
        t_10 = t_10[0]
        t_9 = self.l_38(t_10)
        t_3 = t_7[2]
        t_7 = t_7[3]
        t_1 = self.l_39(t_9)
        t_8 = self.l_40(t_9)
        t_2 = self.l_41(t_9)
        t_9 = t_9.shape
        t_9 = t_9[slice(None, 2, None)]
        t_9 = t_9[0]
        t_1 = t_1.view(t_9, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_8 = t_8.view(t_9, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_2 = t_2.view(t_9, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_8 = t_8.transpose(3, 2)
        t_8 = torch.matmul(t_1, t_8)
        t_8 += t_3
        t_1 = t_8.float()
        t_1 = torch.nn.functional.softmax(t_1, dim=-1, _stacklevel=3, dtype=None)
        t_8 = t_1.type_as(t_8)
        t_8 = self.l_42(t_8)
        t_2 = torch.matmul(t_8, t_2)
        t_2 = t_2.transpose(1, 2)
        t_2 = t_2.contiguous()
        t_9 = t_2.view(t_9, -1, 4096)
        t_9 = self.l_43(t_9)
        t_2 = self.l_44(t_9)
        t_2 = t_10 + t_2
        t_3 = (t_9, None, t_3)
        t_2 = (t_2,)
        t_3 = t_3[slice(1, None, None)]
        t_3 = t_2 + t_3
        t_2 = t_3[slice(None, 2, None)]
        t_9 = t_2[0]
        t_10 = self.l_45(t_9)
        t_2 = t_2[1]
        t_3 = t_3[slice(2, None, None)]
        t_8 = self.l_46(t_10)
        t_10 = t_10.shape
        t_10 = t_10[slice(None, 2, None)]
        t_10 = t_10[0]
        t_8 = t_8.view(t_10, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_4 = t_4.view(t_10, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_5 = t_5.view(t_10, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_8, t_4)
        t_4 += t_7
        t_8 = t_4.float()
        t_8 = torch.nn.functional.softmax(t_8, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_8.type_as(t_4)
        t_4 = self.l_49(t_4)
        t_5 = torch.matmul(t_4, t_5)
        t_5 = t_5.transpose(1, 2)
        t_5 = t_5.contiguous()
        t_10 = t_5.view(t_10, -1, 4096)
        t_10 = self.l_50(t_10)
        t_5 = self.l_51(t_10)
        t_5 = t_9 + t_5
        t_7 = (t_10, None, t_7)
        t_5 = (t_5,)
        t_7 = t_7[slice(1, None, None)]
        t_7 = t_5 + t_7
        t_5 = t_7[0]
        t_10 = self.l_52(t_5)
        t_7 = t_7[slice(2, None, None)]
        t_7 = t_3 + t_7
        t_10 = self.l_53(t_10)
        t_10 = torch.nn.functional.relu(t_10, inplace=False)
        t_10 = self.l_54(t_10)
        t_10 = self.l_55(t_10)
        t_10 = self.l_56(t_10)
        t_10 = t_5 + t_10
        t_2 = (t_10, t_2)
        t_7 = t_2 + t_7
        t_2 = t_7[slice(None, 2, None)]
        t_2 = t_2[0]
        t_10 = t_7[2]
        t_7 = t_7[3]
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___4897
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___4899
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___4901
        return list(flatten((x0, t_2, t_10, t_7)))

    def state_dict(self, *args, **kwargs):
        # we return the state dict of this part as it should be in the original model
        return state_dict(self, *args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return load_state_dict(self, *args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        # we return the named parameters of this part as it should be in the original model
        return named_parameters(self, *args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        # we return the named buffers of this part as it should be in the original model
        return named_buffers(self, *args, **kwargs)

    def cpu(self):
        return cpu(self)

    def cuda(self, device=None):
        return cuda(self, device=device)

    def to(self, *args, **kwargs):
        return to(self, *args, **kwargs)


class Partition14(nn.Module):
    LAYER_SCOPES = [
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout]',
        ]
    TENSORS = [
        ]
    def __init__(self, layers, tensors, device='cuda:14'):
        super().__init__()

        # Initialize partition layers
        for idx, layer_scope in enumerate(self.LAYER_SCOPES):
            self.add_module(f'l_{idx}' ,layers[layer_scope])

        # Initialize partition tensors (params and buffs)
        b = p = 0
        for tensor_scope in self.TENSORS:
            tensor = tensors[tensor_scope]
            if isinstance(tensor, nn.Parameter):
                self.register_parameter(f'p_{p}', tensor)
                p += 1
            else:
                self.register_buffer(f'b_{b}', tensor)
                b += 1

        self.device = torch.device(device)
        self.input_structure = [1, 1, 1, 1]
        self.lookup = {'l_0': 'decoder.block.18.layer.0.layer_norm',
                        'l_1': 'decoder.block.18.layer.0.SelfAttention.q',
                        'l_2': 'decoder.block.18.layer.0.SelfAttention.k',
                        'l_3': 'decoder.block.18.layer.0.SelfAttention.v',
                        'l_4': 'decoder.block.18.layer.0.SelfAttention.dropout',
                        'l_5': 'decoder.block.18.layer.0.SelfAttention.o',
                        'l_6': 'decoder.block.18.layer.0.dropout',
                        'l_7': 'decoder.block.18.layer.1.layer_norm',
                        'l_8': 'decoder.block.18.layer.1.EncDecAttention.q',
                        'l_9': 'decoder.block.18.layer.1.EncDecAttention.k',
                        'l_10': 'decoder.block.18.layer.1.EncDecAttention.v',
                        'l_11': 'decoder.block.18.layer.1.EncDecAttention.dropout',
                        'l_12': 'decoder.block.18.layer.1.EncDecAttention.o',
                        'l_13': 'decoder.block.18.layer.1.dropout',
                        'l_14': 'decoder.block.18.layer.2.layer_norm',
                        'l_15': 'decoder.block.18.layer.2.DenseReluDense.wi',
                        'l_16': 'decoder.block.18.layer.2.DenseReluDense.dropout',
                        'l_17': 'decoder.block.18.layer.2.DenseReluDense.wo',
                        'l_18': 'decoder.block.18.layer.2.dropout',
                        'l_19': 'decoder.block.19.layer.0.layer_norm',
                        'l_20': 'decoder.block.19.layer.0.SelfAttention.q',
                        'l_21': 'decoder.block.19.layer.0.SelfAttention.k',
                        'l_22': 'decoder.block.19.layer.0.SelfAttention.v',
                        'l_23': 'decoder.block.19.layer.0.SelfAttention.dropout',
                        'l_24': 'decoder.block.19.layer.0.SelfAttention.o',
                        'l_25': 'decoder.block.19.layer.0.dropout',
                        'l_26': 'decoder.block.19.layer.1.layer_norm',
                        'l_27': 'decoder.block.19.layer.1.EncDecAttention.q',
                        'l_28': 'decoder.block.19.layer.1.EncDecAttention.k',
                        'l_29': 'decoder.block.19.layer.1.EncDecAttention.v',
                        'l_30': 'decoder.block.19.layer.1.EncDecAttention.dropout',
                        'l_31': 'decoder.block.19.layer.1.EncDecAttention.o',
                        'l_32': 'decoder.block.19.layer.1.dropout',
                        'l_33': 'decoder.block.19.layer.2.layer_norm',
                        'l_34': 'decoder.block.19.layer.2.DenseReluDense.wi',
                        'l_35': 'decoder.block.19.layer.2.DenseReluDense.dropout',
                        'l_36': 'decoder.block.19.layer.2.DenseReluDense.wo',
                        'l_37': 'decoder.block.19.layer.2.dropout',
                        'l_38': 'decoder.block.20.layer.0.layer_norm',
                        'l_39': 'decoder.block.20.layer.0.SelfAttention.q',
                        'l_40': 'decoder.block.20.layer.0.SelfAttention.k',
                        'l_41': 'decoder.block.20.layer.0.SelfAttention.v',
                        'l_42': 'decoder.block.20.layer.0.SelfAttention.dropout',
                        'l_43': 'decoder.block.20.layer.0.SelfAttention.o',
                        'l_44': 'decoder.block.20.layer.0.dropout',
                        'l_45': 'decoder.block.20.layer.1.layer_norm',
                        'l_46': 'decoder.block.20.layer.1.EncDecAttention.q',
                        'l_47': 'decoder.block.20.layer.1.EncDecAttention.k',
                        'l_48': 'decoder.block.20.layer.1.EncDecAttention.v',
                        'l_49': 'decoder.block.20.layer.1.EncDecAttention.dropout',
                        'l_50': 'decoder.block.20.layer.1.EncDecAttention.o',
                        'l_51': 'decoder.block.20.layer.1.dropout',
                        'l_52': 'decoder.block.20.layer.2.layer_norm',
                        'l_53': 'decoder.block.20.layer.2.DenseReluDense.wi',
                        'l_54': 'decoder.block.20.layer.2.DenseReluDense.dropout',
                        'l_55': 'decoder.block.20.layer.2.DenseReluDense.wo',
                        'l_56': 'decoder.block.20.layer.2.dropout'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[18]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_36
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[19]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout] <=> self.l_37
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_38
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_39
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_40
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_41
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_42
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_43
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_44
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_45
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_46
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_47
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_48
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_49
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_50
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_51
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_52
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_53
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_54
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_55
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[20]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout] <=> self.l_56
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout] <=> x0
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___4897 <=> x1
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___4899 <=> x2
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___4901 <=> x3
        x0, x1, x2, x3 = unflatten(args, self.input_structure)
        t_0 = self.l_9(x0)
        t_1 = self.l_10(x0)
        t_2 = self.l_28(x0)
        t_3 = self.l_29(x0)
        t_4 = self.l_47(x0)
        t_5 = self.l_48(x0)
        t_6 = self.l_0(x1)
        t_7 = self.l_1(t_6)
        t_8 = self.l_2(t_6)
        t_9 = self.l_3(t_6)
        t_6 = t_6.shape
        t_6 = t_6[slice(None, 2, None)]
        t_6 = t_6[0]
        t_7 = t_7.view(t_6, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_8 = t_8.view(t_6, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_9 = t_9.view(t_6, -1, 32, 128)
        t_9 = t_9.transpose(1, 2)
        t_8 = t_8.transpose(3, 2)
        t_8 = torch.matmul(t_7, t_8)
        t_8 += x2
        t_7 = t_8.float()
        t_7 = torch.nn.functional.softmax(t_7, dim=-1, _stacklevel=3, dtype=None)
        t_8 = t_7.type_as(t_8)
        t_8 = self.l_4(t_8)
        t_9 = torch.matmul(t_8, t_9)
        t_9 = t_9.transpose(1, 2)
        t_9 = t_9.contiguous()
        t_6 = t_9.view(t_6, -1, 4096)
        t_6 = self.l_5(t_6)
        t_9 = self.l_6(t_6)
        t_9 = x1 + t_9
        t_6 = (t_6, None, x2)
        t_9 = (t_9,)
        t_6 = t_6[slice(1, None, None)]
        t_6 = t_9 + t_6
        t_9 = t_6[slice(None, 2, None)]
        t_8 = t_9[0]
        t_7 = self.l_7(t_8)
        t_9 = t_9[1]
        t_6 = t_6[slice(2, None, None)]
        t_10 = self.l_8(t_7)
        t_7 = t_7.shape
        t_7 = t_7[slice(None, 2, None)]
        t_7 = t_7[0]
        t_10 = t_10.view(t_7, -1, 32, 128)
        t_10 = t_10.transpose(1, 2)
        t_0 = t_0.view(t_7, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_1 = t_1.view(t_7, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_0 = t_0.transpose(3, 2)
        t_0 = torch.matmul(t_10, t_0)
        t_0 += x3
        t_10 = t_0.float()
        t_10 = torch.nn.functional.softmax(t_10, dim=-1, _stacklevel=3, dtype=None)
        t_0 = t_10.type_as(t_0)
        t_0 = self.l_11(t_0)
        t_1 = torch.matmul(t_0, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_7 = t_1.view(t_7, -1, 4096)
        t_7 = self.l_12(t_7)
        t_1 = self.l_13(t_7)
        t_1 = t_8 + t_1
        t_7 = (t_7, None, x3)
        t_1 = (t_1,)
        t_7 = t_7[slice(1, None, None)]
        t_7 = t_1 + t_7
        t_1 = t_7[0]
        t_8 = self.l_14(t_1)
        t_7 = t_7[slice(2, None, None)]
        t_7 = t_6 + t_7
        t_8 = self.l_15(t_8)
        t_8 = torch.nn.functional.relu(t_8, inplace=False)
        t_8 = self.l_16(t_8)
        t_8 = self.l_17(t_8)
        t_8 = self.l_18(t_8)
        t_8 = t_1 + t_8
        t_9 = (t_8, t_9)
        t_7 = t_9 + t_7
        t_9 = t_7[slice(None, 2, None)]
        t_9 = t_9[0]
        t_8 = self.l_19(t_9)
        t_1 = t_7[2]
        t_7 = t_7[3]
        t_6 = self.l_20(t_8)
        t_0 = self.l_21(t_8)
        t_10 = self.l_22(t_8)
        t_8 = t_8.shape
        t_8 = t_8[slice(None, 2, None)]
        t_8 = t_8[0]
        t_6 = t_6.view(t_8, -1, 32, 128)
        t_6 = t_6.transpose(1, 2)
        t_0 = t_0.view(t_8, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_10 = t_10.view(t_8, -1, 32, 128)
        t_10 = t_10.transpose(1, 2)
        t_0 = t_0.transpose(3, 2)
        t_0 = torch.matmul(t_6, t_0)
        t_0 += t_1
        t_6 = t_0.float()
        t_6 = torch.nn.functional.softmax(t_6, dim=-1, _stacklevel=3, dtype=None)
        t_0 = t_6.type_as(t_0)
        t_0 = self.l_23(t_0)
        t_10 = torch.matmul(t_0, t_10)
        t_10 = t_10.transpose(1, 2)
        t_10 = t_10.contiguous()
        t_8 = t_10.view(t_8, -1, 4096)
        t_8 = self.l_24(t_8)
        t_10 = self.l_25(t_8)
        t_10 = t_9 + t_10
        t_1 = (t_8, None, t_1)
        t_10 = (t_10,)
        t_1 = t_1[slice(1, None, None)]
        t_1 = t_10 + t_1
        t_10 = t_1[slice(None, 2, None)]
        t_8 = t_10[0]
        t_9 = self.l_26(t_8)
        t_10 = t_10[1]
        t_1 = t_1[slice(2, None, None)]
        t_0 = self.l_27(t_9)
        t_9 = t_9.shape
        t_9 = t_9[slice(None, 2, None)]
        t_9 = t_9[0]
        t_0 = t_0.view(t_9, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_2 = t_2.view(t_9, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_3 = t_3.view(t_9, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_2 = t_2.transpose(3, 2)
        t_2 = torch.matmul(t_0, t_2)
        t_2 += t_7
        t_0 = t_2.float()
        t_0 = torch.nn.functional.softmax(t_0, dim=-1, _stacklevel=3, dtype=None)
        t_2 = t_0.type_as(t_2)
        t_2 = self.l_30(t_2)
        t_3 = torch.matmul(t_2, t_3)
        t_3 = t_3.transpose(1, 2)
        t_3 = t_3.contiguous()
        t_9 = t_3.view(t_9, -1, 4096)
        t_9 = self.l_31(t_9)
        t_3 = self.l_32(t_9)
        t_3 = t_8 + t_3
        t_7 = (t_9, None, t_7)
        t_3 = (t_3,)
        t_7 = t_7[slice(1, None, None)]
        t_7 = t_3 + t_7
        t_3 = t_7[0]
        t_9 = self.l_33(t_3)
        t_7 = t_7[slice(2, None, None)]
        t_7 = t_1 + t_7
        t_9 = self.l_34(t_9)
        t_9 = torch.nn.functional.relu(t_9, inplace=False)
        t_9 = self.l_35(t_9)
        t_9 = self.l_36(t_9)
        t_9 = self.l_37(t_9)
        t_9 = t_3 + t_9
        t_10 = (t_9, t_10)
        t_7 = t_10 + t_7
        t_10 = t_7[slice(None, 2, None)]
        t_10 = t_10[0]
        t_9 = self.l_38(t_10)
        t_3 = t_7[2]
        t_7 = t_7[3]
        t_1 = self.l_39(t_9)
        t_8 = self.l_40(t_9)
        t_2 = self.l_41(t_9)
        t_9 = t_9.shape
        t_9 = t_9[slice(None, 2, None)]
        t_9 = t_9[0]
        t_1 = t_1.view(t_9, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_8 = t_8.view(t_9, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_2 = t_2.view(t_9, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_8 = t_8.transpose(3, 2)
        t_8 = torch.matmul(t_1, t_8)
        t_8 += t_3
        t_1 = t_8.float()
        t_1 = torch.nn.functional.softmax(t_1, dim=-1, _stacklevel=3, dtype=None)
        t_8 = t_1.type_as(t_8)
        t_8 = self.l_42(t_8)
        t_2 = torch.matmul(t_8, t_2)
        t_2 = t_2.transpose(1, 2)
        t_2 = t_2.contiguous()
        t_9 = t_2.view(t_9, -1, 4096)
        t_9 = self.l_43(t_9)
        t_2 = self.l_44(t_9)
        t_2 = t_10 + t_2
        t_3 = (t_9, None, t_3)
        t_2 = (t_2,)
        t_3 = t_3[slice(1, None, None)]
        t_3 = t_2 + t_3
        t_2 = t_3[slice(None, 2, None)]
        t_9 = t_2[0]
        t_10 = self.l_45(t_9)
        t_2 = t_2[1]
        t_3 = t_3[slice(2, None, None)]
        t_8 = self.l_46(t_10)
        t_10 = t_10.shape
        t_10 = t_10[slice(None, 2, None)]
        t_10 = t_10[0]
        t_8 = t_8.view(t_10, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_4 = t_4.view(t_10, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_5 = t_5.view(t_10, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_8, t_4)
        t_4 += t_7
        t_8 = t_4.float()
        t_8 = torch.nn.functional.softmax(t_8, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_8.type_as(t_4)
        t_4 = self.l_49(t_4)
        t_5 = torch.matmul(t_4, t_5)
        t_5 = t_5.transpose(1, 2)
        t_5 = t_5.contiguous()
        t_10 = t_5.view(t_10, -1, 4096)
        t_10 = self.l_50(t_10)
        t_5 = self.l_51(t_10)
        t_5 = t_9 + t_5
        t_7 = (t_10, None, t_7)
        t_5 = (t_5,)
        t_7 = t_7[slice(1, None, None)]
        t_7 = t_5 + t_7
        t_5 = t_7[0]
        t_10 = self.l_52(t_5)
        t_7 = t_7[slice(2, None, None)]
        t_7 = t_3 + t_7
        t_10 = self.l_53(t_10)
        t_10 = torch.nn.functional.relu(t_10, inplace=False)
        t_10 = self.l_54(t_10)
        t_10 = self.l_55(t_10)
        t_10 = self.l_56(t_10)
        t_10 = t_5 + t_10
        t_2 = (t_10, t_2)
        t_7 = t_2 + t_7
        t_2 = t_7[slice(None, 2, None)]
        t_2 = t_2[0]
        t_10 = t_7[2]
        t_7 = t_7[3]
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___5350
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___5352
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___5354
        return list(flatten((x0, t_2, t_10, t_7)))

    def state_dict(self, *args, **kwargs):
        # we return the state dict of this part as it should be in the original model
        return state_dict(self, *args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return load_state_dict(self, *args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        # we return the named parameters of this part as it should be in the original model
        return named_parameters(self, *args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        # we return the named buffers of this part as it should be in the original model
        return named_buffers(self, *args, **kwargs)

    def cpu(self):
        return cpu(self)

    def cuda(self, device=None):
        return cuda(self, device=device)

    def to(self, *args, **kwargs):
        return to(self, *args, **kwargs)


class Partition15(nn.Module):
    LAYER_SCOPES = [
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5LayerNorm[final_layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/Dropout[dropout]',
            'T5ForConditionalGeneration/Linear[lm_head]',
        ]
    TENSORS = [
        ]
    def __init__(self, layers, tensors, device='cuda:15'):
        super().__init__()

        # Initialize partition layers
        for idx, layer_scope in enumerate(self.LAYER_SCOPES):
            self.add_module(f'l_{idx}' ,layers[layer_scope])

        # Initialize partition tensors (params and buffs)
        b = p = 0
        for tensor_scope in self.TENSORS:
            tensor = tensors[tensor_scope]
            if isinstance(tensor, nn.Parameter):
                self.register_parameter(f'p_{p}', tensor)
                p += 1
            else:
                self.register_buffer(f'b_{b}', tensor)
                b += 1

        self.device = torch.device(device)
        self.input_structure = [1, 1, 1, 1, 1]
        self.lookup = {'l_0': 'decoder.block.21.layer.0.layer_norm',
                        'l_1': 'decoder.block.21.layer.0.SelfAttention.q',
                        'l_2': 'decoder.block.21.layer.0.SelfAttention.k',
                        'l_3': 'decoder.block.21.layer.0.SelfAttention.v',
                        'l_4': 'decoder.block.21.layer.0.SelfAttention.dropout',
                        'l_5': 'decoder.block.21.layer.0.SelfAttention.o',
                        'l_6': 'decoder.block.21.layer.0.dropout',
                        'l_7': 'decoder.block.21.layer.1.layer_norm',
                        'l_8': 'decoder.block.21.layer.1.EncDecAttention.q',
                        'l_9': 'decoder.block.21.layer.1.EncDecAttention.k',
                        'l_10': 'decoder.block.21.layer.1.EncDecAttention.v',
                        'l_11': 'decoder.block.21.layer.1.EncDecAttention.dropout',
                        'l_12': 'decoder.block.21.layer.1.EncDecAttention.o',
                        'l_13': 'decoder.block.21.layer.1.dropout',
                        'l_14': 'decoder.block.21.layer.2.layer_norm',
                        'l_15': 'decoder.block.21.layer.2.DenseReluDense.wi',
                        'l_16': 'decoder.block.21.layer.2.DenseReluDense.dropout',
                        'l_17': 'decoder.block.21.layer.2.DenseReluDense.wo',
                        'l_18': 'decoder.block.21.layer.2.dropout',
                        'l_19': 'decoder.block.22.layer.0.layer_norm',
                        'l_20': 'decoder.block.22.layer.0.SelfAttention.q',
                        'l_21': 'decoder.block.22.layer.0.SelfAttention.k',
                        'l_22': 'decoder.block.22.layer.0.SelfAttention.v',
                        'l_23': 'decoder.block.22.layer.0.SelfAttention.dropout',
                        'l_24': 'decoder.block.22.layer.0.SelfAttention.o',
                        'l_25': 'decoder.block.22.layer.0.dropout',
                        'l_26': 'decoder.block.22.layer.1.layer_norm',
                        'l_27': 'decoder.block.22.layer.1.EncDecAttention.q',
                        'l_28': 'decoder.block.22.layer.1.EncDecAttention.k',
                        'l_29': 'decoder.block.22.layer.1.EncDecAttention.v',
                        'l_30': 'decoder.block.22.layer.1.EncDecAttention.dropout',
                        'l_31': 'decoder.block.22.layer.1.EncDecAttention.o',
                        'l_32': 'decoder.block.22.layer.1.dropout',
                        'l_33': 'decoder.block.22.layer.2.layer_norm',
                        'l_34': 'decoder.block.22.layer.2.DenseReluDense.wi',
                        'l_35': 'decoder.block.22.layer.2.DenseReluDense.dropout',
                        'l_36': 'decoder.block.22.layer.2.DenseReluDense.wo',
                        'l_37': 'decoder.block.22.layer.2.dropout',
                        'l_38': 'decoder.block.23.layer.0.layer_norm',
                        'l_39': 'decoder.block.23.layer.0.SelfAttention.q',
                        'l_40': 'decoder.block.23.layer.0.SelfAttention.k',
                        'l_41': 'decoder.block.23.layer.0.SelfAttention.v',
                        'l_42': 'decoder.block.23.layer.0.SelfAttention.dropout',
                        'l_43': 'decoder.block.23.layer.0.SelfAttention.o',
                        'l_44': 'decoder.block.23.layer.0.dropout',
                        'l_45': 'decoder.block.23.layer.1.layer_norm',
                        'l_46': 'decoder.block.23.layer.1.EncDecAttention.q',
                        'l_47': 'decoder.block.23.layer.1.EncDecAttention.k',
                        'l_48': 'decoder.block.23.layer.1.EncDecAttention.v',
                        'l_49': 'decoder.block.23.layer.1.EncDecAttention.dropout',
                        'l_50': 'decoder.block.23.layer.1.EncDecAttention.o',
                        'l_51': 'decoder.block.23.layer.1.dropout',
                        'l_52': 'decoder.block.23.layer.2.layer_norm',
                        'l_53': 'decoder.block.23.layer.2.DenseReluDense.wi',
                        'l_54': 'decoder.block.23.layer.2.DenseReluDense.dropout',
                        'l_55': 'decoder.block.23.layer.2.DenseReluDense.wo',
                        'l_56': 'decoder.block.23.layer.2.dropout',
                        'l_57': 'decoder.final_layer_norm',
                        'l_58': 'decoder.dropout',
                        'l_59': 'lm_head'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[21]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_36
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[22]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout] <=> self.l_37
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_38
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_39
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_40
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_41
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_42
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_43
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_44
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_45
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_46
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_47
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_48
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_49
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_50
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_51
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_52
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_53
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_54
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_55
        # T5ForConditionalGeneration/T5Stack[decoder]/ModuleList[block]/T5Block[23]/ModuleList[layer]/T5LayerFF[2]/Dropout[dropout] <=> self.l_56
        # T5ForConditionalGeneration/T5Stack[decoder]/T5LayerNorm[final_layer_norm] <=> self.l_57
        # T5ForConditionalGeneration/T5Stack[decoder]/Dropout[dropout] <=> self.l_58
        # T5ForConditionalGeneration/Linear[lm_head] <=> self.l_59
        # input4 <=> labels
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout] <=> x0
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___5350 <=> x1
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___5352 <=> x2
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___5354 <=> x3
        labels, x0, x1, x2, x3 = unflatten(args, self.input_structure)
        t_0 = self.l_9(x0)
        t_1 = self.l_10(x0)
        t_2 = self.l_28(x0)
        t_3 = self.l_29(x0)
        t_4 = self.l_47(x0)
        t_5 = self.l_48(x0)
        t_6 = self.l_0(x1)
        t_7 = self.l_1(t_6)
        t_8 = self.l_2(t_6)
        t_9 = self.l_3(t_6)
        t_6 = t_6.shape
        t_6 = t_6[slice(None, 2, None)]
        t_6 = t_6[0]
        t_7 = t_7.view(t_6, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_8 = t_8.view(t_6, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_9 = t_9.view(t_6, -1, 32, 128)
        t_9 = t_9.transpose(1, 2)
        t_8 = t_8.transpose(3, 2)
        t_8 = torch.matmul(t_7, t_8)
        t_8 += x2
        t_7 = t_8.float()
        t_7 = torch.nn.functional.softmax(t_7, dim=-1, _stacklevel=3, dtype=None)
        t_8 = t_7.type_as(t_8)
        t_8 = self.l_4(t_8)
        t_9 = torch.matmul(t_8, t_9)
        t_9 = t_9.transpose(1, 2)
        t_9 = t_9.contiguous()
        t_6 = t_9.view(t_6, -1, 4096)
        t_6 = self.l_5(t_6)
        t_9 = self.l_6(t_6)
        t_9 = x1 + t_9
        t_6 = (t_6, None, x2)
        t_9 = (t_9,)
        t_6 = t_6[slice(1, None, None)]
        t_6 = t_9 + t_6
        t_9 = t_6[slice(None, 2, None)]
        t_8 = t_9[0]
        t_7 = self.l_7(t_8)
        t_9 = t_9[1]
        t_6 = t_6[slice(2, None, None)]
        t_10 = self.l_8(t_7)
        t_7 = t_7.shape
        t_7 = t_7[slice(None, 2, None)]
        t_7 = t_7[0]
        t_10 = t_10.view(t_7, -1, 32, 128)
        t_10 = t_10.transpose(1, 2)
        t_0 = t_0.view(t_7, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_1 = t_1.view(t_7, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_0 = t_0.transpose(3, 2)
        t_0 = torch.matmul(t_10, t_0)
        t_0 += x3
        t_10 = t_0.float()
        t_10 = torch.nn.functional.softmax(t_10, dim=-1, _stacklevel=3, dtype=None)
        t_0 = t_10.type_as(t_0)
        t_0 = self.l_11(t_0)
        t_1 = torch.matmul(t_0, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_7 = t_1.view(t_7, -1, 4096)
        t_7 = self.l_12(t_7)
        t_1 = self.l_13(t_7)
        t_1 = t_8 + t_1
        t_7 = (t_7, None, x3)
        t_1 = (t_1,)
        t_7 = t_7[slice(1, None, None)]
        t_7 = t_1 + t_7
        t_1 = t_7[0]
        t_8 = self.l_14(t_1)
        t_7 = t_7[slice(2, None, None)]
        t_7 = t_6 + t_7
        t_8 = self.l_15(t_8)
        t_8 = torch.nn.functional.relu(t_8, inplace=False)
        t_8 = self.l_16(t_8)
        t_8 = self.l_17(t_8)
        t_8 = self.l_18(t_8)
        t_8 = t_1 + t_8
        t_9 = (t_8, t_9)
        t_7 = t_9 + t_7
        t_9 = t_7[slice(None, 2, None)]
        t_9 = t_9[0]
        t_8 = self.l_19(t_9)
        t_1 = t_7[2]
        t_7 = t_7[3]
        t_6 = self.l_20(t_8)
        t_0 = self.l_21(t_8)
        t_10 = self.l_22(t_8)
        t_8 = t_8.shape
        t_8 = t_8[slice(None, 2, None)]
        t_8 = t_8[0]
        t_6 = t_6.view(t_8, -1, 32, 128)
        t_6 = t_6.transpose(1, 2)
        t_0 = t_0.view(t_8, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_10 = t_10.view(t_8, -1, 32, 128)
        t_10 = t_10.transpose(1, 2)
        t_0 = t_0.transpose(3, 2)
        t_0 = torch.matmul(t_6, t_0)
        t_0 += t_1
        t_6 = t_0.float()
        t_6 = torch.nn.functional.softmax(t_6, dim=-1, _stacklevel=3, dtype=None)
        t_0 = t_6.type_as(t_0)
        t_0 = self.l_23(t_0)
        t_10 = torch.matmul(t_0, t_10)
        t_10 = t_10.transpose(1, 2)
        t_10 = t_10.contiguous()
        t_8 = t_10.view(t_8, -1, 4096)
        t_8 = self.l_24(t_8)
        t_10 = self.l_25(t_8)
        t_10 = t_9 + t_10
        t_1 = (t_8, None, t_1)
        t_10 = (t_10,)
        t_1 = t_1[slice(1, None, None)]
        t_1 = t_10 + t_1
        t_10 = t_1[slice(None, 2, None)]
        t_8 = t_10[0]
        t_9 = self.l_26(t_8)
        t_10 = t_10[1]
        t_1 = t_1[slice(2, None, None)]
        t_0 = self.l_27(t_9)
        t_9 = t_9.shape
        t_9 = t_9[slice(None, 2, None)]
        t_9 = t_9[0]
        t_0 = t_0.view(t_9, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_2 = t_2.view(t_9, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_3 = t_3.view(t_9, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_2 = t_2.transpose(3, 2)
        t_2 = torch.matmul(t_0, t_2)
        t_2 += t_7
        t_0 = t_2.float()
        t_0 = torch.nn.functional.softmax(t_0, dim=-1, _stacklevel=3, dtype=None)
        t_2 = t_0.type_as(t_2)
        t_2 = self.l_30(t_2)
        t_3 = torch.matmul(t_2, t_3)
        t_3 = t_3.transpose(1, 2)
        t_3 = t_3.contiguous()
        t_9 = t_3.view(t_9, -1, 4096)
        t_9 = self.l_31(t_9)
        t_3 = self.l_32(t_9)
        t_3 = t_8 + t_3
        t_7 = (t_9, None, t_7)
        t_3 = (t_3,)
        t_7 = t_7[slice(1, None, None)]
        t_7 = t_3 + t_7
        t_3 = t_7[0]
        t_9 = self.l_33(t_3)
        t_7 = t_7[slice(2, None, None)]
        t_7 = t_1 + t_7
        t_9 = self.l_34(t_9)
        t_9 = torch.nn.functional.relu(t_9, inplace=False)
        t_9 = self.l_35(t_9)
        t_9 = self.l_36(t_9)
        t_9 = self.l_37(t_9)
        t_9 = t_3 + t_9
        t_10 = (t_9, t_10)
        t_7 = t_10 + t_7
        t_10 = t_7[slice(None, 2, None)]
        t_10 = t_10[0]
        t_9 = self.l_38(t_10)
        t_3 = t_7[2]
        t_7 = t_7[3]
        t_1 = self.l_39(t_9)
        t_8 = self.l_40(t_9)
        t_2 = self.l_41(t_9)
        t_9 = t_9.shape
        t_9 = t_9[slice(None, 2, None)]
        t_9 = t_9[0]
        t_1 = t_1.view(t_9, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_8 = t_8.view(t_9, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_2 = t_2.view(t_9, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_8 = t_8.transpose(3, 2)
        t_8 = torch.matmul(t_1, t_8)
        t_8 += t_3
        t_1 = t_8.float()
        t_1 = torch.nn.functional.softmax(t_1, dim=-1, _stacklevel=3, dtype=None)
        t_8 = t_1.type_as(t_8)
        t_8 = self.l_42(t_8)
        t_2 = torch.matmul(t_8, t_2)
        t_2 = t_2.transpose(1, 2)
        t_2 = t_2.contiguous()
        t_9 = t_2.view(t_9, -1, 4096)
        t_9 = self.l_43(t_9)
        t_2 = self.l_44(t_9)
        t_2 = t_10 + t_2
        t_3 = (t_9, None, t_3)
        t_2 = (t_2,)
        t_3 = t_3[slice(1, None, None)]
        t_3 = t_2 + t_3
        t_2 = t_3[slice(None, 2, None)]
        t_9 = t_2[0]
        t_10 = self.l_45(t_9)
        t_2 = t_2[1]
        t_3 = t_3[slice(2, None, None)]
        t_8 = self.l_46(t_10)
        t_10 = t_10.shape
        t_10 = t_10[slice(None, 2, None)]
        t_10 = t_10[0]
        t_8 = t_8.view(t_10, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_4 = t_4.view(t_10, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_5 = t_5.view(t_10, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_8, t_4)
        t_4 += t_7
        t_8 = t_4.float()
        t_8 = torch.nn.functional.softmax(t_8, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_8.type_as(t_4)
        t_4 = self.l_49(t_4)
        t_5 = torch.matmul(t_4, t_5)
        t_5 = t_5.transpose(1, 2)
        t_5 = t_5.contiguous()
        t_10 = t_5.view(t_10, -1, 4096)
        t_10 = self.l_50(t_10)
        t_5 = self.l_51(t_10)
        t_5 = t_9 + t_5
        t_7 = (t_10, None, t_7)
        t_5 = (t_5,)
        t_7 = t_7[slice(1, None, None)]
        t_7 = t_5 + t_7
        t_5 = t_7[0]
        t_10 = self.l_52(t_5)
        t_7 = t_7[slice(2, None, None)]
        t_7 = t_3 + t_7
        t_10 = self.l_53(t_10)
        t_10 = torch.nn.functional.relu(t_10, inplace=False)
        t_10 = self.l_54(t_10)
        t_10 = self.l_55(t_10)
        t_10 = self.l_56(t_10)
        t_10 = t_5 + t_10
        t_2 = (t_10, t_2)
        t_7 = t_2 + t_7
        t_7 = t_7[slice(None, 2, None)]
        t_7 = t_7[0]
        t_7 = self.l_57(t_7)
        t_7 = self.l_58(t_7)
        t_7 = t_7 * 0.03125
        t_7 = self.l_59(t_7)
        t_2 = t_7.size(-1)
        t_2 = t_7.view(-1, t_2)
        t_7 = labels.view(-1)
        t_7 = torch.nn.functional.cross_entropy(t_2, t_7, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
        # Returning:
        # T5ForConditionalGeneration/torch.nn.functional::cross_entropy_5820
        return (t_7,)

    def state_dict(self, *args, **kwargs):
        # we return the state dict of this part as it should be in the original model
        return state_dict(self, *args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return load_state_dict(self, *args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        # we return the named parameters of this part as it should be in the original model
        return named_parameters(self, *args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        # we return the named buffers of this part as it should be in the original model
        return named_buffers(self, *args, **kwargs)

    def cpu(self):
        return cpu(self)

    def cuda(self, device=None):
        return cuda(self, device=device)

    def to(self, *args, **kwargs):
        return to(self, *args, **kwargs)


def traverse_model(module: nn.Module, depth: int, prefix: Optional[str] = None,
                   basic_blocks: Tuple[Type[nn.Module]] = (), full: bool = False) -> Iterator[
    Tuple[nn.Module, str, nn.Module, Optional[bool]]]:
    """
    iterate over model layers yielding the layer,layer_scope,encasing_module
    Parameters:
    -----------
    model:
        the model to iterate over
    depth:
        how far down in the model tree to go
    basic_blocks:
        a list of modules that if encountered will not be broken down
    full:
        whether to yield only layers specified by the depth and basic_block options or to yield all layers
    """
    if prefix is None:
        prefix = type(module).__name__

    for name, sub_module in module.named_children():
        scope = prefix + "/" + type(sub_module).__name__ + f"[{name}]"
        if len(list(sub_module.children())) == 0 or isinstance(sub_module, tuple(basic_blocks)) or depth == 0:
            if full:
                # TODO:
                # is_explicit_block_limit = len(list(sub_module.children())) != 0 and (isinstance(sub_module, tuple(basic_blocks)) or depth == 0)
                yield sub_module, scope, module, True

            else:
                yield sub_module, scope, module
        else:
            if full:
                yield sub_module, scope, module, False
            yield from traverse_model(sub_module, depth - 1, scope, basic_blocks, full)


def layerDict(model: nn.Module, depth=1000, basic_blocks=()) -> Dict[str, nn.Module]:
    return {s: l for l, s, _ in traverse_model(model, depth, basic_blocks=basic_blocks)}


def traverse_params_buffs(module: nn.Module, prefix: Optional[str] = None) -> Iterator[Tuple[torch.tensor, str]]:
    """
    iterate over model's buffers and parameters yielding obj,obj_scope

    Parameters:
    -----------
    model:
        the model to iterate over
    """
    if prefix is None:
        prefix = type(module).__name__

    # params
    for param_name, param in module.named_parameters(recurse=False):
        param_scope = f"{prefix}/{type(param).__name__}[{param_name}]"
        yield param, param_scope

    # buffs
    for buffer_name, buffer in module.named_buffers(recurse=False):
        buffer_scope = f"{prefix}/{type(buffer).__name__}[{buffer_name}]"
        yield buffer, buffer_scope

    # recurse
    for name, sub_module in module.named_children():
        yield from traverse_params_buffs(sub_module, prefix + "/" + type(sub_module).__name__ + f"[{name}]")


def tensorDict(model: nn.Module) -> OrderedDict[str, Tensor]:
    return collections.OrderedDict((s, t) for t, s in traverse_params_buffs(model))


def move_tensors(ts, device):
    def move(t):
        if isinstance(t, (nn.Module, Tensor)):
            return t.to(device)
        return t

    return nested_map(move, ts)


def nested_map(func, ts, full=False):
    if isinstance(ts, torch.Size):
        # size is inheriting from tuple which is stupid
        return func(ts)
    elif isinstance(ts, (list, tuple, set)):
        return type(ts)(nested_map(func, t, full=full) for t in ts)
    elif isinstance(ts, dict):
        return {k: nested_map(func, v, full=full) for k, v in ts.items()}
    elif isinstance(ts, slice) and full:
        start = nested_map(func, ts.start, full=full)
        stop = nested_map(func, ts.stop, full=full)
        step = nested_map(func, ts.step, full=full)
        return slice(start, stop, step)
    return func(ts)


def flatten(ts):
    if isinstance(ts, torch.Size):
        # size is inheriting from tuple which is stupid
        yield ts
    elif isinstance(ts, (list, tuple, set)):
        yield from chain(*[flatten(t) for t in ts])
    elif isinstance(ts, dict):
        yield from chain(*[flatten(t) for k, t in sorted(ts.items(), key=lambda t: t[0])])
    else:
        yield ts


def unflatten(xs, structure):
    return _unflatten(xs, structure)[0]


def _unflatten(xs, structure):
    if isinstance(structure, torch.Size):
        # torch.Size is subclass of tuple which is stupid
        return xs[0], 1

    if not isinstance(structure, (list, tuple, set, dict)):
        return xs[0], 1

    if isinstance(structure, (list, tuple, set)):
        offset = 0
        elements = []
        for s in structure:
            e, n = _unflatten(xs[offset:], s)
            elements.append(e)
            offset += n

        return type(structure)(elements), offset

    assert isinstance(structure, dict)
    offset = 0
    elements = dict()
    for k, v in sorted(structure.items(), key=lambda t: t[0]):
        e, n = _unflatten(xs[offset:], v)
        elements[k] = e
        offset += n

    return elements, offset


def state_dict(partition, *args, **kwargs):
    # we return the state dict of this part as it should be in the original model
    state = nn.Module.state_dict(partition, *args, **kwargs)
    lookup = partition.lookup
    result = dict()
    for k, v in state.items():
        if k in lookup:
            result[lookup[k]] = v
        else:
            assert '.' in k
            split_idx = k.find('.')
            new_k = lookup[k[:split_idx]] + k[split_idx:]
            result[new_k] = v
    return result


def load_state_dict(partition, state_dict, strict=True):
    reverse_lookup = {v: k for k, v in partition.lookup.items()}
    device = partition.device
    keys = list(partition.state_dict(None).keys())
    new_state = dict()
    for k in keys:
        if k in reverse_lookup:
            new_state[reverse_lookup[k]] = state_dict[k].to(device)
            continue
        idx = k.rfind(".")
        to_replace = k[:idx]
        if to_replace in reverse_lookup:
            key = reverse_lookup[to_replace] + k[idx:]
            new_state[key] = state_dict[k].to(device)
    nn.Module.load_state_dict(partition, new_state, strict=strict)


def named_buffers(partition, prefix='', recurse=True):
    # we return the named buffers of this part as it should be in the original model
    params = nn.Module.named_buffers(partition, prefix=prefix, recurse=recurse)
    lookup = partition.lookup
    for k, v in params:
        if k in lookup:
            yield lookup[k], v
        else:
            assert '.' in k
            split_idx = k.find('.')
            new_k = lookup[k[:split_idx]] + k[split_idx:]
            yield new_k, v


def named_parameters(partition, prefix='', recurse=True):
    # we return the named parameters of this part as it should be in the original model
    params = nn.Module.named_parameters(partition, prefix=prefix, recurse=recurse)
    lookup = partition.lookup
    for k, v in params:
        if k in lookup:
            yield lookup[k], v
        else:
            assert '.' in k
            split_idx = k.find('.')
            new_k = lookup[k[:split_idx]] + k[split_idx:]
            yield new_k, v


def cpu(partition):
    partition.device = torch.device('cpu')
    return nn.Module.cpu(partition)


def cuda(partition, device=None):
    if device is None:
        device = torch.cuda.current_device()
    partition.device = torch.device(device)
    return nn.Module.cuda(partition, partition.device)


def to(partition, *args, **kwargs):
    device = None
    if 'device' in kwargs:
        device = kwargs['device']
    elif 'tensor' in kwargs:
        device = kwargs['tensor'].device
    if args:
        if isinstance(args[0], (torch.device, int, str)):
            device = args[0]
        if torch.is_tensor(args[0]):
            device = args[0].device
    if not (device is None):
        partition.device = torch.device(device)
    return nn.Module.to(partition, *args, **kwargs)

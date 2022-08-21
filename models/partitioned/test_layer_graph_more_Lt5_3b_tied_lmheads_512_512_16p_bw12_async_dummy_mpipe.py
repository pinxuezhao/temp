import math
import torch
import torch.nn.functional
import torch.functional
import torch.fft
import torch.linalg
from torch import Tensor
import torch.nn as nn
from itertools import chain
from typing import Optional, Tuple, Iterator, Iterable, OrderedDict, Dict
import collections

from typing import Type
from models.normal.NLP_models.stateless import StatelessEmbedding
from torch.nn.modules.loss import CrossEntropyLoss
from models.normal.NLP_models.modeling_t5 import T5LayerNorm
from torch.nn.modules.sparse import Embedding
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
# this is an auto generated file do not edit unless you know what you are doing


# partition adjacency
# model inputs {0, 15, 7}
# partition 0 {'inputs': {'input_ids', 'decoder_input_ids', 'attention_mask'}, 'outputs': {1, 2, 3, 4, 5, 6, 7}}
# partition 1 {'inputs': {0}, 'outputs': {2}}
# partition 2 {'inputs': {0, 1}, 'outputs': {3}}
# partition 3 {'inputs': {0, 2}, 'outputs': {4}}
# partition 4 {'inputs': {0, 3}, 'outputs': {5}}
# partition 5 {'inputs': {0, 4}, 'outputs': {6}}
# partition 6 {'inputs': {0, 5}, 'outputs': {7}}
# partition 7 {'inputs': {'inverted_encoder_attention_mask', 'decoder_attention_mask', 0, 6}, 'outputs': {8, 9, 10, 11, 12, 13, 14, 15}}
# partition 8 {'inputs': {7}, 'outputs': {9}}
# partition 9 {'inputs': {8, 7}, 'outputs': {10}}
# partition 10 {'inputs': {9, 7}, 'outputs': {11}}
# partition 11 {'inputs': {10, 7}, 'outputs': {12}}
# partition 12 {'inputs': {11, 7}, 'outputs': {13}}
# partition 13 {'inputs': {12, 7}, 'outputs': {14}}
# partition 14 {'inputs': {13, 7}, 'outputs': {15}}
# partition 15 {'inputs': {'lm_labels', 14, 7}, 'outputs': {'output'}}
# model outputs {15}


def create_pipeline_configuration(DEBUG=False, batch_size=16):
    config = {
        'batch_dim': 0,
        'depth': 10000,
        'basic_blocks': (StatelessEmbedding,CrossEntropyLoss,T5LayerNorm,Embedding,Linear,Dropout),
        'model_inputs': {
            'attention_mask': {
                'shape': torch.Size([16, 1, 1, 70]),
                'dtype': torch.float32,
                'is_batched': True,
                'used_by': [0]},
            'decoder_attention_mask': {
                'shape': torch.Size([16, 1, 70, 70]),
                'dtype': torch.float32,
                'is_batched': True,
                'used_by': [7]},
            'decoder_input_ids': {
                'shape': torch.Size([16, 70]),
                'dtype': torch.int64,
                'is_batched': True,
                'used_by': [0]},
            'input_ids': {
                'shape': torch.Size([16, 70]),
                'dtype': torch.int64,
                'is_batched': True,
                'used_by': [0]},
            'inverted_encoder_attention_mask': {
                'shape': torch.Size([16, 1, 1, 70]),
                'dtype': torch.float32,
                'is_batched': True,
                'used_by': [7]},
            'lm_labels': {
                'shape': torch.Size([16, 70]),
                'dtype': torch.int64,
                'is_batched': True,
                'used_by': [15]}},
        'model_outputs': {
            'T5ForConditionalGeneration/CrossEntropyLoss[lm_loss]': {
                'shape': torch.Size([1]),
                'dtype': torch.float32,
                'is_batched': False,
                'created_by': 15}},
        'stages': {
            0: {
                'stage_cls': Partition0,
                'inputs': {
                    'attention_mask': {
                        'shape': torch.Size([16, 1, 1, 70]),
                        'dtype': torch.float32,
                        'req_grad': False,
                        'is_batched': True,
                        'created_by': -1},
                    'decoder_input_ids': {
                        'shape': torch.Size([16, 70]),
                        'dtype': torch.int64,
                        'req_grad': False,
                        'is_batched': True,
                        'created_by': -1},
                    'input_ids': {
                        'shape': torch.Size([16, 70]),
                        'dtype': torch.int64,
                        'req_grad': False,
                        'is_batched': True,
                        'created_by': -1}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___94_1': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [1]},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerSelfAttention[0]/Tensor::__add___290': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [1]},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerFF[1]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [1]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/StatelessEmbedding[embed_tokens]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [7]}},
                'devices': ['cpu' if DEBUG else 'cuda:0'],
                'stage_depth': 15},
            1: {
                'stage_cls': Partition1,
                'inputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___94_1': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 0},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerSelfAttention[0]/Tensor::__add___290': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 0},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerFF[1]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 0}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___94_2': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [2]},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerSelfAttention[0]/Tensor::__add___467': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [2]},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerFF[1]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [2]}},
                'devices': ['cpu' if DEBUG else 'cuda:1'],
                'stage_depth': 14},
            2: {
                'stage_cls': Partition2,
                'inputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___94_2': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 1},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerSelfAttention[0]/Tensor::__add___467': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 1},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerFF[1]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 1}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___94_3': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [3]},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerSelfAttention[0]/Tensor::__add___644': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [3]},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerFF[1]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [3]}},
                'devices': ['cpu' if DEBUG else 'cuda:2'],
                'stage_depth': 13},
            3: {
                'stage_cls': Partition3,
                'inputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___94_3': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 2},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerSelfAttention[0]/Tensor::__add___644': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 2},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerFF[1]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 2}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___94_4': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [4]},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerSelfAttention[0]/Tensor::__add___821': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [4]},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerFF[1]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [4]}},
                'devices': ['cpu' if DEBUG else 'cuda:3'],
                'stage_depth': 12},
            4: {
                'stage_cls': Partition4,
                'inputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___94_4': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 3},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerSelfAttention[0]/Tensor::__add___821': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 3},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerFF[1]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 3}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___94_5': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [5]},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerSelfAttention[0]/Tensor::__add___998': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [5]},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerFF[1]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [5]}},
                'devices': ['cpu' if DEBUG else 'cuda:4'],
                'stage_depth': 11},
            5: {
                'stage_cls': Partition5,
                'inputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___94_5': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 4},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerSelfAttention[0]/Tensor::__add___998': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 4},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerFF[1]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 4}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___94_6': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [6]},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerSelfAttention[0]/Tensor::__add___1175': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [6]},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerFF[1]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [6]}},
                'devices': ['cpu' if DEBUG else 'cuda:5'],
                'stage_depth': 10},
            6: {
                'stage_cls': Partition6,
                'inputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___94_6': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 5},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerSelfAttention[0]/Tensor::__add___1175': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 5},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerFF[1]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 5}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___94_7': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [7]},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerSelfAttention[0]/Tensor::__add___1352': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [7]},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerFF[1]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [7]}},
                'devices': ['cpu' if DEBUG else 'cuda:6'],
                'stage_depth': 9},
            7: {
                'stage_cls': Partition7,
                'inputs': {
                    'decoder_attention_mask': {
                        'shape': torch.Size([16, 1, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': False,
                        'is_batched': True,
                        'created_by': -1},
                    'inverted_encoder_attention_mask': {
                        'shape': torch.Size([16, 1, 1, 70]),
                        'dtype': torch.float32,
                        'req_grad': False,
                        'is_batched': True,
                        'created_by': -1},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___94_7': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 6},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerSelfAttention[0]/Tensor::__add___1352': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 6},
                    'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerFF[1]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 6},
                    'T5ForConditionalGeneration/T5Stack[decoder]/StatelessEmbedding[embed_tokens]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 0}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_8': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [8]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562_8': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [8]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658_8': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [8]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/Tensor::__add___1677': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [8]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerFF[2]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [8]}},
                'devices': ['cpu' if DEBUG else 'cuda:7'],
                'stage_depth': 8},
            8: {
                'stage_cls': Partition8,
                'inputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_8': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 7},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562_8': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 7},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658_8': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 7},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/Tensor::__add___1677': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 7},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerFF[2]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 7}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_9': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [9]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562_9': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [9]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658_9': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [9]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerCrossAttention[1]/Tensor::__add___2007': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [9]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerFF[2]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [9]}},
                'devices': ['cpu' if DEBUG else 'cuda:8'],
                'stage_depth': 7},
            9: {
                'stage_cls': Partition9,
                'inputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_9': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 8},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562_9': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 8},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658_9': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 8},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerCrossAttention[1]/Tensor::__add___2007': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 8},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerFF[2]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 8}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_10': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [10]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562_10': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [10]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658_10': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [10]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerCrossAttention[1]/Tensor::__add___2337': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [10]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerFF[2]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [10]}},
                'devices': ['cpu' if DEBUG else 'cuda:9'],
                'stage_depth': 6},
            10: {
                'stage_cls': Partition10,
                'inputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_10': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 9},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562_10': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 9},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658_10': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 9},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerCrossAttention[1]/Tensor::__add___2337': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 9},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerFF[2]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 9}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_11': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [11]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562_11': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [11]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658_11': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [11]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerCrossAttention[1]/Tensor::__add___2667': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [11]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerFF[2]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [11]}},
                'devices': ['cpu' if DEBUG else 'cuda:10'],
                'stage_depth': 5},
            11: {
                'stage_cls': Partition11,
                'inputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_11': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 10},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562_11': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 10},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658_11': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 10},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerCrossAttention[1]/Tensor::__add___2667': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 10},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerFF[2]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 10}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_12': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [12]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562_12': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [12]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658_12': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [12]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerCrossAttention[1]/Tensor::__add___2997': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [12]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerFF[2]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [12]}},
                'devices': ['cpu' if DEBUG else 'cuda:11'],
                'stage_depth': 4},
            12: {
                'stage_cls': Partition12,
                'inputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_12': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 11},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562_12': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 11},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658_12': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 11},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerCrossAttention[1]/Tensor::__add___2997': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 11},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerFF[2]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 11}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_13': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [13]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562_13': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [13]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658_13': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [13]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerCrossAttention[1]/Tensor::__add___3327': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [13]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerFF[2]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [13]}},
                'devices': ['cpu' if DEBUG else 'cuda:12'],
                'stage_depth': 3},
            13: {
                'stage_cls': Partition13,
                'inputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_13': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 12},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562_13': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 12},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658_13': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 12},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerCrossAttention[1]/Tensor::__add___3327': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 12},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerFF[2]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 12}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_14': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [14]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562_14': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [14]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658_14': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [14]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerCrossAttention[1]/Tensor::__add___3657': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [14]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerFF[2]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [14]}},
                'devices': ['cpu' if DEBUG else 'cuda:13'],
                'stage_depth': 2},
            14: {
                'stage_cls': Partition14,
                'inputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_14': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 13},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562_14': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 13},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658_14': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 13},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerCrossAttention[1]/Tensor::__add___3657': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 13},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerFF[2]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 13}},
                'outputs': {
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_15': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [15]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562_15': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [15]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658_15': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [15]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerCrossAttention[1]/Tensor::__add___3987': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [15]},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerFF[2]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'used_by': [15]}},
                'devices': ['cpu' if DEBUG else 'cuda:14'],
                'stage_depth': 1},
            15: {
                'stage_cls': Partition15,
                'inputs': {
                    'lm_labels': {
                        'shape': torch.Size([16, 70]),
                        'dtype': torch.int64,
                        'req_grad': False,
                        'is_batched': True,
                        'created_by': -1},
                    'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]_15': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 14},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562_15': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 14},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658_15': {
                        'shape': torch.Size([16, 32, 70, 70]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 14},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerCrossAttention[1]/Tensor::__add___3987': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 14},
                    'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerFF[2]/T5LayerNorm[layer_norm]': {
                        'shape': torch.Size([16, 70, 1024]),
                        'dtype': torch.float32,
                        'req_grad': True,
                        'is_batched': True,
                        'created_by': 14}},
                'outputs': {
                    'T5ForConditionalGeneration/CrossEntropyLoss[lm_loss]': {
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
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Embedding[relative_attention_bias]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/StatelessEmbedding[embed_tokens]',
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
        self.input_structure = [1, 1, 1]
        self.lookup = {'l_0': 'encoder.embed_tokens',
                        'l_1': 'encoder.dropout',
                        'l_2': 'encoder.0.0.layer_norm',
                        'l_3': 'encoder.0.0.SelfAttention.q',
                        'l_4': 'encoder.0.0.SelfAttention.k',
                        'l_5': 'encoder.0.0.SelfAttention.v',
                        'l_6': 'encoder.0.0.SelfAttention.relative_attention_bias',
                        'l_7': 'encoder.0.0.SelfAttention.dropout',
                        'l_8': 'encoder.0.0.SelfAttention.o',
                        'l_9': 'encoder.0.0.dropout',
                        'l_10': 'encoder.0.1.layer_norm',
                        'l_11': 'encoder.0.1.DenseReluDense.wi',
                        'l_12': 'encoder.0.1.DenseReluDense.dropout',
                        'l_13': 'encoder.0.1.DenseReluDense.wo',
                        'l_14': 'encoder.0.1.dropout',
                        'l_15': 'encoder.1.0.layer_norm',
                        'l_16': 'encoder.1.0.SelfAttention.q',
                        'l_17': 'encoder.1.0.SelfAttention.k',
                        'l_18': 'encoder.1.0.SelfAttention.v',
                        'l_19': 'encoder.1.0.SelfAttention.dropout',
                        'l_20': 'encoder.1.0.SelfAttention.o',
                        'l_21': 'encoder.1.0.dropout',
                        'l_22': 'encoder.1.1.layer_norm',
                        'l_23': 'encoder.1.1.DenseReluDense.wi',
                        'l_24': 'encoder.1.1.DenseReluDense.dropout',
                        'l_25': 'encoder.1.1.DenseReluDense.wo',
                        'l_26': 'encoder.1.1.dropout',
                        'l_27': 'encoder.2.0.layer_norm',
                        'l_28': 'encoder.2.0.SelfAttention.q',
                        'l_29': 'encoder.2.0.SelfAttention.k',
                        'l_30': 'encoder.2.0.SelfAttention.v',
                        'l_31': 'encoder.2.0.SelfAttention.dropout',
                        'l_32': 'encoder.2.0.SelfAttention.o',
                        'l_33': 'encoder.2.0.dropout',
                        'l_34': 'encoder.2.1.layer_norm',
                        'l_35': 'encoder.2.1.DenseReluDense.wi',
                        'l_36': 'encoder.2.1.DenseReluDense.dropout',
                        'l_37': 'encoder.2.1.DenseReluDense.wo',
                        'l_38': 'encoder.2.1.dropout',
                        'l_39': 'encoder.3.0.layer_norm',
                        'l_40': 'encoder.3.0.SelfAttention.q',
                        'l_41': 'encoder.3.0.SelfAttention.k',
                        'l_42': 'encoder.3.0.SelfAttention.v',
                        'l_43': 'encoder.3.0.SelfAttention.dropout',
                        'l_44': 'encoder.3.0.SelfAttention.o',
                        'l_45': 'encoder.3.0.dropout',
                        'l_46': 'encoder.3.1.layer_norm',
                        'l_47': 'decoder.embed_tokens',
                        'p_0': 'shared_embed_weight'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[encoder]/StatelessEmbedding[embed_tokens] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Embedding[relative_attention_bias] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerFF[1]/Dropout[dropout] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerFF[1]/Dropout[dropout] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_36
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_37
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerFF[1]/Dropout[dropout] <=> self.l_38
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_39
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_40
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_41
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_42
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_43
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_44
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_45
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_46
        # T5ForConditionalGeneration/T5Stack[decoder]/StatelessEmbedding[embed_tokens] <=> self.l_47
        # T5ForConditionalGeneration/Parameter[shared_embed_weight] <=> self.p_0
        # input0 <=> attention_mask
        # input2 <=> decoder_input_ids
        # input3 <=> input_ids
        attention_mask, decoder_input_ids, input_ids = unflatten(args, self.input_structure)
        t_0 = decoder_input_ids.size()
        t_1 = input_ids.size()
        t_1 = t_1[-1]
        t_1 = input_ids.view(-1, t_1)
        t_1 = self.l_0(self.p_0, t_1)
        t_1 = self.l_1(t_1)
        t_2 = self.l_2(t_1)
        t_3 = t_2.size()
        t_4 = self.l_3(t_2)
        t_5 = self.l_4(t_2)
        t_2 = self.l_5(t_2)
        t_6 = t_3[0]
        t_3 = t_3[1]
        t_4 = t_4.view(t_6, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_5 = t_5.view(t_6, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_2 = t_2.view(t_6, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_5 = t_5.transpose(3, 2)
        t_5 = torch.matmul(t_4, t_5)
        t_4 = torch.arange(t_3, dtype=torch.int64, device=self.device)
        t_4 = t_4[(slice(None, None, None), None)]
        t_3 = torch.arange(t_3, dtype=torch.int64, device=self.device)
        t_3 = t_3[(None, slice(None, None, None))]
        t_4 = t_3 - t_4
        t_4 = -t_4
        t_3 = torch.abs(t_4)
        t_4 = t_4 < 0
        t_4 = t_4.to(torch.int64)
        t_4 = t_4 * 16
        t_4 = 0 + t_4
        t_7 = t_3.float()
        t_8 = torch.less(t_3, 8)
        t_7 = t_7 / 8
        t_7 = torch.log(t_7)
        t_9 = math.log(16.0)
        t_9 = t_7 / t_9
        t_9 = t_9 * 8
        t_9 = t_9.to(torch.int64)
        t_9 = 8 + t_9
        t_7 = torch.full_like(t_9, 15, device=self.device)
        t_7 = torch.min(t_9, t_7)
        t_7 = torch.where(t_8, t_3, t_7)
        t_4 += t_7
        t_7 = t_4
        t_7 = t_7.to(self.device)
        t_7 = self.l_6(t_7)
        t_7 = t_7.permute([2, 0, 1])
        t_7 = t_7.unsqueeze(0)
        t_7 = t_7 + attention_mask
        t_5 += t_7
        t_4 = t_5.float()
        t_4 = torch.nn.functional.softmax(t_4, dim=-1, _stacklevel=3, dtype=None)
        t_5 = t_4.type_as(t_5)
        t_5 = self.l_7(t_5)
        t_2 = torch.matmul(t_5, t_2)
        t_2 = t_2.transpose(1, 2)
        t_2 = t_2.contiguous()
        t_6 = t_2.view(t_6, -1, 4096)
        t_6 = self.l_8(t_6)
        t_6 = self.l_9(t_6)
        t_6 = t_1 + t_6
        t_1 = self.l_10(t_6)
        t_1 = self.l_11(t_1)
        t_1 = torch.nn.functional.relu(t_1, inplace=False)
        t_1 = self.l_12(t_1)
        t_1 = self.l_13(t_1)
        t_1 = self.l_14(t_1)
        t_1 = t_6 + t_1
        t_6 = self.l_15(t_1)
        t_2 = t_6.size()
        t_5 = self.l_16(t_6)
        t_4 = self.l_17(t_6)
        t_6 = self.l_18(t_6)
        t_2 = t_2[0]
        t_5 = t_5.view(t_2, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_4 = t_4.view(t_2, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_6 = t_6.view(t_2, -1, 32, 128)
        t_6 = t_6.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_5, t_4)
        t_4 += t_7
        t_5 = t_4.float()
        t_5 = torch.nn.functional.softmax(t_5, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_5.type_as(t_4)
        t_4 = self.l_19(t_4)
        t_6 = torch.matmul(t_4, t_6)
        t_6 = t_6.transpose(1, 2)
        t_6 = t_6.contiguous()
        t_2 = t_6.view(t_2, -1, 4096)
        t_2 = self.l_20(t_2)
        t_2 = self.l_21(t_2)
        t_2 = t_1 + t_2
        t_1 = self.l_22(t_2)
        t_1 = self.l_23(t_1)
        t_1 = torch.nn.functional.relu(t_1, inplace=False)
        t_1 = self.l_24(t_1)
        t_1 = self.l_25(t_1)
        t_1 = self.l_26(t_1)
        t_1 = t_2 + t_1
        t_2 = self.l_27(t_1)
        t_6 = t_2.size()
        t_4 = self.l_28(t_2)
        t_5 = self.l_29(t_2)
        t_2 = self.l_30(t_2)
        t_6 = t_6[0]
        t_4 = t_4.view(t_6, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_5 = t_5.view(t_6, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_2 = t_2.view(t_6, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_5 = t_5.transpose(3, 2)
        t_5 = torch.matmul(t_4, t_5)
        t_5 += t_7
        t_4 = t_5.float()
        t_4 = torch.nn.functional.softmax(t_4, dim=-1, _stacklevel=3, dtype=None)
        t_5 = t_4.type_as(t_5)
        t_5 = self.l_31(t_5)
        t_2 = torch.matmul(t_5, t_2)
        t_2 = t_2.transpose(1, 2)
        t_2 = t_2.contiguous()
        t_6 = t_2.view(t_6, -1, 4096)
        t_6 = self.l_32(t_6)
        t_6 = self.l_33(t_6)
        t_6 = t_1 + t_6
        t_1 = self.l_34(t_6)
        t_1 = self.l_35(t_1)
        t_1 = torch.nn.functional.relu(t_1, inplace=False)
        t_1 = self.l_36(t_1)
        t_1 = self.l_37(t_1)
        t_1 = self.l_38(t_1)
        t_1 = t_6 + t_1
        t_6 = self.l_39(t_1)
        t_2 = t_6.size()
        t_5 = self.l_40(t_6)
        t_4 = self.l_41(t_6)
        t_6 = self.l_42(t_6)
        t_2 = t_2[0]
        t_5 = t_5.view(t_2, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_4 = t_4.view(t_2, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_6 = t_6.view(t_2, -1, 32, 128)
        t_6 = t_6.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_5, t_4)
        t_4 += t_7
        t_5 = t_4.float()
        t_5 = torch.nn.functional.softmax(t_5, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_5.type_as(t_4)
        t_4 = self.l_43(t_4)
        t_6 = torch.matmul(t_4, t_6)
        t_6 = t_6.transpose(1, 2)
        t_6 = t_6.contiguous()
        t_2 = t_6.view(t_2, -1, 4096)
        t_2 = self.l_44(t_2)
        t_2 = self.l_45(t_2)
        t_2 = t_1 + t_2
        t_1 = self.l_46(t_2)
        t_0 = t_0[-1]
        t_0 = decoder_input_ids.view(-1, t_0)
        t_0 = self.l_47(self.p_0, t_0)
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___94
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerSelfAttention[0]/Tensor::__add___290
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerFF[1]/T5LayerNorm[layer_norm]
        # T5ForConditionalGeneration/T5Stack[decoder]/StatelessEmbedding[embed_tokens]
        return list(flatten((t_7, t_2, t_1, t_0)))

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
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
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
        self.input_structure = [1, 1, 1]
        self.lookup = {'l_0': 'encoder.3.1.DenseReluDense.wi',
                        'l_1': 'encoder.3.1.DenseReluDense.dropout',
                        'l_2': 'encoder.3.1.DenseReluDense.wo',
                        'l_3': 'encoder.3.1.dropout',
                        'l_4': 'encoder.4.0.layer_norm',
                        'l_5': 'encoder.4.0.SelfAttention.q',
                        'l_6': 'encoder.4.0.SelfAttention.k',
                        'l_7': 'encoder.4.0.SelfAttention.v',
                        'l_8': 'encoder.4.0.SelfAttention.dropout',
                        'l_9': 'encoder.4.0.SelfAttention.o',
                        'l_10': 'encoder.4.0.dropout',
                        'l_11': 'encoder.4.1.layer_norm',
                        'l_12': 'encoder.4.1.DenseReluDense.wi',
                        'l_13': 'encoder.4.1.DenseReluDense.dropout',
                        'l_14': 'encoder.4.1.DenseReluDense.wo',
                        'l_15': 'encoder.4.1.dropout',
                        'l_16': 'encoder.5.0.layer_norm',
                        'l_17': 'encoder.5.0.SelfAttention.q',
                        'l_18': 'encoder.5.0.SelfAttention.k',
                        'l_19': 'encoder.5.0.SelfAttention.v',
                        'l_20': 'encoder.5.0.SelfAttention.dropout',
                        'l_21': 'encoder.5.0.SelfAttention.o',
                        'l_22': 'encoder.5.0.dropout',
                        'l_23': 'encoder.5.1.layer_norm',
                        'l_24': 'encoder.5.1.DenseReluDense.wi',
                        'l_25': 'encoder.5.1.DenseReluDense.dropout',
                        'l_26': 'encoder.5.1.DenseReluDense.wo',
                        'l_27': 'encoder.5.1.dropout',
                        'l_28': 'encoder.6.0.layer_norm',
                        'l_29': 'encoder.6.0.SelfAttention.q',
                        'l_30': 'encoder.6.0.SelfAttention.k',
                        'l_31': 'encoder.6.0.SelfAttention.v',
                        'l_32': 'encoder.6.0.SelfAttention.dropout',
                        'l_33': 'encoder.6.0.SelfAttention.o',
                        'l_34': 'encoder.6.0.dropout',
                        'l_35': 'encoder.6.1.layer_norm'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerFF[1]/Dropout[dropout] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerFF[1]/Dropout[dropout] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerFF[1]/Dropout[dropout] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___94 <=> x0
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerSelfAttention[0]/Tensor::__add___290 <=> x1
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> x2
        x0, x1, x2 = unflatten(args, self.input_structure)
        t_0 = self.l_0(x2)
        t_0 = torch.nn.functional.relu(t_0, inplace=False)
        t_0 = self.l_1(t_0)
        t_0 = self.l_2(t_0)
        t_0 = self.l_3(t_0)
        t_0 = x1 + t_0
        t_1 = self.l_4(t_0)
        t_2 = t_1.size()
        t_3 = self.l_5(t_1)
        t_4 = self.l_6(t_1)
        t_1 = self.l_7(t_1)
        t_2 = t_2[0]
        t_3 = t_3.view(t_2, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_4 = t_4.view(t_2, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_1 = t_1.view(t_2, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_3, t_4)
        t_4 += x0
        t_3 = t_4.float()
        t_3 = torch.nn.functional.softmax(t_3, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_3.type_as(t_4)
        t_4 = self.l_8(t_4)
        t_1 = torch.matmul(t_4, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_2 = t_1.view(t_2, -1, 4096)
        t_2 = self.l_9(t_2)
        t_2 = self.l_10(t_2)
        t_2 = t_0 + t_2
        t_0 = self.l_11(t_2)
        t_0 = self.l_12(t_0)
        t_0 = torch.nn.functional.relu(t_0, inplace=False)
        t_0 = self.l_13(t_0)
        t_0 = self.l_14(t_0)
        t_0 = self.l_15(t_0)
        t_0 = t_2 + t_0
        t_2 = self.l_16(t_0)
        t_1 = t_2.size()
        t_4 = self.l_17(t_2)
        t_3 = self.l_18(t_2)
        t_2 = self.l_19(t_2)
        t_1 = t_1[0]
        t_4 = t_4.view(t_1, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_3 = t_3.view(t_1, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_2 = t_2.view(t_1, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_3 = t_3.transpose(3, 2)
        t_3 = torch.matmul(t_4, t_3)
        t_3 += x0
        t_4 = t_3.float()
        t_4 = torch.nn.functional.softmax(t_4, dim=-1, _stacklevel=3, dtype=None)
        t_3 = t_4.type_as(t_3)
        t_3 = self.l_20(t_3)
        t_2 = torch.matmul(t_3, t_2)
        t_2 = t_2.transpose(1, 2)
        t_2 = t_2.contiguous()
        t_1 = t_2.view(t_1, -1, 4096)
        t_1 = self.l_21(t_1)
        t_1 = self.l_22(t_1)
        t_1 = t_0 + t_1
        t_0 = self.l_23(t_1)
        t_0 = self.l_24(t_0)
        t_0 = torch.nn.functional.relu(t_0, inplace=False)
        t_0 = self.l_25(t_0)
        t_0 = self.l_26(t_0)
        t_0 = self.l_27(t_0)
        t_0 = t_1 + t_0
        t_1 = self.l_28(t_0)
        t_2 = t_1.size()
        t_3 = self.l_29(t_1)
        t_4 = self.l_30(t_1)
        t_1 = self.l_31(t_1)
        t_2 = t_2[0]
        t_3 = t_3.view(t_2, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_4 = t_4.view(t_2, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_1 = t_1.view(t_2, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_3, t_4)
        t_4 += x0
        t_3 = t_4.float()
        t_3 = torch.nn.functional.softmax(t_3, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_3.type_as(t_4)
        t_4 = self.l_32(t_4)
        t_1 = torch.matmul(t_4, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_2 = t_1.view(t_2, -1, 4096)
        t_2 = self.l_33(t_2)
        t_2 = self.l_34(t_2)
        t_2 = t_0 + t_2
        t_0 = self.l_35(t_2)
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___94
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerSelfAttention[0]/Tensor::__add___467
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerFF[1]/T5LayerNorm[layer_norm]
        return list(flatten((x0, t_2, t_0)))

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
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[7]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[7]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[7]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[7]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[7]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[7]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[7]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[7]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[7]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[7]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[7]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[7]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[8]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[8]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[8]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[8]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[8]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[8]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[8]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[8]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[8]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[8]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[8]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[8]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
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
        self.input_structure = [1, 1, 1]
        self.lookup = {'l_0': 'encoder.6.1.DenseReluDense.wi',
                        'l_1': 'encoder.6.1.DenseReluDense.dropout',
                        'l_2': 'encoder.6.1.DenseReluDense.wo',
                        'l_3': 'encoder.6.1.dropout',
                        'l_4': 'encoder.7.0.layer_norm',
                        'l_5': 'encoder.7.0.SelfAttention.q',
                        'l_6': 'encoder.7.0.SelfAttention.k',
                        'l_7': 'encoder.7.0.SelfAttention.v',
                        'l_8': 'encoder.7.0.SelfAttention.dropout',
                        'l_9': 'encoder.7.0.SelfAttention.o',
                        'l_10': 'encoder.7.0.dropout',
                        'l_11': 'encoder.7.1.layer_norm',
                        'l_12': 'encoder.7.1.DenseReluDense.wi',
                        'l_13': 'encoder.7.1.DenseReluDense.dropout',
                        'l_14': 'encoder.7.1.DenseReluDense.wo',
                        'l_15': 'encoder.7.1.dropout',
                        'l_16': 'encoder.8.0.layer_norm',
                        'l_17': 'encoder.8.0.SelfAttention.q',
                        'l_18': 'encoder.8.0.SelfAttention.k',
                        'l_19': 'encoder.8.0.SelfAttention.v',
                        'l_20': 'encoder.8.0.SelfAttention.dropout',
                        'l_21': 'encoder.8.0.SelfAttention.o',
                        'l_22': 'encoder.8.0.dropout',
                        'l_23': 'encoder.8.1.layer_norm',
                        'l_24': 'encoder.8.1.DenseReluDense.wi',
                        'l_25': 'encoder.8.1.DenseReluDense.dropout',
                        'l_26': 'encoder.8.1.DenseReluDense.wo',
                        'l_27': 'encoder.8.1.dropout',
                        'l_28': 'encoder.9.0.layer_norm',
                        'l_29': 'encoder.9.0.SelfAttention.q',
                        'l_30': 'encoder.9.0.SelfAttention.k',
                        'l_31': 'encoder.9.0.SelfAttention.v',
                        'l_32': 'encoder.9.0.SelfAttention.dropout',
                        'l_33': 'encoder.9.0.SelfAttention.o',
                        'l_34': 'encoder.9.0.dropout',
                        'l_35': 'encoder.9.1.layer_norm'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerFF[1]/Dropout[dropout] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[7]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[7]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[7]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[7]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[7]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[7]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[7]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[7]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[7]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[7]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[7]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[7]/T5LayerFF[1]/Dropout[dropout] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[8]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[8]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[8]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[8]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[8]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[8]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[8]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[8]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[8]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[8]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[8]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[8]/T5LayerFF[1]/Dropout[dropout] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___94 <=> x0
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerSelfAttention[0]/Tensor::__add___467 <=> x1
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[6]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> x2
        x0, x1, x2 = unflatten(args, self.input_structure)
        t_0 = self.l_0(x2)
        t_0 = torch.nn.functional.relu(t_0, inplace=False)
        t_0 = self.l_1(t_0)
        t_0 = self.l_2(t_0)
        t_0 = self.l_3(t_0)
        t_0 = x1 + t_0
        t_1 = self.l_4(t_0)
        t_2 = t_1.size()
        t_3 = self.l_5(t_1)
        t_4 = self.l_6(t_1)
        t_1 = self.l_7(t_1)
        t_2 = t_2[0]
        t_3 = t_3.view(t_2, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_4 = t_4.view(t_2, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_1 = t_1.view(t_2, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_3, t_4)
        t_4 += x0
        t_3 = t_4.float()
        t_3 = torch.nn.functional.softmax(t_3, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_3.type_as(t_4)
        t_4 = self.l_8(t_4)
        t_1 = torch.matmul(t_4, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_2 = t_1.view(t_2, -1, 4096)
        t_2 = self.l_9(t_2)
        t_2 = self.l_10(t_2)
        t_2 = t_0 + t_2
        t_0 = self.l_11(t_2)
        t_0 = self.l_12(t_0)
        t_0 = torch.nn.functional.relu(t_0, inplace=False)
        t_0 = self.l_13(t_0)
        t_0 = self.l_14(t_0)
        t_0 = self.l_15(t_0)
        t_0 = t_2 + t_0
        t_2 = self.l_16(t_0)
        t_1 = t_2.size()
        t_4 = self.l_17(t_2)
        t_3 = self.l_18(t_2)
        t_2 = self.l_19(t_2)
        t_1 = t_1[0]
        t_4 = t_4.view(t_1, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_3 = t_3.view(t_1, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_2 = t_2.view(t_1, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_3 = t_3.transpose(3, 2)
        t_3 = torch.matmul(t_4, t_3)
        t_3 += x0
        t_4 = t_3.float()
        t_4 = torch.nn.functional.softmax(t_4, dim=-1, _stacklevel=3, dtype=None)
        t_3 = t_4.type_as(t_3)
        t_3 = self.l_20(t_3)
        t_2 = torch.matmul(t_3, t_2)
        t_2 = t_2.transpose(1, 2)
        t_2 = t_2.contiguous()
        t_1 = t_2.view(t_1, -1, 4096)
        t_1 = self.l_21(t_1)
        t_1 = self.l_22(t_1)
        t_1 = t_0 + t_1
        t_0 = self.l_23(t_1)
        t_0 = self.l_24(t_0)
        t_0 = torch.nn.functional.relu(t_0, inplace=False)
        t_0 = self.l_25(t_0)
        t_0 = self.l_26(t_0)
        t_0 = self.l_27(t_0)
        t_0 = t_1 + t_0
        t_1 = self.l_28(t_0)
        t_2 = t_1.size()
        t_3 = self.l_29(t_1)
        t_4 = self.l_30(t_1)
        t_1 = self.l_31(t_1)
        t_2 = t_2[0]
        t_3 = t_3.view(t_2, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_4 = t_4.view(t_2, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_1 = t_1.view(t_2, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_3, t_4)
        t_4 += x0
        t_3 = t_4.float()
        t_3 = torch.nn.functional.softmax(t_3, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_3.type_as(t_4)
        t_4 = self.l_32(t_4)
        t_1 = torch.matmul(t_4, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_2 = t_1.view(t_2, -1, 4096)
        t_2 = self.l_33(t_2)
        t_2 = self.l_34(t_2)
        t_2 = t_0 + t_2
        t_0 = self.l_35(t_2)
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___94
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerSelfAttention[0]/Tensor::__add___644
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerFF[1]/T5LayerNorm[layer_norm]
        return list(flatten((x0, t_2, t_0)))

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
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[10]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[10]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[10]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[10]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[10]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[10]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[10]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[10]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[10]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[10]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[10]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[10]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[11]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[11]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[11]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[11]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[11]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[11]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[11]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[11]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[11]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[11]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[11]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[11]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
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
        self.input_structure = [1, 1, 1]
        self.lookup = {'l_0': 'encoder.9.1.DenseReluDense.wi',
                        'l_1': 'encoder.9.1.DenseReluDense.dropout',
                        'l_2': 'encoder.9.1.DenseReluDense.wo',
                        'l_3': 'encoder.9.1.dropout',
                        'l_4': 'encoder.10.0.layer_norm',
                        'l_5': 'encoder.10.0.SelfAttention.q',
                        'l_6': 'encoder.10.0.SelfAttention.k',
                        'l_7': 'encoder.10.0.SelfAttention.v',
                        'l_8': 'encoder.10.0.SelfAttention.dropout',
                        'l_9': 'encoder.10.0.SelfAttention.o',
                        'l_10': 'encoder.10.0.dropout',
                        'l_11': 'encoder.10.1.layer_norm',
                        'l_12': 'encoder.10.1.DenseReluDense.wi',
                        'l_13': 'encoder.10.1.DenseReluDense.dropout',
                        'l_14': 'encoder.10.1.DenseReluDense.wo',
                        'l_15': 'encoder.10.1.dropout',
                        'l_16': 'encoder.11.0.layer_norm',
                        'l_17': 'encoder.11.0.SelfAttention.q',
                        'l_18': 'encoder.11.0.SelfAttention.k',
                        'l_19': 'encoder.11.0.SelfAttention.v',
                        'l_20': 'encoder.11.0.SelfAttention.dropout',
                        'l_21': 'encoder.11.0.SelfAttention.o',
                        'l_22': 'encoder.11.0.dropout',
                        'l_23': 'encoder.11.1.layer_norm',
                        'l_24': 'encoder.11.1.DenseReluDense.wi',
                        'l_25': 'encoder.11.1.DenseReluDense.dropout',
                        'l_26': 'encoder.11.1.DenseReluDense.wo',
                        'l_27': 'encoder.11.1.dropout',
                        'l_28': 'encoder.12.0.layer_norm',
                        'l_29': 'encoder.12.0.SelfAttention.q',
                        'l_30': 'encoder.12.0.SelfAttention.k',
                        'l_31': 'encoder.12.0.SelfAttention.v',
                        'l_32': 'encoder.12.0.SelfAttention.dropout',
                        'l_33': 'encoder.12.0.SelfAttention.o',
                        'l_34': 'encoder.12.0.dropout',
                        'l_35': 'encoder.12.1.layer_norm'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerFF[1]/Dropout[dropout] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[10]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[10]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[10]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[10]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[10]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[10]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[10]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[10]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[10]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[10]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[10]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[10]/T5LayerFF[1]/Dropout[dropout] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[11]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[11]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[11]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[11]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[11]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[11]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[11]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[11]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[11]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[11]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[11]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[11]/T5LayerFF[1]/Dropout[dropout] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___94 <=> x0
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerSelfAttention[0]/Tensor::__add___644 <=> x1
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[9]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> x2
        x0, x1, x2 = unflatten(args, self.input_structure)
        t_0 = self.l_0(x2)
        t_0 = torch.nn.functional.relu(t_0, inplace=False)
        t_0 = self.l_1(t_0)
        t_0 = self.l_2(t_0)
        t_0 = self.l_3(t_0)
        t_0 = x1 + t_0
        t_1 = self.l_4(t_0)
        t_2 = t_1.size()
        t_3 = self.l_5(t_1)
        t_4 = self.l_6(t_1)
        t_1 = self.l_7(t_1)
        t_2 = t_2[0]
        t_3 = t_3.view(t_2, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_4 = t_4.view(t_2, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_1 = t_1.view(t_2, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_3, t_4)
        t_4 += x0
        t_3 = t_4.float()
        t_3 = torch.nn.functional.softmax(t_3, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_3.type_as(t_4)
        t_4 = self.l_8(t_4)
        t_1 = torch.matmul(t_4, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_2 = t_1.view(t_2, -1, 4096)
        t_2 = self.l_9(t_2)
        t_2 = self.l_10(t_2)
        t_2 = t_0 + t_2
        t_0 = self.l_11(t_2)
        t_0 = self.l_12(t_0)
        t_0 = torch.nn.functional.relu(t_0, inplace=False)
        t_0 = self.l_13(t_0)
        t_0 = self.l_14(t_0)
        t_0 = self.l_15(t_0)
        t_0 = t_2 + t_0
        t_2 = self.l_16(t_0)
        t_1 = t_2.size()
        t_4 = self.l_17(t_2)
        t_3 = self.l_18(t_2)
        t_2 = self.l_19(t_2)
        t_1 = t_1[0]
        t_4 = t_4.view(t_1, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_3 = t_3.view(t_1, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_2 = t_2.view(t_1, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_3 = t_3.transpose(3, 2)
        t_3 = torch.matmul(t_4, t_3)
        t_3 += x0
        t_4 = t_3.float()
        t_4 = torch.nn.functional.softmax(t_4, dim=-1, _stacklevel=3, dtype=None)
        t_3 = t_4.type_as(t_3)
        t_3 = self.l_20(t_3)
        t_2 = torch.matmul(t_3, t_2)
        t_2 = t_2.transpose(1, 2)
        t_2 = t_2.contiguous()
        t_1 = t_2.view(t_1, -1, 4096)
        t_1 = self.l_21(t_1)
        t_1 = self.l_22(t_1)
        t_1 = t_0 + t_1
        t_0 = self.l_23(t_1)
        t_0 = self.l_24(t_0)
        t_0 = torch.nn.functional.relu(t_0, inplace=False)
        t_0 = self.l_25(t_0)
        t_0 = self.l_26(t_0)
        t_0 = self.l_27(t_0)
        t_0 = t_1 + t_0
        t_1 = self.l_28(t_0)
        t_2 = t_1.size()
        t_3 = self.l_29(t_1)
        t_4 = self.l_30(t_1)
        t_1 = self.l_31(t_1)
        t_2 = t_2[0]
        t_3 = t_3.view(t_2, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_4 = t_4.view(t_2, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_1 = t_1.view(t_2, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_3, t_4)
        t_4 += x0
        t_3 = t_4.float()
        t_3 = torch.nn.functional.softmax(t_3, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_3.type_as(t_4)
        t_4 = self.l_32(t_4)
        t_1 = torch.matmul(t_4, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_2 = t_1.view(t_2, -1, 4096)
        t_2 = self.l_33(t_2)
        t_2 = self.l_34(t_2)
        t_2 = t_0 + t_2
        t_0 = self.l_35(t_2)
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___94
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerSelfAttention[0]/Tensor::__add___821
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerFF[1]/T5LayerNorm[layer_norm]
        return list(flatten((x0, t_2, t_0)))

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
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[13]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[13]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[13]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[13]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[13]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[13]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[13]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[13]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[13]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[13]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[13]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[13]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[14]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[14]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[14]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[14]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[14]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[14]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[14]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[14]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[14]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[14]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[14]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[14]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
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
        self.input_structure = [1, 1, 1]
        self.lookup = {'l_0': 'encoder.12.1.DenseReluDense.wi',
                        'l_1': 'encoder.12.1.DenseReluDense.dropout',
                        'l_2': 'encoder.12.1.DenseReluDense.wo',
                        'l_3': 'encoder.12.1.dropout',
                        'l_4': 'encoder.13.0.layer_norm',
                        'l_5': 'encoder.13.0.SelfAttention.q',
                        'l_6': 'encoder.13.0.SelfAttention.k',
                        'l_7': 'encoder.13.0.SelfAttention.v',
                        'l_8': 'encoder.13.0.SelfAttention.dropout',
                        'l_9': 'encoder.13.0.SelfAttention.o',
                        'l_10': 'encoder.13.0.dropout',
                        'l_11': 'encoder.13.1.layer_norm',
                        'l_12': 'encoder.13.1.DenseReluDense.wi',
                        'l_13': 'encoder.13.1.DenseReluDense.dropout',
                        'l_14': 'encoder.13.1.DenseReluDense.wo',
                        'l_15': 'encoder.13.1.dropout',
                        'l_16': 'encoder.14.0.layer_norm',
                        'l_17': 'encoder.14.0.SelfAttention.q',
                        'l_18': 'encoder.14.0.SelfAttention.k',
                        'l_19': 'encoder.14.0.SelfAttention.v',
                        'l_20': 'encoder.14.0.SelfAttention.dropout',
                        'l_21': 'encoder.14.0.SelfAttention.o',
                        'l_22': 'encoder.14.0.dropout',
                        'l_23': 'encoder.14.1.layer_norm',
                        'l_24': 'encoder.14.1.DenseReluDense.wi',
                        'l_25': 'encoder.14.1.DenseReluDense.dropout',
                        'l_26': 'encoder.14.1.DenseReluDense.wo',
                        'l_27': 'encoder.14.1.dropout',
                        'l_28': 'encoder.15.0.layer_norm',
                        'l_29': 'encoder.15.0.SelfAttention.q',
                        'l_30': 'encoder.15.0.SelfAttention.k',
                        'l_31': 'encoder.15.0.SelfAttention.v',
                        'l_32': 'encoder.15.0.SelfAttention.dropout',
                        'l_33': 'encoder.15.0.SelfAttention.o',
                        'l_34': 'encoder.15.0.dropout',
                        'l_35': 'encoder.15.1.layer_norm'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerFF[1]/Dropout[dropout] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[13]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[13]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[13]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[13]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[13]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[13]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[13]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[13]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[13]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[13]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[13]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[13]/T5LayerFF[1]/Dropout[dropout] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[14]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[14]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[14]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[14]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[14]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[14]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[14]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[14]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[14]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[14]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[14]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[14]/T5LayerFF[1]/Dropout[dropout] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___94 <=> x0
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerSelfAttention[0]/Tensor::__add___821 <=> x1
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[12]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> x2
        x0, x1, x2 = unflatten(args, self.input_structure)
        t_0 = self.l_0(x2)
        t_0 = torch.nn.functional.relu(t_0, inplace=False)
        t_0 = self.l_1(t_0)
        t_0 = self.l_2(t_0)
        t_0 = self.l_3(t_0)
        t_0 = x1 + t_0
        t_1 = self.l_4(t_0)
        t_2 = t_1.size()
        t_3 = self.l_5(t_1)
        t_4 = self.l_6(t_1)
        t_1 = self.l_7(t_1)
        t_2 = t_2[0]
        t_3 = t_3.view(t_2, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_4 = t_4.view(t_2, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_1 = t_1.view(t_2, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_3, t_4)
        t_4 += x0
        t_3 = t_4.float()
        t_3 = torch.nn.functional.softmax(t_3, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_3.type_as(t_4)
        t_4 = self.l_8(t_4)
        t_1 = torch.matmul(t_4, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_2 = t_1.view(t_2, -1, 4096)
        t_2 = self.l_9(t_2)
        t_2 = self.l_10(t_2)
        t_2 = t_0 + t_2
        t_0 = self.l_11(t_2)
        t_0 = self.l_12(t_0)
        t_0 = torch.nn.functional.relu(t_0, inplace=False)
        t_0 = self.l_13(t_0)
        t_0 = self.l_14(t_0)
        t_0 = self.l_15(t_0)
        t_0 = t_2 + t_0
        t_2 = self.l_16(t_0)
        t_1 = t_2.size()
        t_4 = self.l_17(t_2)
        t_3 = self.l_18(t_2)
        t_2 = self.l_19(t_2)
        t_1 = t_1[0]
        t_4 = t_4.view(t_1, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_3 = t_3.view(t_1, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_2 = t_2.view(t_1, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_3 = t_3.transpose(3, 2)
        t_3 = torch.matmul(t_4, t_3)
        t_3 += x0
        t_4 = t_3.float()
        t_4 = torch.nn.functional.softmax(t_4, dim=-1, _stacklevel=3, dtype=None)
        t_3 = t_4.type_as(t_3)
        t_3 = self.l_20(t_3)
        t_2 = torch.matmul(t_3, t_2)
        t_2 = t_2.transpose(1, 2)
        t_2 = t_2.contiguous()
        t_1 = t_2.view(t_1, -1, 4096)
        t_1 = self.l_21(t_1)
        t_1 = self.l_22(t_1)
        t_1 = t_0 + t_1
        t_0 = self.l_23(t_1)
        t_0 = self.l_24(t_0)
        t_0 = torch.nn.functional.relu(t_0, inplace=False)
        t_0 = self.l_25(t_0)
        t_0 = self.l_26(t_0)
        t_0 = self.l_27(t_0)
        t_0 = t_1 + t_0
        t_1 = self.l_28(t_0)
        t_2 = t_1.size()
        t_3 = self.l_29(t_1)
        t_4 = self.l_30(t_1)
        t_1 = self.l_31(t_1)
        t_2 = t_2[0]
        t_3 = t_3.view(t_2, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_4 = t_4.view(t_2, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_1 = t_1.view(t_2, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_3, t_4)
        t_4 += x0
        t_3 = t_4.float()
        t_3 = torch.nn.functional.softmax(t_3, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_3.type_as(t_4)
        t_4 = self.l_32(t_4)
        t_1 = torch.matmul(t_4, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_2 = t_1.view(t_2, -1, 4096)
        t_2 = self.l_33(t_2)
        t_2 = self.l_34(t_2)
        t_2 = t_0 + t_2
        t_0 = self.l_35(t_2)
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___94
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerSelfAttention[0]/Tensor::__add___998
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerFF[1]/T5LayerNorm[layer_norm]
        return list(flatten((x0, t_2, t_0)))

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
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[16]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[16]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[16]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[16]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[16]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[16]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[16]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[16]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[16]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[16]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[16]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[16]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[17]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[17]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[17]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[17]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[17]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[17]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[17]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[17]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[17]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[17]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[17]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[17]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
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
        self.input_structure = [1, 1, 1]
        self.lookup = {'l_0': 'encoder.15.1.DenseReluDense.wi',
                        'l_1': 'encoder.15.1.DenseReluDense.dropout',
                        'l_2': 'encoder.15.1.DenseReluDense.wo',
                        'l_3': 'encoder.15.1.dropout',
                        'l_4': 'encoder.16.0.layer_norm',
                        'l_5': 'encoder.16.0.SelfAttention.q',
                        'l_6': 'encoder.16.0.SelfAttention.k',
                        'l_7': 'encoder.16.0.SelfAttention.v',
                        'l_8': 'encoder.16.0.SelfAttention.dropout',
                        'l_9': 'encoder.16.0.SelfAttention.o',
                        'l_10': 'encoder.16.0.dropout',
                        'l_11': 'encoder.16.1.layer_norm',
                        'l_12': 'encoder.16.1.DenseReluDense.wi',
                        'l_13': 'encoder.16.1.DenseReluDense.dropout',
                        'l_14': 'encoder.16.1.DenseReluDense.wo',
                        'l_15': 'encoder.16.1.dropout',
                        'l_16': 'encoder.17.0.layer_norm',
                        'l_17': 'encoder.17.0.SelfAttention.q',
                        'l_18': 'encoder.17.0.SelfAttention.k',
                        'l_19': 'encoder.17.0.SelfAttention.v',
                        'l_20': 'encoder.17.0.SelfAttention.dropout',
                        'l_21': 'encoder.17.0.SelfAttention.o',
                        'l_22': 'encoder.17.0.dropout',
                        'l_23': 'encoder.17.1.layer_norm',
                        'l_24': 'encoder.17.1.DenseReluDense.wi',
                        'l_25': 'encoder.17.1.DenseReluDense.dropout',
                        'l_26': 'encoder.17.1.DenseReluDense.wo',
                        'l_27': 'encoder.17.1.dropout',
                        'l_28': 'encoder.18.0.layer_norm',
                        'l_29': 'encoder.18.0.SelfAttention.q',
                        'l_30': 'encoder.18.0.SelfAttention.k',
                        'l_31': 'encoder.18.0.SelfAttention.v',
                        'l_32': 'encoder.18.0.SelfAttention.dropout',
                        'l_33': 'encoder.18.0.SelfAttention.o',
                        'l_34': 'encoder.18.0.dropout',
                        'l_35': 'encoder.18.1.layer_norm'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerFF[1]/Dropout[dropout] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[16]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[16]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[16]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[16]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[16]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[16]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[16]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[16]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[16]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[16]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[16]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[16]/T5LayerFF[1]/Dropout[dropout] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[17]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[17]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[17]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[17]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[17]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[17]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[17]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[17]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[17]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[17]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[17]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[17]/T5LayerFF[1]/Dropout[dropout] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___94 <=> x0
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerSelfAttention[0]/Tensor::__add___998 <=> x1
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> x2
        x0, x1, x2 = unflatten(args, self.input_structure)
        t_0 = self.l_0(x2)
        t_0 = torch.nn.functional.relu(t_0, inplace=False)
        t_0 = self.l_1(t_0)
        t_0 = self.l_2(t_0)
        t_0 = self.l_3(t_0)
        t_0 = x1 + t_0
        t_1 = self.l_4(t_0)
        t_2 = t_1.size()
        t_3 = self.l_5(t_1)
        t_4 = self.l_6(t_1)
        t_1 = self.l_7(t_1)
        t_2 = t_2[0]
        t_3 = t_3.view(t_2, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_4 = t_4.view(t_2, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_1 = t_1.view(t_2, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_3, t_4)
        t_4 += x0
        t_3 = t_4.float()
        t_3 = torch.nn.functional.softmax(t_3, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_3.type_as(t_4)
        t_4 = self.l_8(t_4)
        t_1 = torch.matmul(t_4, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_2 = t_1.view(t_2, -1, 4096)
        t_2 = self.l_9(t_2)
        t_2 = self.l_10(t_2)
        t_2 = t_0 + t_2
        t_0 = self.l_11(t_2)
        t_0 = self.l_12(t_0)
        t_0 = torch.nn.functional.relu(t_0, inplace=False)
        t_0 = self.l_13(t_0)
        t_0 = self.l_14(t_0)
        t_0 = self.l_15(t_0)
        t_0 = t_2 + t_0
        t_2 = self.l_16(t_0)
        t_1 = t_2.size()
        t_4 = self.l_17(t_2)
        t_3 = self.l_18(t_2)
        t_2 = self.l_19(t_2)
        t_1 = t_1[0]
        t_4 = t_4.view(t_1, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_3 = t_3.view(t_1, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_2 = t_2.view(t_1, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_3 = t_3.transpose(3, 2)
        t_3 = torch.matmul(t_4, t_3)
        t_3 += x0
        t_4 = t_3.float()
        t_4 = torch.nn.functional.softmax(t_4, dim=-1, _stacklevel=3, dtype=None)
        t_3 = t_4.type_as(t_3)
        t_3 = self.l_20(t_3)
        t_2 = torch.matmul(t_3, t_2)
        t_2 = t_2.transpose(1, 2)
        t_2 = t_2.contiguous()
        t_1 = t_2.view(t_1, -1, 4096)
        t_1 = self.l_21(t_1)
        t_1 = self.l_22(t_1)
        t_1 = t_0 + t_1
        t_0 = self.l_23(t_1)
        t_0 = self.l_24(t_0)
        t_0 = torch.nn.functional.relu(t_0, inplace=False)
        t_0 = self.l_25(t_0)
        t_0 = self.l_26(t_0)
        t_0 = self.l_27(t_0)
        t_0 = t_1 + t_0
        t_1 = self.l_28(t_0)
        t_2 = t_1.size()
        t_3 = self.l_29(t_1)
        t_4 = self.l_30(t_1)
        t_1 = self.l_31(t_1)
        t_2 = t_2[0]
        t_3 = t_3.view(t_2, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_4 = t_4.view(t_2, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_1 = t_1.view(t_2, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_3, t_4)
        t_4 += x0
        t_3 = t_4.float()
        t_3 = torch.nn.functional.softmax(t_3, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_3.type_as(t_4)
        t_4 = self.l_32(t_4)
        t_1 = torch.matmul(t_4, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_2 = t_1.view(t_2, -1, 4096)
        t_2 = self.l_33(t_2)
        t_2 = self.l_34(t_2)
        t_2 = t_0 + t_2
        t_0 = self.l_35(t_2)
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___94
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerSelfAttention[0]/Tensor::__add___1175
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerFF[1]/T5LayerNorm[layer_norm]
        return list(flatten((x0, t_2, t_0)))

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
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[19]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[19]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[19]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[19]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[19]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[19]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[19]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[19]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[19]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[19]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[19]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[19]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[20]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[20]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[20]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[20]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[20]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[20]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[20]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[20]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[20]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[20]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[20]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[20]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
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
        self.input_structure = [1, 1, 1]
        self.lookup = {'l_0': 'encoder.18.1.DenseReluDense.wi',
                        'l_1': 'encoder.18.1.DenseReluDense.dropout',
                        'l_2': 'encoder.18.1.DenseReluDense.wo',
                        'l_3': 'encoder.18.1.dropout',
                        'l_4': 'encoder.19.0.layer_norm',
                        'l_5': 'encoder.19.0.SelfAttention.q',
                        'l_6': 'encoder.19.0.SelfAttention.k',
                        'l_7': 'encoder.19.0.SelfAttention.v',
                        'l_8': 'encoder.19.0.SelfAttention.dropout',
                        'l_9': 'encoder.19.0.SelfAttention.o',
                        'l_10': 'encoder.19.0.dropout',
                        'l_11': 'encoder.19.1.layer_norm',
                        'l_12': 'encoder.19.1.DenseReluDense.wi',
                        'l_13': 'encoder.19.1.DenseReluDense.dropout',
                        'l_14': 'encoder.19.1.DenseReluDense.wo',
                        'l_15': 'encoder.19.1.dropout',
                        'l_16': 'encoder.20.0.layer_norm',
                        'l_17': 'encoder.20.0.SelfAttention.q',
                        'l_18': 'encoder.20.0.SelfAttention.k',
                        'l_19': 'encoder.20.0.SelfAttention.v',
                        'l_20': 'encoder.20.0.SelfAttention.dropout',
                        'l_21': 'encoder.20.0.SelfAttention.o',
                        'l_22': 'encoder.20.0.dropout',
                        'l_23': 'encoder.20.1.layer_norm',
                        'l_24': 'encoder.20.1.DenseReluDense.wi',
                        'l_25': 'encoder.20.1.DenseReluDense.dropout',
                        'l_26': 'encoder.20.1.DenseReluDense.wo',
                        'l_27': 'encoder.20.1.dropout',
                        'l_28': 'encoder.21.0.layer_norm',
                        'l_29': 'encoder.21.0.SelfAttention.q',
                        'l_30': 'encoder.21.0.SelfAttention.k',
                        'l_31': 'encoder.21.0.SelfAttention.v',
                        'l_32': 'encoder.21.0.SelfAttention.dropout',
                        'l_33': 'encoder.21.0.SelfAttention.o',
                        'l_34': 'encoder.21.0.dropout',
                        'l_35': 'encoder.21.1.layer_norm'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerFF[1]/Dropout[dropout] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[19]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[19]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[19]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[19]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[19]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[19]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[19]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[19]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[19]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[19]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[19]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[19]/T5LayerFF[1]/Dropout[dropout] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[20]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[20]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[20]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[20]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[20]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[20]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[20]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[20]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[20]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[20]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[20]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[20]/T5LayerFF[1]/Dropout[dropout] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___94 <=> x0
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerSelfAttention[0]/Tensor::__add___1175 <=> x1
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> x2
        x0, x1, x2 = unflatten(args, self.input_structure)
        t_0 = self.l_0(x2)
        t_0 = torch.nn.functional.relu(t_0, inplace=False)
        t_0 = self.l_1(t_0)
        t_0 = self.l_2(t_0)
        t_0 = self.l_3(t_0)
        t_0 = x1 + t_0
        t_1 = self.l_4(t_0)
        t_2 = t_1.size()
        t_3 = self.l_5(t_1)
        t_4 = self.l_6(t_1)
        t_1 = self.l_7(t_1)
        t_2 = t_2[0]
        t_3 = t_3.view(t_2, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_4 = t_4.view(t_2, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_1 = t_1.view(t_2, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_3, t_4)
        t_4 += x0
        t_3 = t_4.float()
        t_3 = torch.nn.functional.softmax(t_3, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_3.type_as(t_4)
        t_4 = self.l_8(t_4)
        t_1 = torch.matmul(t_4, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_2 = t_1.view(t_2, -1, 4096)
        t_2 = self.l_9(t_2)
        t_2 = self.l_10(t_2)
        t_2 = t_0 + t_2
        t_0 = self.l_11(t_2)
        t_0 = self.l_12(t_0)
        t_0 = torch.nn.functional.relu(t_0, inplace=False)
        t_0 = self.l_13(t_0)
        t_0 = self.l_14(t_0)
        t_0 = self.l_15(t_0)
        t_0 = t_2 + t_0
        t_2 = self.l_16(t_0)
        t_1 = t_2.size()
        t_4 = self.l_17(t_2)
        t_3 = self.l_18(t_2)
        t_2 = self.l_19(t_2)
        t_1 = t_1[0]
        t_4 = t_4.view(t_1, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_3 = t_3.view(t_1, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_2 = t_2.view(t_1, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_3 = t_3.transpose(3, 2)
        t_3 = torch.matmul(t_4, t_3)
        t_3 += x0
        t_4 = t_3.float()
        t_4 = torch.nn.functional.softmax(t_4, dim=-1, _stacklevel=3, dtype=None)
        t_3 = t_4.type_as(t_3)
        t_3 = self.l_20(t_3)
        t_2 = torch.matmul(t_3, t_2)
        t_2 = t_2.transpose(1, 2)
        t_2 = t_2.contiguous()
        t_1 = t_2.view(t_1, -1, 4096)
        t_1 = self.l_21(t_1)
        t_1 = self.l_22(t_1)
        t_1 = t_0 + t_1
        t_0 = self.l_23(t_1)
        t_0 = self.l_24(t_0)
        t_0 = torch.nn.functional.relu(t_0, inplace=False)
        t_0 = self.l_25(t_0)
        t_0 = self.l_26(t_0)
        t_0 = self.l_27(t_0)
        t_0 = t_1 + t_0
        t_1 = self.l_28(t_0)
        t_2 = t_1.size()
        t_3 = self.l_29(t_1)
        t_4 = self.l_30(t_1)
        t_1 = self.l_31(t_1)
        t_2 = t_2[0]
        t_3 = t_3.view(t_2, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_4 = t_4.view(t_2, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_1 = t_1.view(t_2, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_3, t_4)
        t_4 += x0
        t_3 = t_4.float()
        t_3 = torch.nn.functional.softmax(t_3, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_3.type_as(t_4)
        t_4 = self.l_32(t_4)
        t_1 = torch.matmul(t_4, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_2 = t_1.view(t_2, -1, 4096)
        t_2 = self.l_33(t_2)
        t_2 = self.l_34(t_2)
        t_2 = t_0 + t_2
        t_0 = self.l_35(t_2)
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___94
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerSelfAttention[0]/Tensor::__add___1352
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerFF[1]/T5LayerNorm[layer_norm]
        return list(flatten((x0, t_2, t_0)))

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
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[22]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[22]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[22]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[22]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[22]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[22]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[22]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[22]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[22]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[22]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[22]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[22]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[23]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[23]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[23]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[23]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[23]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[23]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[23]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[23]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[23]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[23]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[23]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[23]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5LayerNorm[final_layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Embedding[relative_attention_bias]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Embedding[relative_attention_bias]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
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
        self.input_structure = [1, 1, 1, 1, 1, 1]
        self.lookup = {'l_0': 'encoder.21.1.DenseReluDense.wi',
                        'l_1': 'encoder.21.1.DenseReluDense.dropout',
                        'l_2': 'encoder.21.1.DenseReluDense.wo',
                        'l_3': 'encoder.21.1.dropout',
                        'l_4': 'encoder.22.0.layer_norm',
                        'l_5': 'encoder.22.0.SelfAttention.q',
                        'l_6': 'encoder.22.0.SelfAttention.k',
                        'l_7': 'encoder.22.0.SelfAttention.v',
                        'l_8': 'encoder.22.0.SelfAttention.dropout',
                        'l_9': 'encoder.22.0.SelfAttention.o',
                        'l_10': 'encoder.22.0.dropout',
                        'l_11': 'encoder.22.1.layer_norm',
                        'l_12': 'encoder.22.1.DenseReluDense.wi',
                        'l_13': 'encoder.22.1.DenseReluDense.dropout',
                        'l_14': 'encoder.22.1.DenseReluDense.wo',
                        'l_15': 'encoder.22.1.dropout',
                        'l_16': 'encoder.23.0.layer_norm',
                        'l_17': 'encoder.23.0.SelfAttention.q',
                        'l_18': 'encoder.23.0.SelfAttention.k',
                        'l_19': 'encoder.23.0.SelfAttention.v',
                        'l_20': 'encoder.23.0.SelfAttention.dropout',
                        'l_21': 'encoder.23.0.SelfAttention.o',
                        'l_22': 'encoder.23.0.dropout',
                        'l_23': 'encoder.23.1.layer_norm',
                        'l_24': 'encoder.23.1.DenseReluDense.wi',
                        'l_25': 'encoder.23.1.DenseReluDense.dropout',
                        'l_26': 'encoder.23.1.DenseReluDense.wo',
                        'l_27': 'encoder.23.1.dropout',
                        'l_28': 'encoder.final_layer_norm',
                        'l_29': 'encoder.dropout',
                        'l_30': 'decoder.dropout',
                        'l_31': 'decoder.0.0.layer_norm',
                        'l_32': 'decoder.0.0.SelfAttention.q',
                        'l_33': 'decoder.0.0.SelfAttention.k',
                        'l_34': 'decoder.0.0.SelfAttention.v',
                        'l_35': 'decoder.0.0.SelfAttention.relative_attention_bias',
                        'l_36': 'decoder.0.0.SelfAttention.dropout',
                        'l_37': 'decoder.0.0.SelfAttention.o',
                        'l_38': 'decoder.0.0.dropout',
                        'l_39': 'decoder.0.1.layer_norm',
                        'l_40': 'decoder.0.1.EncDecAttention.q',
                        'l_41': 'decoder.0.1.EncDecAttention.k',
                        'l_42': 'decoder.0.1.EncDecAttention.v',
                        'l_43': 'decoder.0.1.EncDecAttention.relative_attention_bias',
                        'l_44': 'decoder.0.1.EncDecAttention.dropout',
                        'l_45': 'decoder.0.1.EncDecAttention.o',
                        'l_46': 'decoder.0.1.dropout',
                        'l_47': 'decoder.0.2.layer_norm'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerFF[1]/Dropout[dropout] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[22]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[22]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[22]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[22]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[22]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[22]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[22]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[22]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[22]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[22]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[22]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[22]/T5LayerFF[1]/Dropout[dropout] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[23]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[23]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[23]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[23]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[23]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[23]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[23]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[23]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[23]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[23]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[23]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[23]/T5LayerFF[1]/Dropout[dropout] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[encoder]/T5LayerNorm[final_layer_norm] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[decoder]/Dropout[dropout] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Embedding[relative_attention_bias] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_36
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_37
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_38
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_39
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_40
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_41
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_42
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Embedding[relative_attention_bias] <=> self.l_43
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_44
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_45
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_46
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_47
        # input1 <=> decoder_attention_mask
        # input4 <=> inverted_encoder_attention_mask
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___94 <=> x0
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerSelfAttention[0]/Tensor::__add___1352 <=> x1
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[21]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> x2
        # T5ForConditionalGeneration/T5Stack[decoder]/StatelessEmbedding[embed_tokens] <=> x3
        decoder_attention_mask, inverted_encoder_attention_mask, x0, x1, x2, x3 = unflatten(args, self.input_structure)
        t_0 = self.l_0(x2)
        t_0 = torch.nn.functional.relu(t_0, inplace=False)
        t_0 = self.l_1(t_0)
        t_0 = self.l_2(t_0)
        t_0 = self.l_3(t_0)
        t_0 = x1 + t_0
        t_1 = self.l_4(t_0)
        t_2 = t_1.size()
        t_3 = self.l_5(t_1)
        t_4 = self.l_6(t_1)
        t_1 = self.l_7(t_1)
        t_2 = t_2[0]
        t_3 = t_3.view(t_2, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_4 = t_4.view(t_2, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_1 = t_1.view(t_2, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_3, t_4)
        t_4 += x0
        t_3 = t_4.float()
        t_3 = torch.nn.functional.softmax(t_3, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_3.type_as(t_4)
        t_4 = self.l_8(t_4)
        t_1 = torch.matmul(t_4, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_2 = t_1.view(t_2, -1, 4096)
        t_2 = self.l_9(t_2)
        t_2 = self.l_10(t_2)
        t_2 = t_0 + t_2
        t_0 = self.l_11(t_2)
        t_0 = self.l_12(t_0)
        t_0 = torch.nn.functional.relu(t_0, inplace=False)
        t_0 = self.l_13(t_0)
        t_0 = self.l_14(t_0)
        t_0 = self.l_15(t_0)
        t_0 = t_2 + t_0
        t_2 = self.l_16(t_0)
        t_1 = t_2.size()
        t_4 = self.l_17(t_2)
        t_3 = self.l_18(t_2)
        t_2 = self.l_19(t_2)
        t_1 = t_1[0]
        t_4 = t_4.view(t_1, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_3 = t_3.view(t_1, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_2 = t_2.view(t_1, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_3 = t_3.transpose(3, 2)
        t_3 = torch.matmul(t_4, t_3)
        t_3 += x0
        t_4 = t_3.float()
        t_4 = torch.nn.functional.softmax(t_4, dim=-1, _stacklevel=3, dtype=None)
        t_3 = t_4.type_as(t_3)
        t_3 = self.l_20(t_3)
        t_2 = torch.matmul(t_3, t_2)
        t_2 = t_2.transpose(1, 2)
        t_2 = t_2.contiguous()
        t_1 = t_2.view(t_1, -1, 4096)
        t_1 = self.l_21(t_1)
        t_1 = self.l_22(t_1)
        t_1 = t_0 + t_1
        t_0 = self.l_23(t_1)
        t_0 = self.l_24(t_0)
        t_0 = torch.nn.functional.relu(t_0, inplace=False)
        t_0 = self.l_25(t_0)
        t_0 = self.l_26(t_0)
        t_0 = self.l_27(t_0)
        t_0 = t_1 + t_0
        t_0 = self.l_28(t_0)
        t_0 = self.l_29(t_0)
        t_1 = self.l_41(t_0)
        t_2 = self.l_42(t_0)
        t_3 = self.l_30(x3)
        t_4 = self.l_31(t_3)
        t_5 = t_4.size()
        t_6 = self.l_32(t_4)
        t_7 = self.l_33(t_4)
        t_4 = self.l_34(t_4)
        t_8 = t_5[0]
        t_5 = t_5[1]
        t_6 = t_6.view(t_8, -1, 32, 128)
        t_6 = t_6.transpose(1, 2)
        t_7 = t_7.view(t_8, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_4 = t_4.view(t_8, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_7 = t_7.transpose(3, 2)
        t_7 = torch.matmul(t_6, t_7)
        t_6 = torch.arange(t_5, dtype=torch.int64, device=self.device)
        t_6 = t_6[(slice(None, None, None), None)]
        t_5 = torch.arange(t_5, dtype=torch.int64, device=self.device)
        t_5 = t_5[(None, slice(None, None, None))]
        t_6 = t_5 - t_6
        t_6 = -t_6
        t_5 = torch.zeros_like(t_6, device=self.device)
        t_5 = torch.max(t_6, t_5)
        t_6 = t_5.float()
        t_9 = torch.less(t_5, 16)
        t_6 = t_6 / 16
        t_6 = torch.log(t_6)
        t_10 = math.log(8.0)
        t_10 = t_6 / t_10
        t_10 = t_10 * 16
        t_10 = t_10.to(torch.int64)
        t_10 = 16 + t_10
        t_6 = torch.full_like(t_10, 31, device=self.device)
        t_6 = torch.min(t_10, t_6)
        t_6 = torch.where(t_9, t_5, t_6)
        t_6 = 0 + t_6
        t_6 = t_6.to(self.device)
        t_6 = self.l_35(t_6)
        t_6 = t_6.permute([2, 0, 1])
        t_6 = t_6.unsqueeze(0)
        t_6 = t_6 + decoder_attention_mask
        t_7 += t_6
        t_5 = t_7.float()
        t_5 = torch.nn.functional.softmax(t_5, dim=-1, _stacklevel=3, dtype=None)
        t_7 = t_5.type_as(t_7)
        t_7 = self.l_36(t_7)
        t_4 = torch.matmul(t_7, t_4)
        t_4 = t_4.transpose(1, 2)
        t_4 = t_4.contiguous()
        t_8 = t_4.view(t_8, -1, 4096)
        t_8 = self.l_37(t_8)
        t_8 = self.l_38(t_8)
        t_8 = t_3 + t_8
        t_3 = self.l_39(t_8)
        t_4 = t_3.size()
        t_3 = self.l_40(t_3)
        t_7 = t_4[0]
        t_4 = t_4[1]
        t_5 = t_0.size(1)
        t_3 = t_3.view(t_7, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_1 = t_1.view(t_7, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_2 = t_2.view(t_7, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_1 = t_1.transpose(3, 2)
        t_1 = torch.matmul(t_3, t_1)
        t_4 = torch.arange(t_4, dtype=torch.int64, device=self.device)
        t_4 = t_4[(slice(None, None, None), None)]
        t_5 = torch.arange(t_5, dtype=torch.int64, device=self.device)
        t_5 = t_5[(None, slice(None, None, None))]
        t_4 = t_5 - t_4
        t_4 = -t_4
        t_5 = torch.zeros_like(t_4, device=self.device)
        t_5 = torch.max(t_4, t_5)
        t_4 = t_5.float()
        t_3 = torch.less(t_5, 16)
        t_4 = t_4 / 16
        t_4 = torch.log(t_4)
        t_9 = math.log(8.0)
        t_9 = t_4 / t_9
        t_9 = t_9 * 16
        t_9 = t_9.to(torch.int64)
        t_9 = 16 + t_9
        t_4 = torch.full_like(t_9, 31, device=self.device)
        t_4 = torch.min(t_9, t_4)
        t_4 = torch.where(t_3, t_5, t_4)
        t_4 = 0 + t_4
        t_4 = t_4.to(self.device)
        t_4 = self.l_43(t_4)
        t_4 = t_4.permute([2, 0, 1])
        t_4 = t_4.unsqueeze(0)
        t_4 = t_4 + inverted_encoder_attention_mask
        t_1 += t_4
        t_5 = t_1.float()
        t_5 = torch.nn.functional.softmax(t_5, dim=-1, _stacklevel=3, dtype=None)
        t_1 = t_5.type_as(t_1)
        t_1 = self.l_44(t_1)
        t_2 = torch.matmul(t_1, t_2)
        t_2 = t_2.transpose(1, 2)
        t_2 = t_2.contiguous()
        t_7 = t_2.view(t_7, -1, 4096)
        t_7 = self.l_45(t_7)
        t_7 = self.l_46(t_7)
        t_7 = t_8 + t_7
        t_8 = self.l_47(t_7)
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/Tensor::__add___1677
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerFF[2]/T5LayerNorm[layer_norm]
        return list(flatten((t_0, t_6, t_4, t_7, t_8)))

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
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
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
        self.input_structure = [1, 1, 1, 1, 1]
        self.lookup = {'l_0': 'decoder.0.2.DenseReluDense.wi',
                        'l_1': 'decoder.0.2.DenseReluDense.dropout',
                        'l_2': 'decoder.0.2.DenseReluDense.wo',
                        'l_3': 'decoder.0.2.dropout',
                        'l_4': 'decoder.1.0.layer_norm',
                        'l_5': 'decoder.1.0.SelfAttention.q',
                        'l_6': 'decoder.1.0.SelfAttention.k',
                        'l_7': 'decoder.1.0.SelfAttention.v',
                        'l_8': 'decoder.1.0.SelfAttention.dropout',
                        'l_9': 'decoder.1.0.SelfAttention.o',
                        'l_10': 'decoder.1.0.dropout',
                        'l_11': 'decoder.1.1.layer_norm',
                        'l_12': 'decoder.1.1.EncDecAttention.q',
                        'l_13': 'decoder.1.1.EncDecAttention.k',
                        'l_14': 'decoder.1.1.EncDecAttention.v',
                        'l_15': 'decoder.1.1.EncDecAttention.dropout',
                        'l_16': 'decoder.1.1.EncDecAttention.o',
                        'l_17': 'decoder.1.1.dropout',
                        'l_18': 'decoder.1.2.layer_norm',
                        'l_19': 'decoder.1.2.DenseReluDense.wi',
                        'l_20': 'decoder.1.2.DenseReluDense.dropout',
                        'l_21': 'decoder.1.2.DenseReluDense.wo',
                        'l_22': 'decoder.1.2.dropout',
                        'l_23': 'decoder.2.0.layer_norm',
                        'l_24': 'decoder.2.0.SelfAttention.q',
                        'l_25': 'decoder.2.0.SelfAttention.k',
                        'l_26': 'decoder.2.0.SelfAttention.v',
                        'l_27': 'decoder.2.0.SelfAttention.dropout',
                        'l_28': 'decoder.2.0.SelfAttention.o',
                        'l_29': 'decoder.2.0.dropout',
                        'l_30': 'decoder.2.1.layer_norm',
                        'l_31': 'decoder.2.1.EncDecAttention.q',
                        'l_32': 'decoder.2.1.EncDecAttention.k',
                        'l_33': 'decoder.2.1.EncDecAttention.v',
                        'l_34': 'decoder.2.1.EncDecAttention.dropout',
                        'l_35': 'decoder.2.1.EncDecAttention.o',
                        'l_36': 'decoder.2.1.dropout',
                        'l_37': 'decoder.2.2.layer_norm',
                        'l_38': 'decoder.2.2.DenseReluDense.wi',
                        'l_39': 'decoder.2.2.DenseReluDense.dropout',
                        'l_40': 'decoder.2.2.DenseReluDense.wo',
                        'l_41': 'decoder.2.2.dropout',
                        'l_42': 'decoder.3.0.layer_norm',
                        'l_43': 'decoder.3.0.SelfAttention.q',
                        'l_44': 'decoder.3.0.SelfAttention.k',
                        'l_45': 'decoder.3.0.SelfAttention.v',
                        'l_46': 'decoder.3.0.SelfAttention.dropout',
                        'l_47': 'decoder.3.0.SelfAttention.o',
                        'l_48': 'decoder.3.0.dropout',
                        'l_49': 'decoder.3.1.layer_norm',
                        'l_50': 'decoder.3.1.EncDecAttention.q',
                        'l_51': 'decoder.3.1.EncDecAttention.k',
                        'l_52': 'decoder.3.1.EncDecAttention.v',
                        'l_53': 'decoder.3.1.EncDecAttention.dropout',
                        'l_54': 'decoder.3.1.EncDecAttention.o',
                        'l_55': 'decoder.3.1.dropout',
                        'l_56': 'decoder.3.2.layer_norm'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerFF[2]/Dropout[dropout] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerFF[2]/Dropout[dropout] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_36
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_37
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_38
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_39
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_40
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerFF[2]/Dropout[dropout] <=> self.l_41
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_42
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_43
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_44
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_45
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_46
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_47
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_48
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_49
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_50
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_51
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_52
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_53
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_54
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_55
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_56
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout] <=> x0
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562 <=> x1
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658 <=> x2
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/Tensor::__add___1677 <=> x3
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> x4
        x0, x1, x2, x3, x4 = unflatten(args, self.input_structure)
        t_0 = self.l_13(x0)
        t_1 = self.l_14(x0)
        t_2 = self.l_32(x0)
        t_3 = self.l_33(x0)
        t_4 = self.l_51(x0)
        t_5 = self.l_52(x0)
        t_6 = self.l_0(x4)
        t_6 = torch.nn.functional.relu(t_6, inplace=False)
        t_6 = self.l_1(t_6)
        t_6 = self.l_2(t_6)
        t_6 = self.l_3(t_6)
        t_6 = x3 + t_6
        t_7 = self.l_4(t_6)
        t_8 = t_7.size()
        t_9 = self.l_5(t_7)
        t_10 = self.l_6(t_7)
        t_7 = self.l_7(t_7)
        t_8 = t_8[0]
        t_9 = t_9.view(t_8, -1, 32, 128)
        t_9 = t_9.transpose(1, 2)
        t_10 = t_10.view(t_8, -1, 32, 128)
        t_10 = t_10.transpose(1, 2)
        t_7 = t_7.view(t_8, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_10 = t_10.transpose(3, 2)
        t_10 = torch.matmul(t_9, t_10)
        t_10 += x1
        t_9 = t_10.float()
        t_9 = torch.nn.functional.softmax(t_9, dim=-1, _stacklevel=3, dtype=None)
        t_10 = t_9.type_as(t_10)
        t_10 = self.l_8(t_10)
        t_7 = torch.matmul(t_10, t_7)
        t_7 = t_7.transpose(1, 2)
        t_7 = t_7.contiguous()
        t_8 = t_7.view(t_8, -1, 4096)
        t_8 = self.l_9(t_8)
        t_8 = self.l_10(t_8)
        t_8 = t_6 + t_8
        t_6 = self.l_11(t_8)
        t_7 = t_6.size()
        t_6 = self.l_12(t_6)
        t_7 = t_7[0]
        t_6 = t_6.view(t_7, -1, 32, 128)
        t_6 = t_6.transpose(1, 2)
        t_0 = t_0.view(t_7, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_1 = t_1.view(t_7, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_0 = t_0.transpose(3, 2)
        t_0 = torch.matmul(t_6, t_0)
        t_0 += x2
        t_6 = t_0.float()
        t_6 = torch.nn.functional.softmax(t_6, dim=-1, _stacklevel=3, dtype=None)
        t_0 = t_6.type_as(t_0)
        t_0 = self.l_15(t_0)
        t_1 = torch.matmul(t_0, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_7 = t_1.view(t_7, -1, 4096)
        t_7 = self.l_16(t_7)
        t_7 = self.l_17(t_7)
        t_7 = t_8 + t_7
        t_8 = self.l_18(t_7)
        t_8 = self.l_19(t_8)
        t_8 = torch.nn.functional.relu(t_8, inplace=False)
        t_8 = self.l_20(t_8)
        t_8 = self.l_21(t_8)
        t_8 = self.l_22(t_8)
        t_8 = t_7 + t_8
        t_7 = self.l_23(t_8)
        t_1 = t_7.size()
        t_0 = self.l_24(t_7)
        t_6 = self.l_25(t_7)
        t_7 = self.l_26(t_7)
        t_1 = t_1[0]
        t_0 = t_0.view(t_1, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_6 = t_6.view(t_1, -1, 32, 128)
        t_6 = t_6.transpose(1, 2)
        t_7 = t_7.view(t_1, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_6 = t_6.transpose(3, 2)
        t_6 = torch.matmul(t_0, t_6)
        t_6 += x1
        t_0 = t_6.float()
        t_0 = torch.nn.functional.softmax(t_0, dim=-1, _stacklevel=3, dtype=None)
        t_6 = t_0.type_as(t_6)
        t_6 = self.l_27(t_6)
        t_7 = torch.matmul(t_6, t_7)
        t_7 = t_7.transpose(1, 2)
        t_7 = t_7.contiguous()
        t_1 = t_7.view(t_1, -1, 4096)
        t_1 = self.l_28(t_1)
        t_1 = self.l_29(t_1)
        t_1 = t_8 + t_1
        t_8 = self.l_30(t_1)
        t_7 = t_8.size()
        t_8 = self.l_31(t_8)
        t_7 = t_7[0]
        t_8 = t_8.view(t_7, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_2 = t_2.view(t_7, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_3 = t_3.view(t_7, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_2 = t_2.transpose(3, 2)
        t_2 = torch.matmul(t_8, t_2)
        t_2 += x2
        t_8 = t_2.float()
        t_8 = torch.nn.functional.softmax(t_8, dim=-1, _stacklevel=3, dtype=None)
        t_2 = t_8.type_as(t_2)
        t_2 = self.l_34(t_2)
        t_3 = torch.matmul(t_2, t_3)
        t_3 = t_3.transpose(1, 2)
        t_3 = t_3.contiguous()
        t_7 = t_3.view(t_7, -1, 4096)
        t_7 = self.l_35(t_7)
        t_7 = self.l_36(t_7)
        t_7 = t_1 + t_7
        t_1 = self.l_37(t_7)
        t_1 = self.l_38(t_1)
        t_1 = torch.nn.functional.relu(t_1, inplace=False)
        t_1 = self.l_39(t_1)
        t_1 = self.l_40(t_1)
        t_1 = self.l_41(t_1)
        t_1 = t_7 + t_1
        t_7 = self.l_42(t_1)
        t_3 = t_7.size()
        t_2 = self.l_43(t_7)
        t_8 = self.l_44(t_7)
        t_7 = self.l_45(t_7)
        t_3 = t_3[0]
        t_2 = t_2.view(t_3, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_8 = t_8.view(t_3, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_7 = t_7.view(t_3, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_8 = t_8.transpose(3, 2)
        t_8 = torch.matmul(t_2, t_8)
        t_8 += x1
        t_2 = t_8.float()
        t_2 = torch.nn.functional.softmax(t_2, dim=-1, _stacklevel=3, dtype=None)
        t_8 = t_2.type_as(t_8)
        t_8 = self.l_46(t_8)
        t_7 = torch.matmul(t_8, t_7)
        t_7 = t_7.transpose(1, 2)
        t_7 = t_7.contiguous()
        t_3 = t_7.view(t_3, -1, 4096)
        t_3 = self.l_47(t_3)
        t_3 = self.l_48(t_3)
        t_3 = t_1 + t_3
        t_1 = self.l_49(t_3)
        t_7 = t_1.size()
        t_1 = self.l_50(t_1)
        t_7 = t_7[0]
        t_1 = t_1.view(t_7, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_4 = t_4.view(t_7, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_5 = t_5.view(t_7, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_1, t_4)
        t_4 += x2
        t_1 = t_4.float()
        t_1 = torch.nn.functional.softmax(t_1, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_1.type_as(t_4)
        t_4 = self.l_53(t_4)
        t_5 = torch.matmul(t_4, t_5)
        t_5 = t_5.transpose(1, 2)
        t_5 = t_5.contiguous()
        t_7 = t_5.view(t_7, -1, 4096)
        t_7 = self.l_54(t_7)
        t_7 = self.l_55(t_7)
        t_7 = t_3 + t_7
        t_3 = self.l_56(t_7)
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerCrossAttention[1]/Tensor::__add___2007
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerFF[2]/T5LayerNorm[layer_norm]
        return list(flatten((x0, x1, x2, t_7, t_3)))

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
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
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
        self.input_structure = [1, 1, 1, 1, 1]
        self.lookup = {'l_0': 'decoder.3.2.DenseReluDense.wi',
                        'l_1': 'decoder.3.2.DenseReluDense.dropout',
                        'l_2': 'decoder.3.2.DenseReluDense.wo',
                        'l_3': 'decoder.3.2.dropout',
                        'l_4': 'decoder.4.0.layer_norm',
                        'l_5': 'decoder.4.0.SelfAttention.q',
                        'l_6': 'decoder.4.0.SelfAttention.k',
                        'l_7': 'decoder.4.0.SelfAttention.v',
                        'l_8': 'decoder.4.0.SelfAttention.dropout',
                        'l_9': 'decoder.4.0.SelfAttention.o',
                        'l_10': 'decoder.4.0.dropout',
                        'l_11': 'decoder.4.1.layer_norm',
                        'l_12': 'decoder.4.1.EncDecAttention.q',
                        'l_13': 'decoder.4.1.EncDecAttention.k',
                        'l_14': 'decoder.4.1.EncDecAttention.v',
                        'l_15': 'decoder.4.1.EncDecAttention.dropout',
                        'l_16': 'decoder.4.1.EncDecAttention.o',
                        'l_17': 'decoder.4.1.dropout',
                        'l_18': 'decoder.4.2.layer_norm',
                        'l_19': 'decoder.4.2.DenseReluDense.wi',
                        'l_20': 'decoder.4.2.DenseReluDense.dropout',
                        'l_21': 'decoder.4.2.DenseReluDense.wo',
                        'l_22': 'decoder.4.2.dropout',
                        'l_23': 'decoder.5.0.layer_norm',
                        'l_24': 'decoder.5.0.SelfAttention.q',
                        'l_25': 'decoder.5.0.SelfAttention.k',
                        'l_26': 'decoder.5.0.SelfAttention.v',
                        'l_27': 'decoder.5.0.SelfAttention.dropout',
                        'l_28': 'decoder.5.0.SelfAttention.o',
                        'l_29': 'decoder.5.0.dropout',
                        'l_30': 'decoder.5.1.layer_norm',
                        'l_31': 'decoder.5.1.EncDecAttention.q',
                        'l_32': 'decoder.5.1.EncDecAttention.k',
                        'l_33': 'decoder.5.1.EncDecAttention.v',
                        'l_34': 'decoder.5.1.EncDecAttention.dropout',
                        'l_35': 'decoder.5.1.EncDecAttention.o',
                        'l_36': 'decoder.5.1.dropout',
                        'l_37': 'decoder.5.2.layer_norm',
                        'l_38': 'decoder.5.2.DenseReluDense.wi',
                        'l_39': 'decoder.5.2.DenseReluDense.dropout',
                        'l_40': 'decoder.5.2.DenseReluDense.wo',
                        'l_41': 'decoder.5.2.dropout',
                        'l_42': 'decoder.6.0.layer_norm',
                        'l_43': 'decoder.6.0.SelfAttention.q',
                        'l_44': 'decoder.6.0.SelfAttention.k',
                        'l_45': 'decoder.6.0.SelfAttention.v',
                        'l_46': 'decoder.6.0.SelfAttention.dropout',
                        'l_47': 'decoder.6.0.SelfAttention.o',
                        'l_48': 'decoder.6.0.dropout',
                        'l_49': 'decoder.6.1.layer_norm',
                        'l_50': 'decoder.6.1.EncDecAttention.q',
                        'l_51': 'decoder.6.1.EncDecAttention.k',
                        'l_52': 'decoder.6.1.EncDecAttention.v',
                        'l_53': 'decoder.6.1.EncDecAttention.dropout',
                        'l_54': 'decoder.6.1.EncDecAttention.o',
                        'l_55': 'decoder.6.1.dropout',
                        'l_56': 'decoder.6.2.layer_norm'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerFF[2]/Dropout[dropout] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerFF[2]/Dropout[dropout] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_36
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_37
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_38
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_39
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_40
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerFF[2]/Dropout[dropout] <=> self.l_41
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_42
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_43
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_44
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_45
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_46
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_47
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_48
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_49
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_50
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_51
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_52
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_53
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_54
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_55
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_56
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout] <=> x0
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562 <=> x1
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658 <=> x2
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerCrossAttention[1]/Tensor::__add___2007 <=> x3
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> x4
        x0, x1, x2, x3, x4 = unflatten(args, self.input_structure)
        t_0 = self.l_13(x0)
        t_1 = self.l_14(x0)
        t_2 = self.l_32(x0)
        t_3 = self.l_33(x0)
        t_4 = self.l_51(x0)
        t_5 = self.l_52(x0)
        t_6 = self.l_0(x4)
        t_6 = torch.nn.functional.relu(t_6, inplace=False)
        t_6 = self.l_1(t_6)
        t_6 = self.l_2(t_6)
        t_6 = self.l_3(t_6)
        t_6 = x3 + t_6
        t_7 = self.l_4(t_6)
        t_8 = t_7.size()
        t_9 = self.l_5(t_7)
        t_10 = self.l_6(t_7)
        t_7 = self.l_7(t_7)
        t_8 = t_8[0]
        t_9 = t_9.view(t_8, -1, 32, 128)
        t_9 = t_9.transpose(1, 2)
        t_10 = t_10.view(t_8, -1, 32, 128)
        t_10 = t_10.transpose(1, 2)
        t_7 = t_7.view(t_8, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_10 = t_10.transpose(3, 2)
        t_10 = torch.matmul(t_9, t_10)
        t_10 += x1
        t_9 = t_10.float()
        t_9 = torch.nn.functional.softmax(t_9, dim=-1, _stacklevel=3, dtype=None)
        t_10 = t_9.type_as(t_10)
        t_10 = self.l_8(t_10)
        t_7 = torch.matmul(t_10, t_7)
        t_7 = t_7.transpose(1, 2)
        t_7 = t_7.contiguous()
        t_8 = t_7.view(t_8, -1, 4096)
        t_8 = self.l_9(t_8)
        t_8 = self.l_10(t_8)
        t_8 = t_6 + t_8
        t_6 = self.l_11(t_8)
        t_7 = t_6.size()
        t_6 = self.l_12(t_6)
        t_7 = t_7[0]
        t_6 = t_6.view(t_7, -1, 32, 128)
        t_6 = t_6.transpose(1, 2)
        t_0 = t_0.view(t_7, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_1 = t_1.view(t_7, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_0 = t_0.transpose(3, 2)
        t_0 = torch.matmul(t_6, t_0)
        t_0 += x2
        t_6 = t_0.float()
        t_6 = torch.nn.functional.softmax(t_6, dim=-1, _stacklevel=3, dtype=None)
        t_0 = t_6.type_as(t_0)
        t_0 = self.l_15(t_0)
        t_1 = torch.matmul(t_0, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_7 = t_1.view(t_7, -1, 4096)
        t_7 = self.l_16(t_7)
        t_7 = self.l_17(t_7)
        t_7 = t_8 + t_7
        t_8 = self.l_18(t_7)
        t_8 = self.l_19(t_8)
        t_8 = torch.nn.functional.relu(t_8, inplace=False)
        t_8 = self.l_20(t_8)
        t_8 = self.l_21(t_8)
        t_8 = self.l_22(t_8)
        t_8 = t_7 + t_8
        t_7 = self.l_23(t_8)
        t_1 = t_7.size()
        t_0 = self.l_24(t_7)
        t_6 = self.l_25(t_7)
        t_7 = self.l_26(t_7)
        t_1 = t_1[0]
        t_0 = t_0.view(t_1, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_6 = t_6.view(t_1, -1, 32, 128)
        t_6 = t_6.transpose(1, 2)
        t_7 = t_7.view(t_1, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_6 = t_6.transpose(3, 2)
        t_6 = torch.matmul(t_0, t_6)
        t_6 += x1
        t_0 = t_6.float()
        t_0 = torch.nn.functional.softmax(t_0, dim=-1, _stacklevel=3, dtype=None)
        t_6 = t_0.type_as(t_6)
        t_6 = self.l_27(t_6)
        t_7 = torch.matmul(t_6, t_7)
        t_7 = t_7.transpose(1, 2)
        t_7 = t_7.contiguous()
        t_1 = t_7.view(t_1, -1, 4096)
        t_1 = self.l_28(t_1)
        t_1 = self.l_29(t_1)
        t_1 = t_8 + t_1
        t_8 = self.l_30(t_1)
        t_7 = t_8.size()
        t_8 = self.l_31(t_8)
        t_7 = t_7[0]
        t_8 = t_8.view(t_7, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_2 = t_2.view(t_7, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_3 = t_3.view(t_7, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_2 = t_2.transpose(3, 2)
        t_2 = torch.matmul(t_8, t_2)
        t_2 += x2
        t_8 = t_2.float()
        t_8 = torch.nn.functional.softmax(t_8, dim=-1, _stacklevel=3, dtype=None)
        t_2 = t_8.type_as(t_2)
        t_2 = self.l_34(t_2)
        t_3 = torch.matmul(t_2, t_3)
        t_3 = t_3.transpose(1, 2)
        t_3 = t_3.contiguous()
        t_7 = t_3.view(t_7, -1, 4096)
        t_7 = self.l_35(t_7)
        t_7 = self.l_36(t_7)
        t_7 = t_1 + t_7
        t_1 = self.l_37(t_7)
        t_1 = self.l_38(t_1)
        t_1 = torch.nn.functional.relu(t_1, inplace=False)
        t_1 = self.l_39(t_1)
        t_1 = self.l_40(t_1)
        t_1 = self.l_41(t_1)
        t_1 = t_7 + t_1
        t_7 = self.l_42(t_1)
        t_3 = t_7.size()
        t_2 = self.l_43(t_7)
        t_8 = self.l_44(t_7)
        t_7 = self.l_45(t_7)
        t_3 = t_3[0]
        t_2 = t_2.view(t_3, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_8 = t_8.view(t_3, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_7 = t_7.view(t_3, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_8 = t_8.transpose(3, 2)
        t_8 = torch.matmul(t_2, t_8)
        t_8 += x1
        t_2 = t_8.float()
        t_2 = torch.nn.functional.softmax(t_2, dim=-1, _stacklevel=3, dtype=None)
        t_8 = t_2.type_as(t_8)
        t_8 = self.l_46(t_8)
        t_7 = torch.matmul(t_8, t_7)
        t_7 = t_7.transpose(1, 2)
        t_7 = t_7.contiguous()
        t_3 = t_7.view(t_3, -1, 4096)
        t_3 = self.l_47(t_3)
        t_3 = self.l_48(t_3)
        t_3 = t_1 + t_3
        t_1 = self.l_49(t_3)
        t_7 = t_1.size()
        t_1 = self.l_50(t_1)
        t_7 = t_7[0]
        t_1 = t_1.view(t_7, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_4 = t_4.view(t_7, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_5 = t_5.view(t_7, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_1, t_4)
        t_4 += x2
        t_1 = t_4.float()
        t_1 = torch.nn.functional.softmax(t_1, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_1.type_as(t_4)
        t_4 = self.l_53(t_4)
        t_5 = torch.matmul(t_4, t_5)
        t_5 = t_5.transpose(1, 2)
        t_5 = t_5.contiguous()
        t_7 = t_5.view(t_7, -1, 4096)
        t_7 = self.l_54(t_7)
        t_7 = self.l_55(t_7)
        t_7 = t_3 + t_7
        t_3 = self.l_56(t_7)
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerCrossAttention[1]/Tensor::__add___2337
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerFF[2]/T5LayerNorm[layer_norm]
        return list(flatten((x0, x1, x2, t_7, t_3)))

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
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
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
        self.input_structure = [1, 1, 1, 1, 1]
        self.lookup = {'l_0': 'decoder.6.2.DenseReluDense.wi',
                        'l_1': 'decoder.6.2.DenseReluDense.dropout',
                        'l_2': 'decoder.6.2.DenseReluDense.wo',
                        'l_3': 'decoder.6.2.dropout',
                        'l_4': 'decoder.7.0.layer_norm',
                        'l_5': 'decoder.7.0.SelfAttention.q',
                        'l_6': 'decoder.7.0.SelfAttention.k',
                        'l_7': 'decoder.7.0.SelfAttention.v',
                        'l_8': 'decoder.7.0.SelfAttention.dropout',
                        'l_9': 'decoder.7.0.SelfAttention.o',
                        'l_10': 'decoder.7.0.dropout',
                        'l_11': 'decoder.7.1.layer_norm',
                        'l_12': 'decoder.7.1.EncDecAttention.q',
                        'l_13': 'decoder.7.1.EncDecAttention.k',
                        'l_14': 'decoder.7.1.EncDecAttention.v',
                        'l_15': 'decoder.7.1.EncDecAttention.dropout',
                        'l_16': 'decoder.7.1.EncDecAttention.o',
                        'l_17': 'decoder.7.1.dropout',
                        'l_18': 'decoder.7.2.layer_norm',
                        'l_19': 'decoder.7.2.DenseReluDense.wi',
                        'l_20': 'decoder.7.2.DenseReluDense.dropout',
                        'l_21': 'decoder.7.2.DenseReluDense.wo',
                        'l_22': 'decoder.7.2.dropout',
                        'l_23': 'decoder.8.0.layer_norm',
                        'l_24': 'decoder.8.0.SelfAttention.q',
                        'l_25': 'decoder.8.0.SelfAttention.k',
                        'l_26': 'decoder.8.0.SelfAttention.v',
                        'l_27': 'decoder.8.0.SelfAttention.dropout',
                        'l_28': 'decoder.8.0.SelfAttention.o',
                        'l_29': 'decoder.8.0.dropout',
                        'l_30': 'decoder.8.1.layer_norm',
                        'l_31': 'decoder.8.1.EncDecAttention.q',
                        'l_32': 'decoder.8.1.EncDecAttention.k',
                        'l_33': 'decoder.8.1.EncDecAttention.v',
                        'l_34': 'decoder.8.1.EncDecAttention.dropout',
                        'l_35': 'decoder.8.1.EncDecAttention.o',
                        'l_36': 'decoder.8.1.dropout',
                        'l_37': 'decoder.8.2.layer_norm',
                        'l_38': 'decoder.8.2.DenseReluDense.wi',
                        'l_39': 'decoder.8.2.DenseReluDense.dropout',
                        'l_40': 'decoder.8.2.DenseReluDense.wo',
                        'l_41': 'decoder.8.2.dropout',
                        'l_42': 'decoder.9.0.layer_norm',
                        'l_43': 'decoder.9.0.SelfAttention.q',
                        'l_44': 'decoder.9.0.SelfAttention.k',
                        'l_45': 'decoder.9.0.SelfAttention.v',
                        'l_46': 'decoder.9.0.SelfAttention.dropout',
                        'l_47': 'decoder.9.0.SelfAttention.o',
                        'l_48': 'decoder.9.0.dropout',
                        'l_49': 'decoder.9.1.layer_norm',
                        'l_50': 'decoder.9.1.EncDecAttention.q',
                        'l_51': 'decoder.9.1.EncDecAttention.k',
                        'l_52': 'decoder.9.1.EncDecAttention.v',
                        'l_53': 'decoder.9.1.EncDecAttention.dropout',
                        'l_54': 'decoder.9.1.EncDecAttention.o',
                        'l_55': 'decoder.9.1.dropout',
                        'l_56': 'decoder.9.2.layer_norm'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerFF[2]/Dropout[dropout] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[7]/T5LayerFF[2]/Dropout[dropout] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_36
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_37
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_38
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_39
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_40
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[8]/T5LayerFF[2]/Dropout[dropout] <=> self.l_41
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_42
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_43
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_44
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_45
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_46
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_47
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_48
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_49
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_50
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_51
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_52
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_53
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_54
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_55
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_56
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout] <=> x0
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562 <=> x1
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658 <=> x2
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerCrossAttention[1]/Tensor::__add___2337 <=> x3
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[6]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> x4
        x0, x1, x2, x3, x4 = unflatten(args, self.input_structure)
        t_0 = self.l_13(x0)
        t_1 = self.l_14(x0)
        t_2 = self.l_32(x0)
        t_3 = self.l_33(x0)
        t_4 = self.l_51(x0)
        t_5 = self.l_52(x0)
        t_6 = self.l_0(x4)
        t_6 = torch.nn.functional.relu(t_6, inplace=False)
        t_6 = self.l_1(t_6)
        t_6 = self.l_2(t_6)
        t_6 = self.l_3(t_6)
        t_6 = x3 + t_6
        t_7 = self.l_4(t_6)
        t_8 = t_7.size()
        t_9 = self.l_5(t_7)
        t_10 = self.l_6(t_7)
        t_7 = self.l_7(t_7)
        t_8 = t_8[0]
        t_9 = t_9.view(t_8, -1, 32, 128)
        t_9 = t_9.transpose(1, 2)
        t_10 = t_10.view(t_8, -1, 32, 128)
        t_10 = t_10.transpose(1, 2)
        t_7 = t_7.view(t_8, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_10 = t_10.transpose(3, 2)
        t_10 = torch.matmul(t_9, t_10)
        t_10 += x1
        t_9 = t_10.float()
        t_9 = torch.nn.functional.softmax(t_9, dim=-1, _stacklevel=3, dtype=None)
        t_10 = t_9.type_as(t_10)
        t_10 = self.l_8(t_10)
        t_7 = torch.matmul(t_10, t_7)
        t_7 = t_7.transpose(1, 2)
        t_7 = t_7.contiguous()
        t_8 = t_7.view(t_8, -1, 4096)
        t_8 = self.l_9(t_8)
        t_8 = self.l_10(t_8)
        t_8 = t_6 + t_8
        t_6 = self.l_11(t_8)
        t_7 = t_6.size()
        t_6 = self.l_12(t_6)
        t_7 = t_7[0]
        t_6 = t_6.view(t_7, -1, 32, 128)
        t_6 = t_6.transpose(1, 2)
        t_0 = t_0.view(t_7, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_1 = t_1.view(t_7, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_0 = t_0.transpose(3, 2)
        t_0 = torch.matmul(t_6, t_0)
        t_0 += x2
        t_6 = t_0.float()
        t_6 = torch.nn.functional.softmax(t_6, dim=-1, _stacklevel=3, dtype=None)
        t_0 = t_6.type_as(t_0)
        t_0 = self.l_15(t_0)
        t_1 = torch.matmul(t_0, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_7 = t_1.view(t_7, -1, 4096)
        t_7 = self.l_16(t_7)
        t_7 = self.l_17(t_7)
        t_7 = t_8 + t_7
        t_8 = self.l_18(t_7)
        t_8 = self.l_19(t_8)
        t_8 = torch.nn.functional.relu(t_8, inplace=False)
        t_8 = self.l_20(t_8)
        t_8 = self.l_21(t_8)
        t_8 = self.l_22(t_8)
        t_8 = t_7 + t_8
        t_7 = self.l_23(t_8)
        t_1 = t_7.size()
        t_0 = self.l_24(t_7)
        t_6 = self.l_25(t_7)
        t_7 = self.l_26(t_7)
        t_1 = t_1[0]
        t_0 = t_0.view(t_1, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_6 = t_6.view(t_1, -1, 32, 128)
        t_6 = t_6.transpose(1, 2)
        t_7 = t_7.view(t_1, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_6 = t_6.transpose(3, 2)
        t_6 = torch.matmul(t_0, t_6)
        t_6 += x1
        t_0 = t_6.float()
        t_0 = torch.nn.functional.softmax(t_0, dim=-1, _stacklevel=3, dtype=None)
        t_6 = t_0.type_as(t_6)
        t_6 = self.l_27(t_6)
        t_7 = torch.matmul(t_6, t_7)
        t_7 = t_7.transpose(1, 2)
        t_7 = t_7.contiguous()
        t_1 = t_7.view(t_1, -1, 4096)
        t_1 = self.l_28(t_1)
        t_1 = self.l_29(t_1)
        t_1 = t_8 + t_1
        t_8 = self.l_30(t_1)
        t_7 = t_8.size()
        t_8 = self.l_31(t_8)
        t_7 = t_7[0]
        t_8 = t_8.view(t_7, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_2 = t_2.view(t_7, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_3 = t_3.view(t_7, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_2 = t_2.transpose(3, 2)
        t_2 = torch.matmul(t_8, t_2)
        t_2 += x2
        t_8 = t_2.float()
        t_8 = torch.nn.functional.softmax(t_8, dim=-1, _stacklevel=3, dtype=None)
        t_2 = t_8.type_as(t_2)
        t_2 = self.l_34(t_2)
        t_3 = torch.matmul(t_2, t_3)
        t_3 = t_3.transpose(1, 2)
        t_3 = t_3.contiguous()
        t_7 = t_3.view(t_7, -1, 4096)
        t_7 = self.l_35(t_7)
        t_7 = self.l_36(t_7)
        t_7 = t_1 + t_7
        t_1 = self.l_37(t_7)
        t_1 = self.l_38(t_1)
        t_1 = torch.nn.functional.relu(t_1, inplace=False)
        t_1 = self.l_39(t_1)
        t_1 = self.l_40(t_1)
        t_1 = self.l_41(t_1)
        t_1 = t_7 + t_1
        t_7 = self.l_42(t_1)
        t_3 = t_7.size()
        t_2 = self.l_43(t_7)
        t_8 = self.l_44(t_7)
        t_7 = self.l_45(t_7)
        t_3 = t_3[0]
        t_2 = t_2.view(t_3, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_8 = t_8.view(t_3, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_7 = t_7.view(t_3, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_8 = t_8.transpose(3, 2)
        t_8 = torch.matmul(t_2, t_8)
        t_8 += x1
        t_2 = t_8.float()
        t_2 = torch.nn.functional.softmax(t_2, dim=-1, _stacklevel=3, dtype=None)
        t_8 = t_2.type_as(t_8)
        t_8 = self.l_46(t_8)
        t_7 = torch.matmul(t_8, t_7)
        t_7 = t_7.transpose(1, 2)
        t_7 = t_7.contiguous()
        t_3 = t_7.view(t_3, -1, 4096)
        t_3 = self.l_47(t_3)
        t_3 = self.l_48(t_3)
        t_3 = t_1 + t_3
        t_1 = self.l_49(t_3)
        t_7 = t_1.size()
        t_1 = self.l_50(t_1)
        t_7 = t_7[0]
        t_1 = t_1.view(t_7, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_4 = t_4.view(t_7, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_5 = t_5.view(t_7, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_1, t_4)
        t_4 += x2
        t_1 = t_4.float()
        t_1 = torch.nn.functional.softmax(t_1, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_1.type_as(t_4)
        t_4 = self.l_53(t_4)
        t_5 = torch.matmul(t_4, t_5)
        t_5 = t_5.transpose(1, 2)
        t_5 = t_5.contiguous()
        t_7 = t_5.view(t_7, -1, 4096)
        t_7 = self.l_54(t_7)
        t_7 = self.l_55(t_7)
        t_7 = t_3 + t_7
        t_3 = self.l_56(t_7)
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerCrossAttention[1]/Tensor::__add___2667
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerFF[2]/T5LayerNorm[layer_norm]
        return list(flatten((x0, x1, x2, t_7, t_3)))

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
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
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
        self.input_structure = [1, 1, 1, 1, 1]
        self.lookup = {'l_0': 'decoder.9.2.DenseReluDense.wi',
                        'l_1': 'decoder.9.2.DenseReluDense.dropout',
                        'l_2': 'decoder.9.2.DenseReluDense.wo',
                        'l_3': 'decoder.9.2.dropout',
                        'l_4': 'decoder.10.0.layer_norm',
                        'l_5': 'decoder.10.0.SelfAttention.q',
                        'l_6': 'decoder.10.0.SelfAttention.k',
                        'l_7': 'decoder.10.0.SelfAttention.v',
                        'l_8': 'decoder.10.0.SelfAttention.dropout',
                        'l_9': 'decoder.10.0.SelfAttention.o',
                        'l_10': 'decoder.10.0.dropout',
                        'l_11': 'decoder.10.1.layer_norm',
                        'l_12': 'decoder.10.1.EncDecAttention.q',
                        'l_13': 'decoder.10.1.EncDecAttention.k',
                        'l_14': 'decoder.10.1.EncDecAttention.v',
                        'l_15': 'decoder.10.1.EncDecAttention.dropout',
                        'l_16': 'decoder.10.1.EncDecAttention.o',
                        'l_17': 'decoder.10.1.dropout',
                        'l_18': 'decoder.10.2.layer_norm',
                        'l_19': 'decoder.10.2.DenseReluDense.wi',
                        'l_20': 'decoder.10.2.DenseReluDense.dropout',
                        'l_21': 'decoder.10.2.DenseReluDense.wo',
                        'l_22': 'decoder.10.2.dropout',
                        'l_23': 'decoder.11.0.layer_norm',
                        'l_24': 'decoder.11.0.SelfAttention.q',
                        'l_25': 'decoder.11.0.SelfAttention.k',
                        'l_26': 'decoder.11.0.SelfAttention.v',
                        'l_27': 'decoder.11.0.SelfAttention.dropout',
                        'l_28': 'decoder.11.0.SelfAttention.o',
                        'l_29': 'decoder.11.0.dropout',
                        'l_30': 'decoder.11.1.layer_norm',
                        'l_31': 'decoder.11.1.EncDecAttention.q',
                        'l_32': 'decoder.11.1.EncDecAttention.k',
                        'l_33': 'decoder.11.1.EncDecAttention.v',
                        'l_34': 'decoder.11.1.EncDecAttention.dropout',
                        'l_35': 'decoder.11.1.EncDecAttention.o',
                        'l_36': 'decoder.11.1.dropout',
                        'l_37': 'decoder.11.2.layer_norm',
                        'l_38': 'decoder.11.2.DenseReluDense.wi',
                        'l_39': 'decoder.11.2.DenseReluDense.dropout',
                        'l_40': 'decoder.11.2.DenseReluDense.wo',
                        'l_41': 'decoder.11.2.dropout',
                        'l_42': 'decoder.12.0.layer_norm',
                        'l_43': 'decoder.12.0.SelfAttention.q',
                        'l_44': 'decoder.12.0.SelfAttention.k',
                        'l_45': 'decoder.12.0.SelfAttention.v',
                        'l_46': 'decoder.12.0.SelfAttention.dropout',
                        'l_47': 'decoder.12.0.SelfAttention.o',
                        'l_48': 'decoder.12.0.dropout',
                        'l_49': 'decoder.12.1.layer_norm',
                        'l_50': 'decoder.12.1.EncDecAttention.q',
                        'l_51': 'decoder.12.1.EncDecAttention.k',
                        'l_52': 'decoder.12.1.EncDecAttention.v',
                        'l_53': 'decoder.12.1.EncDecAttention.dropout',
                        'l_54': 'decoder.12.1.EncDecAttention.o',
                        'l_55': 'decoder.12.1.dropout',
                        'l_56': 'decoder.12.2.layer_norm'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerFF[2]/Dropout[dropout] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[10]/T5LayerFF[2]/Dropout[dropout] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_36
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_37
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_38
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_39
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_40
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[11]/T5LayerFF[2]/Dropout[dropout] <=> self.l_41
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_42
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_43
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_44
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_45
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_46
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_47
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_48
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_49
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_50
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_51
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_52
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_53
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_54
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_55
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_56
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout] <=> x0
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562 <=> x1
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658 <=> x2
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerCrossAttention[1]/Tensor::__add___2667 <=> x3
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[9]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> x4
        x0, x1, x2, x3, x4 = unflatten(args, self.input_structure)
        t_0 = self.l_13(x0)
        t_1 = self.l_14(x0)
        t_2 = self.l_32(x0)
        t_3 = self.l_33(x0)
        t_4 = self.l_51(x0)
        t_5 = self.l_52(x0)
        t_6 = self.l_0(x4)
        t_6 = torch.nn.functional.relu(t_6, inplace=False)
        t_6 = self.l_1(t_6)
        t_6 = self.l_2(t_6)
        t_6 = self.l_3(t_6)
        t_6 = x3 + t_6
        t_7 = self.l_4(t_6)
        t_8 = t_7.size()
        t_9 = self.l_5(t_7)
        t_10 = self.l_6(t_7)
        t_7 = self.l_7(t_7)
        t_8 = t_8[0]
        t_9 = t_9.view(t_8, -1, 32, 128)
        t_9 = t_9.transpose(1, 2)
        t_10 = t_10.view(t_8, -1, 32, 128)
        t_10 = t_10.transpose(1, 2)
        t_7 = t_7.view(t_8, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_10 = t_10.transpose(3, 2)
        t_10 = torch.matmul(t_9, t_10)
        t_10 += x1
        t_9 = t_10.float()
        t_9 = torch.nn.functional.softmax(t_9, dim=-1, _stacklevel=3, dtype=None)
        t_10 = t_9.type_as(t_10)
        t_10 = self.l_8(t_10)
        t_7 = torch.matmul(t_10, t_7)
        t_7 = t_7.transpose(1, 2)
        t_7 = t_7.contiguous()
        t_8 = t_7.view(t_8, -1, 4096)
        t_8 = self.l_9(t_8)
        t_8 = self.l_10(t_8)
        t_8 = t_6 + t_8
        t_6 = self.l_11(t_8)
        t_7 = t_6.size()
        t_6 = self.l_12(t_6)
        t_7 = t_7[0]
        t_6 = t_6.view(t_7, -1, 32, 128)
        t_6 = t_6.transpose(1, 2)
        t_0 = t_0.view(t_7, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_1 = t_1.view(t_7, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_0 = t_0.transpose(3, 2)
        t_0 = torch.matmul(t_6, t_0)
        t_0 += x2
        t_6 = t_0.float()
        t_6 = torch.nn.functional.softmax(t_6, dim=-1, _stacklevel=3, dtype=None)
        t_0 = t_6.type_as(t_0)
        t_0 = self.l_15(t_0)
        t_1 = torch.matmul(t_0, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_7 = t_1.view(t_7, -1, 4096)
        t_7 = self.l_16(t_7)
        t_7 = self.l_17(t_7)
        t_7 = t_8 + t_7
        t_8 = self.l_18(t_7)
        t_8 = self.l_19(t_8)
        t_8 = torch.nn.functional.relu(t_8, inplace=False)
        t_8 = self.l_20(t_8)
        t_8 = self.l_21(t_8)
        t_8 = self.l_22(t_8)
        t_8 = t_7 + t_8
        t_7 = self.l_23(t_8)
        t_1 = t_7.size()
        t_0 = self.l_24(t_7)
        t_6 = self.l_25(t_7)
        t_7 = self.l_26(t_7)
        t_1 = t_1[0]
        t_0 = t_0.view(t_1, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_6 = t_6.view(t_1, -1, 32, 128)
        t_6 = t_6.transpose(1, 2)
        t_7 = t_7.view(t_1, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_6 = t_6.transpose(3, 2)
        t_6 = torch.matmul(t_0, t_6)
        t_6 += x1
        t_0 = t_6.float()
        t_0 = torch.nn.functional.softmax(t_0, dim=-1, _stacklevel=3, dtype=None)
        t_6 = t_0.type_as(t_6)
        t_6 = self.l_27(t_6)
        t_7 = torch.matmul(t_6, t_7)
        t_7 = t_7.transpose(1, 2)
        t_7 = t_7.contiguous()
        t_1 = t_7.view(t_1, -1, 4096)
        t_1 = self.l_28(t_1)
        t_1 = self.l_29(t_1)
        t_1 = t_8 + t_1
        t_8 = self.l_30(t_1)
        t_7 = t_8.size()
        t_8 = self.l_31(t_8)
        t_7 = t_7[0]
        t_8 = t_8.view(t_7, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_2 = t_2.view(t_7, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_3 = t_3.view(t_7, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_2 = t_2.transpose(3, 2)
        t_2 = torch.matmul(t_8, t_2)
        t_2 += x2
        t_8 = t_2.float()
        t_8 = torch.nn.functional.softmax(t_8, dim=-1, _stacklevel=3, dtype=None)
        t_2 = t_8.type_as(t_2)
        t_2 = self.l_34(t_2)
        t_3 = torch.matmul(t_2, t_3)
        t_3 = t_3.transpose(1, 2)
        t_3 = t_3.contiguous()
        t_7 = t_3.view(t_7, -1, 4096)
        t_7 = self.l_35(t_7)
        t_7 = self.l_36(t_7)
        t_7 = t_1 + t_7
        t_1 = self.l_37(t_7)
        t_1 = self.l_38(t_1)
        t_1 = torch.nn.functional.relu(t_1, inplace=False)
        t_1 = self.l_39(t_1)
        t_1 = self.l_40(t_1)
        t_1 = self.l_41(t_1)
        t_1 = t_7 + t_1
        t_7 = self.l_42(t_1)
        t_3 = t_7.size()
        t_2 = self.l_43(t_7)
        t_8 = self.l_44(t_7)
        t_7 = self.l_45(t_7)
        t_3 = t_3[0]
        t_2 = t_2.view(t_3, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_8 = t_8.view(t_3, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_7 = t_7.view(t_3, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_8 = t_8.transpose(3, 2)
        t_8 = torch.matmul(t_2, t_8)
        t_8 += x1
        t_2 = t_8.float()
        t_2 = torch.nn.functional.softmax(t_2, dim=-1, _stacklevel=3, dtype=None)
        t_8 = t_2.type_as(t_8)
        t_8 = self.l_46(t_8)
        t_7 = torch.matmul(t_8, t_7)
        t_7 = t_7.transpose(1, 2)
        t_7 = t_7.contiguous()
        t_3 = t_7.view(t_3, -1, 4096)
        t_3 = self.l_47(t_3)
        t_3 = self.l_48(t_3)
        t_3 = t_1 + t_3
        t_1 = self.l_49(t_3)
        t_7 = t_1.size()
        t_1 = self.l_50(t_1)
        t_7 = t_7[0]
        t_1 = t_1.view(t_7, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_4 = t_4.view(t_7, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_5 = t_5.view(t_7, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_1, t_4)
        t_4 += x2
        t_1 = t_4.float()
        t_1 = torch.nn.functional.softmax(t_1, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_1.type_as(t_4)
        t_4 = self.l_53(t_4)
        t_5 = torch.matmul(t_4, t_5)
        t_5 = t_5.transpose(1, 2)
        t_5 = t_5.contiguous()
        t_7 = t_5.view(t_7, -1, 4096)
        t_7 = self.l_54(t_7)
        t_7 = self.l_55(t_7)
        t_7 = t_3 + t_7
        t_3 = self.l_56(t_7)
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerCrossAttention[1]/Tensor::__add___2997
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerFF[2]/T5LayerNorm[layer_norm]
        return list(flatten((x0, x1, x2, t_7, t_3)))

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
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
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
        self.input_structure = [1, 1, 1, 1, 1]
        self.lookup = {'l_0': 'decoder.12.2.DenseReluDense.wi',
                        'l_1': 'decoder.12.2.DenseReluDense.dropout',
                        'l_2': 'decoder.12.2.DenseReluDense.wo',
                        'l_3': 'decoder.12.2.dropout',
                        'l_4': 'decoder.13.0.layer_norm',
                        'l_5': 'decoder.13.0.SelfAttention.q',
                        'l_6': 'decoder.13.0.SelfAttention.k',
                        'l_7': 'decoder.13.0.SelfAttention.v',
                        'l_8': 'decoder.13.0.SelfAttention.dropout',
                        'l_9': 'decoder.13.0.SelfAttention.o',
                        'l_10': 'decoder.13.0.dropout',
                        'l_11': 'decoder.13.1.layer_norm',
                        'l_12': 'decoder.13.1.EncDecAttention.q',
                        'l_13': 'decoder.13.1.EncDecAttention.k',
                        'l_14': 'decoder.13.1.EncDecAttention.v',
                        'l_15': 'decoder.13.1.EncDecAttention.dropout',
                        'l_16': 'decoder.13.1.EncDecAttention.o',
                        'l_17': 'decoder.13.1.dropout',
                        'l_18': 'decoder.13.2.layer_norm',
                        'l_19': 'decoder.13.2.DenseReluDense.wi',
                        'l_20': 'decoder.13.2.DenseReluDense.dropout',
                        'l_21': 'decoder.13.2.DenseReluDense.wo',
                        'l_22': 'decoder.13.2.dropout',
                        'l_23': 'decoder.14.0.layer_norm',
                        'l_24': 'decoder.14.0.SelfAttention.q',
                        'l_25': 'decoder.14.0.SelfAttention.k',
                        'l_26': 'decoder.14.0.SelfAttention.v',
                        'l_27': 'decoder.14.0.SelfAttention.dropout',
                        'l_28': 'decoder.14.0.SelfAttention.o',
                        'l_29': 'decoder.14.0.dropout',
                        'l_30': 'decoder.14.1.layer_norm',
                        'l_31': 'decoder.14.1.EncDecAttention.q',
                        'l_32': 'decoder.14.1.EncDecAttention.k',
                        'l_33': 'decoder.14.1.EncDecAttention.v',
                        'l_34': 'decoder.14.1.EncDecAttention.dropout',
                        'l_35': 'decoder.14.1.EncDecAttention.o',
                        'l_36': 'decoder.14.1.dropout',
                        'l_37': 'decoder.14.2.layer_norm',
                        'l_38': 'decoder.14.2.DenseReluDense.wi',
                        'l_39': 'decoder.14.2.DenseReluDense.dropout',
                        'l_40': 'decoder.14.2.DenseReluDense.wo',
                        'l_41': 'decoder.14.2.dropout',
                        'l_42': 'decoder.15.0.layer_norm',
                        'l_43': 'decoder.15.0.SelfAttention.q',
                        'l_44': 'decoder.15.0.SelfAttention.k',
                        'l_45': 'decoder.15.0.SelfAttention.v',
                        'l_46': 'decoder.15.0.SelfAttention.dropout',
                        'l_47': 'decoder.15.0.SelfAttention.o',
                        'l_48': 'decoder.15.0.dropout',
                        'l_49': 'decoder.15.1.layer_norm',
                        'l_50': 'decoder.15.1.EncDecAttention.q',
                        'l_51': 'decoder.15.1.EncDecAttention.k',
                        'l_52': 'decoder.15.1.EncDecAttention.v',
                        'l_53': 'decoder.15.1.EncDecAttention.dropout',
                        'l_54': 'decoder.15.1.EncDecAttention.o',
                        'l_55': 'decoder.15.1.dropout',
                        'l_56': 'decoder.15.2.layer_norm'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerFF[2]/Dropout[dropout] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[13]/T5LayerFF[2]/Dropout[dropout] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_36
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_37
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_38
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_39
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_40
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[14]/T5LayerFF[2]/Dropout[dropout] <=> self.l_41
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_42
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_43
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_44
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_45
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_46
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_47
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_48
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_49
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_50
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_51
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_52
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_53
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_54
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_55
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_56
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout] <=> x0
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562 <=> x1
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658 <=> x2
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerCrossAttention[1]/Tensor::__add___2997 <=> x3
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[12]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> x4
        x0, x1, x2, x3, x4 = unflatten(args, self.input_structure)
        t_0 = self.l_13(x0)
        t_1 = self.l_14(x0)
        t_2 = self.l_32(x0)
        t_3 = self.l_33(x0)
        t_4 = self.l_51(x0)
        t_5 = self.l_52(x0)
        t_6 = self.l_0(x4)
        t_6 = torch.nn.functional.relu(t_6, inplace=False)
        t_6 = self.l_1(t_6)
        t_6 = self.l_2(t_6)
        t_6 = self.l_3(t_6)
        t_6 = x3 + t_6
        t_7 = self.l_4(t_6)
        t_8 = t_7.size()
        t_9 = self.l_5(t_7)
        t_10 = self.l_6(t_7)
        t_7 = self.l_7(t_7)
        t_8 = t_8[0]
        t_9 = t_9.view(t_8, -1, 32, 128)
        t_9 = t_9.transpose(1, 2)
        t_10 = t_10.view(t_8, -1, 32, 128)
        t_10 = t_10.transpose(1, 2)
        t_7 = t_7.view(t_8, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_10 = t_10.transpose(3, 2)
        t_10 = torch.matmul(t_9, t_10)
        t_10 += x1
        t_9 = t_10.float()
        t_9 = torch.nn.functional.softmax(t_9, dim=-1, _stacklevel=3, dtype=None)
        t_10 = t_9.type_as(t_10)
        t_10 = self.l_8(t_10)
        t_7 = torch.matmul(t_10, t_7)
        t_7 = t_7.transpose(1, 2)
        t_7 = t_7.contiguous()
        t_8 = t_7.view(t_8, -1, 4096)
        t_8 = self.l_9(t_8)
        t_8 = self.l_10(t_8)
        t_8 = t_6 + t_8
        t_6 = self.l_11(t_8)
        t_7 = t_6.size()
        t_6 = self.l_12(t_6)
        t_7 = t_7[0]
        t_6 = t_6.view(t_7, -1, 32, 128)
        t_6 = t_6.transpose(1, 2)
        t_0 = t_0.view(t_7, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_1 = t_1.view(t_7, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_0 = t_0.transpose(3, 2)
        t_0 = torch.matmul(t_6, t_0)
        t_0 += x2
        t_6 = t_0.float()
        t_6 = torch.nn.functional.softmax(t_6, dim=-1, _stacklevel=3, dtype=None)
        t_0 = t_6.type_as(t_0)
        t_0 = self.l_15(t_0)
        t_1 = torch.matmul(t_0, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_7 = t_1.view(t_7, -1, 4096)
        t_7 = self.l_16(t_7)
        t_7 = self.l_17(t_7)
        t_7 = t_8 + t_7
        t_8 = self.l_18(t_7)
        t_8 = self.l_19(t_8)
        t_8 = torch.nn.functional.relu(t_8, inplace=False)
        t_8 = self.l_20(t_8)
        t_8 = self.l_21(t_8)
        t_8 = self.l_22(t_8)
        t_8 = t_7 + t_8
        t_7 = self.l_23(t_8)
        t_1 = t_7.size()
        t_0 = self.l_24(t_7)
        t_6 = self.l_25(t_7)
        t_7 = self.l_26(t_7)
        t_1 = t_1[0]
        t_0 = t_0.view(t_1, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_6 = t_6.view(t_1, -1, 32, 128)
        t_6 = t_6.transpose(1, 2)
        t_7 = t_7.view(t_1, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_6 = t_6.transpose(3, 2)
        t_6 = torch.matmul(t_0, t_6)
        t_6 += x1
        t_0 = t_6.float()
        t_0 = torch.nn.functional.softmax(t_0, dim=-1, _stacklevel=3, dtype=None)
        t_6 = t_0.type_as(t_6)
        t_6 = self.l_27(t_6)
        t_7 = torch.matmul(t_6, t_7)
        t_7 = t_7.transpose(1, 2)
        t_7 = t_7.contiguous()
        t_1 = t_7.view(t_1, -1, 4096)
        t_1 = self.l_28(t_1)
        t_1 = self.l_29(t_1)
        t_1 = t_8 + t_1
        t_8 = self.l_30(t_1)
        t_7 = t_8.size()
        t_8 = self.l_31(t_8)
        t_7 = t_7[0]
        t_8 = t_8.view(t_7, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_2 = t_2.view(t_7, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_3 = t_3.view(t_7, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_2 = t_2.transpose(3, 2)
        t_2 = torch.matmul(t_8, t_2)
        t_2 += x2
        t_8 = t_2.float()
        t_8 = torch.nn.functional.softmax(t_8, dim=-1, _stacklevel=3, dtype=None)
        t_2 = t_8.type_as(t_2)
        t_2 = self.l_34(t_2)
        t_3 = torch.matmul(t_2, t_3)
        t_3 = t_3.transpose(1, 2)
        t_3 = t_3.contiguous()
        t_7 = t_3.view(t_7, -1, 4096)
        t_7 = self.l_35(t_7)
        t_7 = self.l_36(t_7)
        t_7 = t_1 + t_7
        t_1 = self.l_37(t_7)
        t_1 = self.l_38(t_1)
        t_1 = torch.nn.functional.relu(t_1, inplace=False)
        t_1 = self.l_39(t_1)
        t_1 = self.l_40(t_1)
        t_1 = self.l_41(t_1)
        t_1 = t_7 + t_1
        t_7 = self.l_42(t_1)
        t_3 = t_7.size()
        t_2 = self.l_43(t_7)
        t_8 = self.l_44(t_7)
        t_7 = self.l_45(t_7)
        t_3 = t_3[0]
        t_2 = t_2.view(t_3, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_8 = t_8.view(t_3, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_7 = t_7.view(t_3, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_8 = t_8.transpose(3, 2)
        t_8 = torch.matmul(t_2, t_8)
        t_8 += x1
        t_2 = t_8.float()
        t_2 = torch.nn.functional.softmax(t_2, dim=-1, _stacklevel=3, dtype=None)
        t_8 = t_2.type_as(t_8)
        t_8 = self.l_46(t_8)
        t_7 = torch.matmul(t_8, t_7)
        t_7 = t_7.transpose(1, 2)
        t_7 = t_7.contiguous()
        t_3 = t_7.view(t_3, -1, 4096)
        t_3 = self.l_47(t_3)
        t_3 = self.l_48(t_3)
        t_3 = t_1 + t_3
        t_1 = self.l_49(t_3)
        t_7 = t_1.size()
        t_1 = self.l_50(t_1)
        t_7 = t_7[0]
        t_1 = t_1.view(t_7, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_4 = t_4.view(t_7, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_5 = t_5.view(t_7, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_1, t_4)
        t_4 += x2
        t_1 = t_4.float()
        t_1 = torch.nn.functional.softmax(t_1, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_1.type_as(t_4)
        t_4 = self.l_53(t_4)
        t_5 = torch.matmul(t_4, t_5)
        t_5 = t_5.transpose(1, 2)
        t_5 = t_5.contiguous()
        t_7 = t_5.view(t_7, -1, 4096)
        t_7 = self.l_54(t_7)
        t_7 = self.l_55(t_7)
        t_7 = t_3 + t_7
        t_3 = self.l_56(t_7)
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerCrossAttention[1]/Tensor::__add___3327
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerFF[2]/T5LayerNorm[layer_norm]
        return list(flatten((x0, x1, x2, t_7, t_3)))

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
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
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
        self.input_structure = [1, 1, 1, 1, 1]
        self.lookup = {'l_0': 'decoder.15.2.DenseReluDense.wi',
                        'l_1': 'decoder.15.2.DenseReluDense.dropout',
                        'l_2': 'decoder.15.2.DenseReluDense.wo',
                        'l_3': 'decoder.15.2.dropout',
                        'l_4': 'decoder.16.0.layer_norm',
                        'l_5': 'decoder.16.0.SelfAttention.q',
                        'l_6': 'decoder.16.0.SelfAttention.k',
                        'l_7': 'decoder.16.0.SelfAttention.v',
                        'l_8': 'decoder.16.0.SelfAttention.dropout',
                        'l_9': 'decoder.16.0.SelfAttention.o',
                        'l_10': 'decoder.16.0.dropout',
                        'l_11': 'decoder.16.1.layer_norm',
                        'l_12': 'decoder.16.1.EncDecAttention.q',
                        'l_13': 'decoder.16.1.EncDecAttention.k',
                        'l_14': 'decoder.16.1.EncDecAttention.v',
                        'l_15': 'decoder.16.1.EncDecAttention.dropout',
                        'l_16': 'decoder.16.1.EncDecAttention.o',
                        'l_17': 'decoder.16.1.dropout',
                        'l_18': 'decoder.16.2.layer_norm',
                        'l_19': 'decoder.16.2.DenseReluDense.wi',
                        'l_20': 'decoder.16.2.DenseReluDense.dropout',
                        'l_21': 'decoder.16.2.DenseReluDense.wo',
                        'l_22': 'decoder.16.2.dropout',
                        'l_23': 'decoder.17.0.layer_norm',
                        'l_24': 'decoder.17.0.SelfAttention.q',
                        'l_25': 'decoder.17.0.SelfAttention.k',
                        'l_26': 'decoder.17.0.SelfAttention.v',
                        'l_27': 'decoder.17.0.SelfAttention.dropout',
                        'l_28': 'decoder.17.0.SelfAttention.o',
                        'l_29': 'decoder.17.0.dropout',
                        'l_30': 'decoder.17.1.layer_norm',
                        'l_31': 'decoder.17.1.EncDecAttention.q',
                        'l_32': 'decoder.17.1.EncDecAttention.k',
                        'l_33': 'decoder.17.1.EncDecAttention.v',
                        'l_34': 'decoder.17.1.EncDecAttention.dropout',
                        'l_35': 'decoder.17.1.EncDecAttention.o',
                        'l_36': 'decoder.17.1.dropout',
                        'l_37': 'decoder.17.2.layer_norm',
                        'l_38': 'decoder.17.2.DenseReluDense.wi',
                        'l_39': 'decoder.17.2.DenseReluDense.dropout',
                        'l_40': 'decoder.17.2.DenseReluDense.wo',
                        'l_41': 'decoder.17.2.dropout',
                        'l_42': 'decoder.18.0.layer_norm',
                        'l_43': 'decoder.18.0.SelfAttention.q',
                        'l_44': 'decoder.18.0.SelfAttention.k',
                        'l_45': 'decoder.18.0.SelfAttention.v',
                        'l_46': 'decoder.18.0.SelfAttention.dropout',
                        'l_47': 'decoder.18.0.SelfAttention.o',
                        'l_48': 'decoder.18.0.dropout',
                        'l_49': 'decoder.18.1.layer_norm',
                        'l_50': 'decoder.18.1.EncDecAttention.q',
                        'l_51': 'decoder.18.1.EncDecAttention.k',
                        'l_52': 'decoder.18.1.EncDecAttention.v',
                        'l_53': 'decoder.18.1.EncDecAttention.dropout',
                        'l_54': 'decoder.18.1.EncDecAttention.o',
                        'l_55': 'decoder.18.1.dropout',
                        'l_56': 'decoder.18.2.layer_norm'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerFF[2]/Dropout[dropout] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[16]/T5LayerFF[2]/Dropout[dropout] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_36
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_37
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_38
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_39
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_40
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[17]/T5LayerFF[2]/Dropout[dropout] <=> self.l_41
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_42
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_43
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_44
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_45
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_46
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_47
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_48
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_49
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_50
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_51
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_52
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_53
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_54
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_55
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_56
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout] <=> x0
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562 <=> x1
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658 <=> x2
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerCrossAttention[1]/Tensor::__add___3327 <=> x3
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[15]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> x4
        x0, x1, x2, x3, x4 = unflatten(args, self.input_structure)
        t_0 = self.l_13(x0)
        t_1 = self.l_14(x0)
        t_2 = self.l_32(x0)
        t_3 = self.l_33(x0)
        t_4 = self.l_51(x0)
        t_5 = self.l_52(x0)
        t_6 = self.l_0(x4)
        t_6 = torch.nn.functional.relu(t_6, inplace=False)
        t_6 = self.l_1(t_6)
        t_6 = self.l_2(t_6)
        t_6 = self.l_3(t_6)
        t_6 = x3 + t_6
        t_7 = self.l_4(t_6)
        t_8 = t_7.size()
        t_9 = self.l_5(t_7)
        t_10 = self.l_6(t_7)
        t_7 = self.l_7(t_7)
        t_8 = t_8[0]
        t_9 = t_9.view(t_8, -1, 32, 128)
        t_9 = t_9.transpose(1, 2)
        t_10 = t_10.view(t_8, -1, 32, 128)
        t_10 = t_10.transpose(1, 2)
        t_7 = t_7.view(t_8, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_10 = t_10.transpose(3, 2)
        t_10 = torch.matmul(t_9, t_10)
        t_10 += x1
        t_9 = t_10.float()
        t_9 = torch.nn.functional.softmax(t_9, dim=-1, _stacklevel=3, dtype=None)
        t_10 = t_9.type_as(t_10)
        t_10 = self.l_8(t_10)
        t_7 = torch.matmul(t_10, t_7)
        t_7 = t_7.transpose(1, 2)
        t_7 = t_7.contiguous()
        t_8 = t_7.view(t_8, -1, 4096)
        t_8 = self.l_9(t_8)
        t_8 = self.l_10(t_8)
        t_8 = t_6 + t_8
        t_6 = self.l_11(t_8)
        t_7 = t_6.size()
        t_6 = self.l_12(t_6)
        t_7 = t_7[0]
        t_6 = t_6.view(t_7, -1, 32, 128)
        t_6 = t_6.transpose(1, 2)
        t_0 = t_0.view(t_7, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_1 = t_1.view(t_7, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_0 = t_0.transpose(3, 2)
        t_0 = torch.matmul(t_6, t_0)
        t_0 += x2
        t_6 = t_0.float()
        t_6 = torch.nn.functional.softmax(t_6, dim=-1, _stacklevel=3, dtype=None)
        t_0 = t_6.type_as(t_0)
        t_0 = self.l_15(t_0)
        t_1 = torch.matmul(t_0, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_7 = t_1.view(t_7, -1, 4096)
        t_7 = self.l_16(t_7)
        t_7 = self.l_17(t_7)
        t_7 = t_8 + t_7
        t_8 = self.l_18(t_7)
        t_8 = self.l_19(t_8)
        t_8 = torch.nn.functional.relu(t_8, inplace=False)
        t_8 = self.l_20(t_8)
        t_8 = self.l_21(t_8)
        t_8 = self.l_22(t_8)
        t_8 = t_7 + t_8
        t_7 = self.l_23(t_8)
        t_1 = t_7.size()
        t_0 = self.l_24(t_7)
        t_6 = self.l_25(t_7)
        t_7 = self.l_26(t_7)
        t_1 = t_1[0]
        t_0 = t_0.view(t_1, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_6 = t_6.view(t_1, -1, 32, 128)
        t_6 = t_6.transpose(1, 2)
        t_7 = t_7.view(t_1, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_6 = t_6.transpose(3, 2)
        t_6 = torch.matmul(t_0, t_6)
        t_6 += x1
        t_0 = t_6.float()
        t_0 = torch.nn.functional.softmax(t_0, dim=-1, _stacklevel=3, dtype=None)
        t_6 = t_0.type_as(t_6)
        t_6 = self.l_27(t_6)
        t_7 = torch.matmul(t_6, t_7)
        t_7 = t_7.transpose(1, 2)
        t_7 = t_7.contiguous()
        t_1 = t_7.view(t_1, -1, 4096)
        t_1 = self.l_28(t_1)
        t_1 = self.l_29(t_1)
        t_1 = t_8 + t_1
        t_8 = self.l_30(t_1)
        t_7 = t_8.size()
        t_8 = self.l_31(t_8)
        t_7 = t_7[0]
        t_8 = t_8.view(t_7, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_2 = t_2.view(t_7, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_3 = t_3.view(t_7, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_2 = t_2.transpose(3, 2)
        t_2 = torch.matmul(t_8, t_2)
        t_2 += x2
        t_8 = t_2.float()
        t_8 = torch.nn.functional.softmax(t_8, dim=-1, _stacklevel=3, dtype=None)
        t_2 = t_8.type_as(t_2)
        t_2 = self.l_34(t_2)
        t_3 = torch.matmul(t_2, t_3)
        t_3 = t_3.transpose(1, 2)
        t_3 = t_3.contiguous()
        t_7 = t_3.view(t_7, -1, 4096)
        t_7 = self.l_35(t_7)
        t_7 = self.l_36(t_7)
        t_7 = t_1 + t_7
        t_1 = self.l_37(t_7)
        t_1 = self.l_38(t_1)
        t_1 = torch.nn.functional.relu(t_1, inplace=False)
        t_1 = self.l_39(t_1)
        t_1 = self.l_40(t_1)
        t_1 = self.l_41(t_1)
        t_1 = t_7 + t_1
        t_7 = self.l_42(t_1)
        t_3 = t_7.size()
        t_2 = self.l_43(t_7)
        t_8 = self.l_44(t_7)
        t_7 = self.l_45(t_7)
        t_3 = t_3[0]
        t_2 = t_2.view(t_3, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_8 = t_8.view(t_3, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_7 = t_7.view(t_3, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_8 = t_8.transpose(3, 2)
        t_8 = torch.matmul(t_2, t_8)
        t_8 += x1
        t_2 = t_8.float()
        t_2 = torch.nn.functional.softmax(t_2, dim=-1, _stacklevel=3, dtype=None)
        t_8 = t_2.type_as(t_8)
        t_8 = self.l_46(t_8)
        t_7 = torch.matmul(t_8, t_7)
        t_7 = t_7.transpose(1, 2)
        t_7 = t_7.contiguous()
        t_3 = t_7.view(t_3, -1, 4096)
        t_3 = self.l_47(t_3)
        t_3 = self.l_48(t_3)
        t_3 = t_1 + t_3
        t_1 = self.l_49(t_3)
        t_7 = t_1.size()
        t_1 = self.l_50(t_1)
        t_7 = t_7[0]
        t_1 = t_1.view(t_7, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_4 = t_4.view(t_7, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_5 = t_5.view(t_7, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_1, t_4)
        t_4 += x2
        t_1 = t_4.float()
        t_1 = torch.nn.functional.softmax(t_1, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_1.type_as(t_4)
        t_4 = self.l_53(t_4)
        t_5 = torch.matmul(t_4, t_5)
        t_5 = t_5.transpose(1, 2)
        t_5 = t_5.contiguous()
        t_7 = t_5.view(t_7, -1, 4096)
        t_7 = self.l_54(t_7)
        t_7 = self.l_55(t_7)
        t_7 = t_3 + t_7
        t_3 = self.l_56(t_7)
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerCrossAttention[1]/Tensor::__add___3657
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerFF[2]/T5LayerNorm[layer_norm]
        return list(flatten((x0, x1, x2, t_7, t_3)))

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
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
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
        self.input_structure = [1, 1, 1, 1, 1]
        self.lookup = {'l_0': 'decoder.18.2.DenseReluDense.wi',
                        'l_1': 'decoder.18.2.DenseReluDense.dropout',
                        'l_2': 'decoder.18.2.DenseReluDense.wo',
                        'l_3': 'decoder.18.2.dropout',
                        'l_4': 'decoder.19.0.layer_norm',
                        'l_5': 'decoder.19.0.SelfAttention.q',
                        'l_6': 'decoder.19.0.SelfAttention.k',
                        'l_7': 'decoder.19.0.SelfAttention.v',
                        'l_8': 'decoder.19.0.SelfAttention.dropout',
                        'l_9': 'decoder.19.0.SelfAttention.o',
                        'l_10': 'decoder.19.0.dropout',
                        'l_11': 'decoder.19.1.layer_norm',
                        'l_12': 'decoder.19.1.EncDecAttention.q',
                        'l_13': 'decoder.19.1.EncDecAttention.k',
                        'l_14': 'decoder.19.1.EncDecAttention.v',
                        'l_15': 'decoder.19.1.EncDecAttention.dropout',
                        'l_16': 'decoder.19.1.EncDecAttention.o',
                        'l_17': 'decoder.19.1.dropout',
                        'l_18': 'decoder.19.2.layer_norm',
                        'l_19': 'decoder.19.2.DenseReluDense.wi',
                        'l_20': 'decoder.19.2.DenseReluDense.dropout',
                        'l_21': 'decoder.19.2.DenseReluDense.wo',
                        'l_22': 'decoder.19.2.dropout',
                        'l_23': 'decoder.20.0.layer_norm',
                        'l_24': 'decoder.20.0.SelfAttention.q',
                        'l_25': 'decoder.20.0.SelfAttention.k',
                        'l_26': 'decoder.20.0.SelfAttention.v',
                        'l_27': 'decoder.20.0.SelfAttention.dropout',
                        'l_28': 'decoder.20.0.SelfAttention.o',
                        'l_29': 'decoder.20.0.dropout',
                        'l_30': 'decoder.20.1.layer_norm',
                        'l_31': 'decoder.20.1.EncDecAttention.q',
                        'l_32': 'decoder.20.1.EncDecAttention.k',
                        'l_33': 'decoder.20.1.EncDecAttention.v',
                        'l_34': 'decoder.20.1.EncDecAttention.dropout',
                        'l_35': 'decoder.20.1.EncDecAttention.o',
                        'l_36': 'decoder.20.1.dropout',
                        'l_37': 'decoder.20.2.layer_norm',
                        'l_38': 'decoder.20.2.DenseReluDense.wi',
                        'l_39': 'decoder.20.2.DenseReluDense.dropout',
                        'l_40': 'decoder.20.2.DenseReluDense.wo',
                        'l_41': 'decoder.20.2.dropout',
                        'l_42': 'decoder.21.0.layer_norm',
                        'l_43': 'decoder.21.0.SelfAttention.q',
                        'l_44': 'decoder.21.0.SelfAttention.k',
                        'l_45': 'decoder.21.0.SelfAttention.v',
                        'l_46': 'decoder.21.0.SelfAttention.dropout',
                        'l_47': 'decoder.21.0.SelfAttention.o',
                        'l_48': 'decoder.21.0.dropout',
                        'l_49': 'decoder.21.1.layer_norm',
                        'l_50': 'decoder.21.1.EncDecAttention.q',
                        'l_51': 'decoder.21.1.EncDecAttention.k',
                        'l_52': 'decoder.21.1.EncDecAttention.v',
                        'l_53': 'decoder.21.1.EncDecAttention.dropout',
                        'l_54': 'decoder.21.1.EncDecAttention.o',
                        'l_55': 'decoder.21.1.dropout',
                        'l_56': 'decoder.21.2.layer_norm'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerFF[2]/Dropout[dropout] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[19]/T5LayerFF[2]/Dropout[dropout] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_36
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_37
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_38
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_39
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_40
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[20]/T5LayerFF[2]/Dropout[dropout] <=> self.l_41
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_42
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_43
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_44
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_45
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_46
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_47
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_48
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_49
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_50
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_51
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_52
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_53
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_54
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_55
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_56
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout] <=> x0
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562 <=> x1
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658 <=> x2
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerCrossAttention[1]/Tensor::__add___3657 <=> x3
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[18]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> x4
        x0, x1, x2, x3, x4 = unflatten(args, self.input_structure)
        t_0 = self.l_13(x0)
        t_1 = self.l_14(x0)
        t_2 = self.l_32(x0)
        t_3 = self.l_33(x0)
        t_4 = self.l_51(x0)
        t_5 = self.l_52(x0)
        t_6 = self.l_0(x4)
        t_6 = torch.nn.functional.relu(t_6, inplace=False)
        t_6 = self.l_1(t_6)
        t_6 = self.l_2(t_6)
        t_6 = self.l_3(t_6)
        t_6 = x3 + t_6
        t_7 = self.l_4(t_6)
        t_8 = t_7.size()
        t_9 = self.l_5(t_7)
        t_10 = self.l_6(t_7)
        t_7 = self.l_7(t_7)
        t_8 = t_8[0]
        t_9 = t_9.view(t_8, -1, 32, 128)
        t_9 = t_9.transpose(1, 2)
        t_10 = t_10.view(t_8, -1, 32, 128)
        t_10 = t_10.transpose(1, 2)
        t_7 = t_7.view(t_8, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_10 = t_10.transpose(3, 2)
        t_10 = torch.matmul(t_9, t_10)
        t_10 += x1
        t_9 = t_10.float()
        t_9 = torch.nn.functional.softmax(t_9, dim=-1, _stacklevel=3, dtype=None)
        t_10 = t_9.type_as(t_10)
        t_10 = self.l_8(t_10)
        t_7 = torch.matmul(t_10, t_7)
        t_7 = t_7.transpose(1, 2)
        t_7 = t_7.contiguous()
        t_8 = t_7.view(t_8, -1, 4096)
        t_8 = self.l_9(t_8)
        t_8 = self.l_10(t_8)
        t_8 = t_6 + t_8
        t_6 = self.l_11(t_8)
        t_7 = t_6.size()
        t_6 = self.l_12(t_6)
        t_7 = t_7[0]
        t_6 = t_6.view(t_7, -1, 32, 128)
        t_6 = t_6.transpose(1, 2)
        t_0 = t_0.view(t_7, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_1 = t_1.view(t_7, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_0 = t_0.transpose(3, 2)
        t_0 = torch.matmul(t_6, t_0)
        t_0 += x2
        t_6 = t_0.float()
        t_6 = torch.nn.functional.softmax(t_6, dim=-1, _stacklevel=3, dtype=None)
        t_0 = t_6.type_as(t_0)
        t_0 = self.l_15(t_0)
        t_1 = torch.matmul(t_0, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_7 = t_1.view(t_7, -1, 4096)
        t_7 = self.l_16(t_7)
        t_7 = self.l_17(t_7)
        t_7 = t_8 + t_7
        t_8 = self.l_18(t_7)
        t_8 = self.l_19(t_8)
        t_8 = torch.nn.functional.relu(t_8, inplace=False)
        t_8 = self.l_20(t_8)
        t_8 = self.l_21(t_8)
        t_8 = self.l_22(t_8)
        t_8 = t_7 + t_8
        t_7 = self.l_23(t_8)
        t_1 = t_7.size()
        t_0 = self.l_24(t_7)
        t_6 = self.l_25(t_7)
        t_7 = self.l_26(t_7)
        t_1 = t_1[0]
        t_0 = t_0.view(t_1, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_6 = t_6.view(t_1, -1, 32, 128)
        t_6 = t_6.transpose(1, 2)
        t_7 = t_7.view(t_1, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_6 = t_6.transpose(3, 2)
        t_6 = torch.matmul(t_0, t_6)
        t_6 += x1
        t_0 = t_6.float()
        t_0 = torch.nn.functional.softmax(t_0, dim=-1, _stacklevel=3, dtype=None)
        t_6 = t_0.type_as(t_6)
        t_6 = self.l_27(t_6)
        t_7 = torch.matmul(t_6, t_7)
        t_7 = t_7.transpose(1, 2)
        t_7 = t_7.contiguous()
        t_1 = t_7.view(t_1, -1, 4096)
        t_1 = self.l_28(t_1)
        t_1 = self.l_29(t_1)
        t_1 = t_8 + t_1
        t_8 = self.l_30(t_1)
        t_7 = t_8.size()
        t_8 = self.l_31(t_8)
        t_7 = t_7[0]
        t_8 = t_8.view(t_7, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_2 = t_2.view(t_7, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_3 = t_3.view(t_7, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_2 = t_2.transpose(3, 2)
        t_2 = torch.matmul(t_8, t_2)
        t_2 += x2
        t_8 = t_2.float()
        t_8 = torch.nn.functional.softmax(t_8, dim=-1, _stacklevel=3, dtype=None)
        t_2 = t_8.type_as(t_2)
        t_2 = self.l_34(t_2)
        t_3 = torch.matmul(t_2, t_3)
        t_3 = t_3.transpose(1, 2)
        t_3 = t_3.contiguous()
        t_7 = t_3.view(t_7, -1, 4096)
        t_7 = self.l_35(t_7)
        t_7 = self.l_36(t_7)
        t_7 = t_1 + t_7
        t_1 = self.l_37(t_7)
        t_1 = self.l_38(t_1)
        t_1 = torch.nn.functional.relu(t_1, inplace=False)
        t_1 = self.l_39(t_1)
        t_1 = self.l_40(t_1)
        t_1 = self.l_41(t_1)
        t_1 = t_7 + t_1
        t_7 = self.l_42(t_1)
        t_3 = t_7.size()
        t_2 = self.l_43(t_7)
        t_8 = self.l_44(t_7)
        t_7 = self.l_45(t_7)
        t_3 = t_3[0]
        t_2 = t_2.view(t_3, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_8 = t_8.view(t_3, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_7 = t_7.view(t_3, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_8 = t_8.transpose(3, 2)
        t_8 = torch.matmul(t_2, t_8)
        t_8 += x1
        t_2 = t_8.float()
        t_2 = torch.nn.functional.softmax(t_2, dim=-1, _stacklevel=3, dtype=None)
        t_8 = t_2.type_as(t_8)
        t_8 = self.l_46(t_8)
        t_7 = torch.matmul(t_8, t_7)
        t_7 = t_7.transpose(1, 2)
        t_7 = t_7.contiguous()
        t_3 = t_7.view(t_3, -1, 4096)
        t_3 = self.l_47(t_3)
        t_3 = self.l_48(t_3)
        t_3 = t_1 + t_3
        t_1 = self.l_49(t_3)
        t_7 = t_1.size()
        t_1 = self.l_50(t_1)
        t_7 = t_7[0]
        t_1 = t_1.view(t_7, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_4 = t_4.view(t_7, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_5 = t_5.view(t_7, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_1, t_4)
        t_4 += x2
        t_1 = t_4.float()
        t_1 = torch.nn.functional.softmax(t_1, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_1.type_as(t_4)
        t_4 = self.l_53(t_4)
        t_5 = torch.matmul(t_4, t_5)
        t_5 = t_5.transpose(1, 2)
        t_5 = t_5.contiguous()
        t_7 = t_5.view(t_7, -1, 4096)
        t_7 = self.l_54(t_7)
        t_7 = self.l_55(t_7)
        t_7 = t_3 + t_7
        t_3 = self.l_56(t_7)
        # Returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerCrossAttention[1]/Tensor::__add___3987
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerFF[2]/T5LayerNorm[layer_norm]
        return list(flatten((x0, x1, x2, t_7, t_3)))

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
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5LayerNorm[final_layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/Dropout[dropout]',
            'T5ForConditionalGeneration/Linear[lm_head]',
            'T5ForConditionalGeneration/CrossEntropyLoss[lm_loss]',
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
        self.input_structure = [1, 1, 1, 1, 1, 1]
        self.lookup = {'l_0': 'decoder.21.2.DenseReluDense.wi',
                        'l_1': 'decoder.21.2.DenseReluDense.dropout',
                        'l_2': 'decoder.21.2.DenseReluDense.wo',
                        'l_3': 'decoder.21.2.dropout',
                        'l_4': 'decoder.22.0.layer_norm',
                        'l_5': 'decoder.22.0.SelfAttention.q',
                        'l_6': 'decoder.22.0.SelfAttention.k',
                        'l_7': 'decoder.22.0.SelfAttention.v',
                        'l_8': 'decoder.22.0.SelfAttention.dropout',
                        'l_9': 'decoder.22.0.SelfAttention.o',
                        'l_10': 'decoder.22.0.dropout',
                        'l_11': 'decoder.22.1.layer_norm',
                        'l_12': 'decoder.22.1.EncDecAttention.q',
                        'l_13': 'decoder.22.1.EncDecAttention.k',
                        'l_14': 'decoder.22.1.EncDecAttention.v',
                        'l_15': 'decoder.22.1.EncDecAttention.dropout',
                        'l_16': 'decoder.22.1.EncDecAttention.o',
                        'l_17': 'decoder.22.1.dropout',
                        'l_18': 'decoder.22.2.layer_norm',
                        'l_19': 'decoder.22.2.DenseReluDense.wi',
                        'l_20': 'decoder.22.2.DenseReluDense.dropout',
                        'l_21': 'decoder.22.2.DenseReluDense.wo',
                        'l_22': 'decoder.22.2.dropout',
                        'l_23': 'decoder.23.0.layer_norm',
                        'l_24': 'decoder.23.0.SelfAttention.q',
                        'l_25': 'decoder.23.0.SelfAttention.k',
                        'l_26': 'decoder.23.0.SelfAttention.v',
                        'l_27': 'decoder.23.0.SelfAttention.dropout',
                        'l_28': 'decoder.23.0.SelfAttention.o',
                        'l_29': 'decoder.23.0.dropout',
                        'l_30': 'decoder.23.1.layer_norm',
                        'l_31': 'decoder.23.1.EncDecAttention.q',
                        'l_32': 'decoder.23.1.EncDecAttention.k',
                        'l_33': 'decoder.23.1.EncDecAttention.v',
                        'l_34': 'decoder.23.1.EncDecAttention.dropout',
                        'l_35': 'decoder.23.1.EncDecAttention.o',
                        'l_36': 'decoder.23.1.dropout',
                        'l_37': 'decoder.23.2.layer_norm',
                        'l_38': 'decoder.23.2.DenseReluDense.wi',
                        'l_39': 'decoder.23.2.DenseReluDense.dropout',
                        'l_40': 'decoder.23.2.DenseReluDense.wo',
                        'l_41': 'decoder.23.2.dropout',
                        'l_42': 'decoder.final_layer_norm',
                        'l_43': 'decoder.dropout',
                        'l_44': 'lm_head',
                        'l_45': 'lm_loss'}
        self.to(self.device)

    def forward(self, *args):
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerFF[2]/Dropout[dropout] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[22]/T5LayerFF[2]/Dropout[dropout] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[q] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[k] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[v] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Dropout[dropout] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Linear[o] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[q] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[k] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[v] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Dropout[dropout] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Linear[o] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_36
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_37
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_38
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_39
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_40
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[23]/T5LayerFF[2]/Dropout[dropout] <=> self.l_41
        # T5ForConditionalGeneration/T5Stack[decoder]/T5LayerNorm[final_layer_norm] <=> self.l_42
        # T5ForConditionalGeneration/T5Stack[decoder]/Dropout[dropout] <=> self.l_43
        # T5ForConditionalGeneration/Linear[lm_head] <=> self.l_44
        # T5ForConditionalGeneration/CrossEntropyLoss[lm_loss] <=> self.l_45
        # input5 <=> lm_labels
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout] <=> x0
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]/Tensor::__add___1562 <=> x1
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]/Tensor::__add___1658 <=> x2
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerCrossAttention[1]/Tensor::__add___3987 <=> x3
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[21]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> x4
        lm_labels, x0, x1, x2, x3, x4 = unflatten(args, self.input_structure)
        t_0 = self.l_13(x0)
        t_1 = self.l_14(x0)
        t_2 = self.l_32(x0)
        t_3 = self.l_33(x0)
        t_4 = self.l_0(x4)
        t_4 = torch.nn.functional.relu(t_4, inplace=False)
        t_4 = self.l_1(t_4)
        t_4 = self.l_2(t_4)
        t_4 = self.l_3(t_4)
        t_4 = x3 + t_4
        t_5 = self.l_4(t_4)
        t_6 = t_5.size()
        t_7 = self.l_5(t_5)
        t_8 = self.l_6(t_5)
        t_5 = self.l_7(t_5)
        t_6 = t_6[0]
        t_7 = t_7.view(t_6, -1, 32, 128)
        t_7 = t_7.transpose(1, 2)
        t_8 = t_8.view(t_6, -1, 32, 128)
        t_8 = t_8.transpose(1, 2)
        t_5 = t_5.view(t_6, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_8 = t_8.transpose(3, 2)
        t_8 = torch.matmul(t_7, t_8)
        t_8 += x1
        t_7 = t_8.float()
        t_7 = torch.nn.functional.softmax(t_7, dim=-1, _stacklevel=3, dtype=None)
        t_8 = t_7.type_as(t_8)
        t_8 = self.l_8(t_8)
        t_5 = torch.matmul(t_8, t_5)
        t_5 = t_5.transpose(1, 2)
        t_5 = t_5.contiguous()
        t_6 = t_5.view(t_6, -1, 4096)
        t_6 = self.l_9(t_6)
        t_6 = self.l_10(t_6)
        t_6 = t_4 + t_6
        t_4 = self.l_11(t_6)
        t_5 = t_4.size()
        t_4 = self.l_12(t_4)
        t_5 = t_5[0]
        t_4 = t_4.view(t_5, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_0 = t_0.view(t_5, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_1 = t_1.view(t_5, -1, 32, 128)
        t_1 = t_1.transpose(1, 2)
        t_0 = t_0.transpose(3, 2)
        t_0 = torch.matmul(t_4, t_0)
        t_0 += x2
        t_4 = t_0.float()
        t_4 = torch.nn.functional.softmax(t_4, dim=-1, _stacklevel=3, dtype=None)
        t_0 = t_4.type_as(t_0)
        t_0 = self.l_15(t_0)
        t_1 = torch.matmul(t_0, t_1)
        t_1 = t_1.transpose(1, 2)
        t_1 = t_1.contiguous()
        t_5 = t_1.view(t_5, -1, 4096)
        t_5 = self.l_16(t_5)
        t_5 = self.l_17(t_5)
        t_5 = t_6 + t_5
        t_6 = self.l_18(t_5)
        t_6 = self.l_19(t_6)
        t_6 = torch.nn.functional.relu(t_6, inplace=False)
        t_6 = self.l_20(t_6)
        t_6 = self.l_21(t_6)
        t_6 = self.l_22(t_6)
        t_6 = t_5 + t_6
        t_5 = self.l_23(t_6)
        t_1 = t_5.size()
        t_0 = self.l_24(t_5)
        t_4 = self.l_25(t_5)
        t_5 = self.l_26(t_5)
        t_1 = t_1[0]
        t_0 = t_0.view(t_1, -1, 32, 128)
        t_0 = t_0.transpose(1, 2)
        t_4 = t_4.view(t_1, -1, 32, 128)
        t_4 = t_4.transpose(1, 2)
        t_5 = t_5.view(t_1, -1, 32, 128)
        t_5 = t_5.transpose(1, 2)
        t_4 = t_4.transpose(3, 2)
        t_4 = torch.matmul(t_0, t_4)
        t_4 += x1
        t_0 = t_4.float()
        t_0 = torch.nn.functional.softmax(t_0, dim=-1, _stacklevel=3, dtype=None)
        t_4 = t_0.type_as(t_4)
        t_4 = self.l_27(t_4)
        t_5 = torch.matmul(t_4, t_5)
        t_5 = t_5.transpose(1, 2)
        t_5 = t_5.contiguous()
        t_1 = t_5.view(t_1, -1, 4096)
        t_1 = self.l_28(t_1)
        t_1 = self.l_29(t_1)
        t_1 = t_6 + t_1
        t_6 = self.l_30(t_1)
        t_5 = t_6.size()
        t_6 = self.l_31(t_6)
        t_5 = t_5[0]
        t_6 = t_6.view(t_5, -1, 32, 128)
        t_6 = t_6.transpose(1, 2)
        t_2 = t_2.view(t_5, -1, 32, 128)
        t_2 = t_2.transpose(1, 2)
        t_3 = t_3.view(t_5, -1, 32, 128)
        t_3 = t_3.transpose(1, 2)
        t_2 = t_2.transpose(3, 2)
        t_2 = torch.matmul(t_6, t_2)
        t_2 += x2
        t_6 = t_2.float()
        t_6 = torch.nn.functional.softmax(t_6, dim=-1, _stacklevel=3, dtype=None)
        t_2 = t_6.type_as(t_2)
        t_2 = self.l_34(t_2)
        t_3 = torch.matmul(t_2, t_3)
        t_3 = t_3.transpose(1, 2)
        t_3 = t_3.contiguous()
        t_5 = t_3.view(t_5, -1, 4096)
        t_5 = self.l_35(t_5)
        t_5 = self.l_36(t_5)
        t_5 = t_1 + t_5
        t_1 = self.l_37(t_5)
        t_1 = self.l_38(t_1)
        t_1 = torch.nn.functional.relu(t_1, inplace=False)
        t_1 = self.l_39(t_1)
        t_1 = self.l_40(t_1)
        t_1 = self.l_41(t_1)
        t_1 = t_5 + t_1
        t_1 = self.l_42(t_1)
        t_1 = self.l_43(t_1)
        t_1 = t_1 * 0.03125
        t_1 = self.l_44(t_1)
        t_5 = t_1.size(-1)
        t_5 = t_1.view(-1, t_5)
        t_1 = lm_labels.view(-1)
        t_1 = self.l_45(t_5, t_1)
        # Returning:
        # T5ForConditionalGeneration/CrossEntropyLoss[lm_loss]
        return (t_1,)

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

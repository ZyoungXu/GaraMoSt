from functools import partial
import torch.nn as nn

from model import feature_extractor
from model import flow_estimation


def init_model_config(F=32, lambda_range='local', depth=[2, 2, 2, 2, 4, 4], lambda_r=[7, 7]):
    return {
        'embed_dims':[F, 2*F, 4*F, 8*F, 8*F, 16*F],
        'motion_dims':[0, 0, 0, 0, 8*F//depth[-2], 16*F//depth[-1]],
        'num_heads':[4, 8],
        'mlp_ratios':[4, 4],
        'lambda_global_or_local': lambda_range,
        'lambda_dim_k':[16, 16],
        'lambda_dim_u':[1, 1],
        'lambda_n':[32, 32],
        'lambda_r':lambda_r,
        'norm_layer':partial(nn.LayerNorm, eps=1e-6),
        'depths':depth,
    }, {
        'embed_dims':[F, 2*F, 4*F, 8*F, 8*F, 16*F],
        'motion_dims':[0, 0, 0, 0, 8*F//depth[-2], 16*F//depth[-1]],
        'depths':depth,
        'scales':[4, 8, 16],
        'hidden_dims':[4*F, 4*F],
        'c':F
    }

MODEL_CONFIG = {
                'LOGNAME': 'GaraMoSt',
                'MODEL_TYPE': (feature_extractor, flow_estimation),
                'MODEL_ARCH': init_model_config(
                    F = 32,
                    lambda_range='local',
                    depth = [2, 2, 2, 2, 4, 4],
                    lambda_r=[7, 7] # single-frame interpolation
                )
            }

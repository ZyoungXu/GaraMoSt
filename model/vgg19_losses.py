from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_net(layer_type: str, input_tensor: torch.Tensor,
               weight_bias: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Callable[[Any], Any]:
    if layer_type == 'conv':
        return F.relu(F.conv2d(input = input_tensor, weight = weight_bias[0], bias = weight_bias[1], stride=1, padding=0))
    elif layer_type == 'pool':
        return F.avg_pool2d(input = input_tensor, kernel_size=2, stride=2)
    else:
        raise ValueError('不支持的层类型：%s' % layer_type)


def _get_weight_and_bias(vgg_layers: np.ndarray,
                         index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    weights = vgg_layers[index][0][0][2][0][0]
    weights = torch.tensor(weights).permute(3, 2, 0, 1).to(device) # (filter_height, filter_width, in_channels, out_channels)转到(out_channels, in_channels/groups(groups默认1), kH, kW)
    bias = vgg_layers[index][0][0][2][0][1]
    bias = torch.tensor(np.reshape(bias, bias.size)).to(device)

    return weights, bias


def _build_vgg19(image: torch.Tensor, model_filepath: str) -> Dict[str, torch.Tensor]:
    net = {}
    if not hasattr(_build_vgg19, 'vgg_rawnet'):
        _build_vgg19.vgg_rawnet = sio.loadmat(model_filepath)
    vgg_layers = _build_vgg19.vgg_rawnet['layers'][0]
    imagenet_mean = torch.tensor([123.6800, 116.7790, 103.9390]).reshape(1, 3, 1, 1).to(device)
    net['input'] = image - imagenet_mean
    net['conv1_1'] = _build_net(
        'conv',
        net['input'],
        _get_weight_and_bias(vgg_layers, 0))
    net['conv1_2'] = _build_net(
        'conv',
        net['conv1_1'],
        _get_weight_and_bias(vgg_layers, 2))
    net['pool1'] = _build_net('pool', net['conv1_2'])
    net['conv2_1'] = _build_net(
        'conv',
        net['pool1'],
        _get_weight_and_bias(vgg_layers, 5))
    net['conv2_2'] = _build_net(
        'conv',
        net['conv2_1'],
        _get_weight_and_bias(vgg_layers, 7))
    net['pool2'] = _build_net('pool', net['conv2_2'])
    net['conv3_1'] = _build_net(
        'conv',
        net['pool2'],
        _get_weight_and_bias(vgg_layers, 10))
    net['conv3_2'] = _build_net(
        'conv',
        net['conv3_1'],
        _get_weight_and_bias(vgg_layers, 12))
    net['conv3_3'] = _build_net(
        'conv',
        net['conv3_2'],
        _get_weight_and_bias(vgg_layers, 14))
    net['conv3_4'] = _build_net(
        'conv',
        net['conv3_3'],
        _get_weight_and_bias(vgg_layers, 16))
    net['pool3'] = _build_net('pool', net['conv3_4'])
    net['conv4_1'] = _build_net(
        'conv',
        net['pool3'],
        _get_weight_and_bias(vgg_layers, 19))
    net['conv4_2'] = _build_net(
        'conv',
        net['conv4_1'],
        _get_weight_and_bias(vgg_layers, 21))
    net['conv4_3'] = _build_net(
        'conv',
        net['conv4_2'],
        _get_weight_and_bias(vgg_layers, 23))
    net['conv4_4'] = _build_net(
        'conv',
        net['conv4_3'],
        _get_weight_and_bias(vgg_layers, 25))
    net['pool4'] = _build_net('pool', net['conv4_4'])
    net['conv5_1'] = _build_net(
        'conv',
        net['pool4'],
        _get_weight_and_bias(vgg_layers, 28))
    net['conv5_2'] = _build_net(
        'conv',
        net['conv5_1'],
        _get_weight_and_bias(vgg_layers, 30))

    return net


def _compute_error(fake: torch.Tensor,
                   real: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        return torch.mean(torch.abs(fake - real))
    else:
        size = (fake.size(2), fake.size(3))
        resized_mask = F.interpolate(mask, size=size, mode='bilinear', align_corners=False)
        return torch.mean(torch.abs(fake - real) * resized_mask)


def perceptual_loss(image: torch.Tensor,
             reference: torch.Tensor,
             vgg_model_file: str,
             weights: Optional[Sequence[float]] = None,
             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if weights is None:
        weights = [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10.0 / 1.5]

    vgg_ref = _build_vgg19(reference * 255.0, vgg_model_file)
    vgg_img = _build_vgg19(image * 255.0, vgg_model_file)

    p1 = _compute_error(vgg_ref['conv1_2'], vgg_img['conv1_2'], mask) * weights[0]
    p2 = _compute_error(vgg_ref['conv2_2'], vgg_img['conv2_2'], mask) * weights[1]
    p3 = _compute_error(vgg_ref['conv3_2'], vgg_img['conv3_2'], mask) * weights[2]
    p4 = _compute_error(vgg_ref['conv4_2'], vgg_img['conv4_2'], mask) * weights[3]
    p5 = _compute_error(vgg_ref['conv5_2'], vgg_img['conv5_2'], mask) * weights[4]

    final_loss = p1 + p2 + p3 + p4 + p5
    final_loss /= 255.0

    return final_loss


def _compute_gram_matrix(input_features: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
    b, c, h, w = input_features.size()
    if mask is None:
        reshaped_features = input_features.view(b, c, h * w)
    else:
        resized_mask = F.interpolate(
            mask, size=(h, w), mode='bilinear', align_corners=False)
        reshaped_features = (input_features * resized_mask).view(b, c, h * w)
    return torch.matmul(
        reshaped_features, reshaped_features.transpose(1, 2)) / float(h * w)


def style_loss(image: torch.Tensor,
                reference: torch.Tensor,
                vgg_model_file: str,
                weights: Optional[Sequence[float]] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if not weights:
        weights = [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10.0 / 1.5]

    vgg_ref = _build_vgg19(reference * 255.0, vgg_model_file)
    vgg_img = _build_vgg19(image * 255.0, vgg_model_file)

    p1 = torch.mean(
        torch.square(
            _compute_gram_matrix(vgg_ref['conv1_2'] / 255.0, mask) -
            _compute_gram_matrix(vgg_img['conv1_2'] / 255.0, mask))) * weights[0]
    p2 = torch.mean(
        torch.square(
            _compute_gram_matrix(vgg_ref['conv2_2'] / 255.0, mask) -
            _compute_gram_matrix(vgg_img['conv2_2'] / 255.0, mask))) * weights[1]
    p3 = torch.mean(
        torch.square(
            _compute_gram_matrix(vgg_ref['conv3_2'] / 255.0, mask) -
            _compute_gram_matrix(vgg_img['conv3_2'] / 255.0, mask))) * weights[2]
    p4 = torch.mean(
        torch.square(
            _compute_gram_matrix(vgg_ref['conv4_2'] / 255.0, mask) -
            _compute_gram_matrix(vgg_img['conv4_2'] / 255.0, mask))) * weights[3]
    p5 = torch.mean(
        torch.square(
            _compute_gram_matrix(vgg_ref['conv5_2'] / 255.0, mask) -
            _compute_gram_matrix(vgg_img['conv5_2'] / 255.0, mask))) * weights[4]

    final_loss = p1 + p2 + p3 + p4 + p5
    return final_loss

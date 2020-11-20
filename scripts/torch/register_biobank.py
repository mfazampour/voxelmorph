#!/usr/bin/env python

"""
Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register a
scan (moving) to an atlas (fixed). To register a scan to the atlas and save the warp field, run:

    register.py --moving moving.nii.gz --fixed fixed.nii.gz --model model.pt --moved moved.nii.gz --warp warp.nii.gz

The source and target input images are expected to be affinely registered.
"""

import os
import argparse
import numpy as np
import nibabel as nib
import torch
import torchio
from torchio.transforms import (
    RescaleIntensity,
    Compose,
    CropOrPad
)

from monai.metrics import compute_meandice
from monai.metrics import compute_hausdorff_distance
from monai.metrics import compute_average_surface_distance

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm


def apply_model(model, inputs, y_true, device, losses=None, weights=None, is_test=False, has_seg=False):
    # generate inputs (and true outputs) and convert them to tensors
    if not isinstance(inputs[0], torch.Tensor):
        inputs = [torch.from_numpy(d).float().permute(0, 4, 1, 2, 3) for d in inputs]
        y_true = [torch.from_numpy(d).float().permute(0, 4, 1, 2, 3) for d in y_true]
    inputs = [t.to(device) for t in inputs]
    y_true = [t.to(device) for t in y_true]
    # run inputs through the model to produce a warped image and flow field
    if has_seg:
        y_pred = model(*inputs[:-1], registration=is_test)
    else:
        y_pred = model(*inputs, registration=is_test)

    if losses is None:
        return inputs, y_true, y_pred
    # calculate total loss
    loss = torch.tensor([0], dtype=torch.float).to(device)
    loss_list = []
    for n, loss_function in enumerate(losses):
        curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
        loss_list.append('%.6f' % curr_loss.item())
        loss += curr_loss
    return loss, loss_list, inputs, y_true, y_pred


def calc_scores(device, model, inputs, y_true, transformer, mask_values,
                args: argparse.Namespace, calc_statistics=False):
    reps = 1
    if calc_statistics:
        reps = args.num_statistics_runs
    dice_scores = []
    hd_scores = []
    asd_scores = []

    with torch.no_grad():
        for n in range(reps):
            asd_score, dice_score, hd_score = get_scores(device, mask_values, model, inputs, y_true, transformer)

            dice_scores.append(dice_score)
            hd_scores.append(hd_score)
            asd_scores.append(asd_score)
        # calculate mean and return if no calc_statistics
        dice_score = torch.cat(dice_scores).mean(dim=0, keepdim=True)
        hd_score = torch.cat(hd_scores).mean(dim=0, keepdim=True)
        asd_score = torch.cat(asd_scores).mean(dim=0, keepdim=True)
        if calc_statistics:
            dice_std = torch.cat(dice_scores).std(dim=0, keepdim=True)
            hd_std = torch.cat(hd_scores).std(dim=0, keepdim=True)
            asd_std = torch.cat(asd_scores).std(dim=0, keepdim=True)
        else:
            dice_std, hd_std, asd_std = (torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]))
        return dice_score, hd_score, asd_score, dice_std, hd_std, asd_std


def get_scores(device, mask_values, model, inputs, y_true, transformer):
    inputs, y_true, y_pred = apply_model(model, inputs, y_true, device=device,
                                         is_test=True, has_seg=True)
    seg_fixed = y_true[-1]
    seg_moving = inputs[-1]
    dvf = y_pred[-1].detach()
    morphed = transformer(seg_moving, dvf)
    morphed = morphed.round()
    shape = list(seg_fixed.shape)
    shape[1] = len(mask_values)
    one_hot_fixed = torch.zeros(shape, device=device)
    one_hot_morphed = torch.zeros(shape, device=device)
    for i, (val) in enumerate(mask_values):
        one_hot_fixed[:, i, seg_fixed[0, 0, ...] == val] = 1
        one_hot_morphed[:, i, morphed[0, 0, ...] == val] = 1
        seg_fixed[:, 0, seg_fixed[0, 0, ...] == val] = i
        morphed[:, 0, morphed[0, 0, ...] == val] = i
    dice_score = compute_meandice(one_hot_fixed, one_hot_morphed, to_onehot_y=False)
    hd_score = torch.zeros_like(dice_score)
    asd_score = torch.zeros_like(dice_score)
    for i in range(len(mask_values)):
        hd_score[0, i] = compute_hausdorff_distance(morphed, seg_fixed, i)
        asd_score[0, i] = compute_average_surface_distance(morphed, seg_fixed, i)
    return asd_score, dice_score, hd_score

def biobank_transform(target_shape=None, min_value=0, max_value=1):
    if min_value is None:
        transforms = []
    else:
        rescale = RescaleIntensity((min_value, max_value))
        transforms = [rescale]
    if target_shape is not None:
        transforms.append(CropOrPad(target_shape=target_shape))
    return Compose(transforms)


# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--moving', required=True, help='moving image (source) filename')
parser.add_argument('--fixed', required=True, help='fixed image (target) filename')
parser.add_argument('--moved', required=True, help='warped image output filename')
parser.add_argument('--model', required=True, help='pytorch model for nonlinear registration')
parser.add_argument('--moving-seg', default=None, type=str, help='path to the moving image segmentation file')
parser.add_argument('--fixed-seg', default=None, type=str, help='path to the fixed image segmentation file')
parser.add_argument('--moved-seg', default=None, type=str, help='path to save the moved image segmentation file')
parser.add_argument('--warp', help='output warp deformation filename')
parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('--multichannel', action='store_true', help='specify that data has multiple channels')
parser.add_argument('--inshape', type=int, nargs='+',
                        help='after cropping shape of input. '
                             'default is equal to image size. specify if the input can\'t path through UNet')
parser.add_argument('--use-probs', action='store_true', help='enable probabilities')
parser.add_argument('--num-statistics-runs', type=int, default=50,
                        help='number of runs to get each statistic')
args = parser.parse_args()

# device handling
if args.gpu and (args.gpu != '-1'):
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load moving and fixed images
moving = {'image': torchio.ScalarImage(args.moving)}
fixed = {'image': torchio.ScalarImage(args.fixed)}
if args.moving_seg and args.fixed_seg:
    moving['label'] = torchio.LabelMap(args.moving_seg)
    fixed['label'] = torchio.LabelMap(args.fixed_seg)

moving = torchio.Subject(moving)
fixed = torchio.Subject(fixed)

transform = biobank_transform(args.inshape)

moving = transform(moving)
fixed = transform(fixed)

# load and set up model
model = vxm.networks.VxmDense.load(args.model, device)
model.to(device)
model.eval()

# predict
moved, warp = model(moving.image.tensor.unsqueeze(dim=0).cuda(),
                    fixed.image.tensor.unsqueeze(dim=0).cuda(),
                    registration=True)

# save moved image
if args.moved:
    moved = moved.detach().cpu().numpy().squeeze()
    vxm.py.utils.save_volfile(moved, args.moved)

if args.moving_seg:
    transformer = vxm.layers.SpatialTransformer(size=args.inshape).to(device)
    morphed = transformer(moving.label.tensor.unsqueeze(dim=0).cuda(), warp)
    structures_dict = {0: 'backround',
            10: 'left_thalamus', 11: 'left_caudate', 12: 'left_putamen',
            13: 'left_pallidum', 16: 'brain_stem', 17: 'left_hippocampus',
            18: 'left_amygdala', 26: 'left_accumbens', 49: 'right_thalamus',
            50: 'right_caudate', 51: 'right_putamen', 52: 'right_pallidum',
            53: 'right_hippocampus', 54: 'right_amygdala', 58: 'right_accumbens'}
    mask_values = list(structures_dict.keys())
    shape = list(fixed.label.tensor.unsqueeze(dim=0).shape)
    shape[1] = len(mask_values)
    one_hot_fixed = torch.zeros(shape, device=device)
    one_hot_morphed = torch.zeros(shape, device=device)
    for i, (val) in enumerate(mask_values):
        one_hot_fixed[:, i, fixed.label.tensor.unsqueeze(dim=0)[0, 0, ...] == val] = 1
        one_hot_morphed[:, i, morphed[0, 0, ...] == val] = 1
    dice_score = compute_meandice(one_hot_fixed, one_hot_morphed, to_onehot_y=False).cpu()

    for i, (val) in enumerate(dice_score.squeeze()):
        print(f'dice score for {structures_dict[int(mask_values[i])]} is {val}')

# save warp
if args.warp:
    warp = warp.detach().cpu().numpy().squeeze()
    vxm.py.utils.save_volfile(warp, args.warp)

if args.use_prob and args.moving_seg:
    transformer = vxm.layers.SpatialTransformer(size=args.inshape).to(device)
    structures_dict = {0: 'backround',
                       10: 'left_thalamus', 11: 'left_caudate', 12: 'left_putamen',
                       13: 'left_pallidum', 16: 'brain_stem', 17: 'left_hippocampus',
                       18: 'left_amygdala', 26: 'left_accumbens', 49: 'right_thalamus',
                       50: 'right_caudate', 51: 'right_putamen', 52: 'right_pallidum',
                       53: 'right_hippocampus', 54: 'right_amygdala', 58: 'right_accumbens'}
    mask_values = list(structures_dict.keys())
    dice_score, hd_score, asd_score, dice_std, hd_std, asd_std = \
        calc_scores(device, model, moving.image.tensor.unsqueeze(dim=0).cuda(),
                    fixed.image.tensor.unsqueeze(dim=0).cuda(), transformer=transformer,
                    mask_values=mask_values, args=args)

    for i, (d, d_std, a, a_std) in enumerate(zip(dice_score.squeeze(), dice_std.squeeze(),
                                                 asd_score.squeeze(), asd_std.squeeze())):
        print(f'dice score for {structures_dict[int(mask_values[i])]} is {d}')
        print(f'dice score std for {structures_dict[int(mask_values[i])]} is {d_std}')
        print(f'MSD for {structures_dict[int(mask_values[i])]} is {a}')
        print(f'MSD std for {structures_dict[int(mask_values[i])]} is {a_std}')





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

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm

from scripts.torch.utils import calc_scores


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

structures_dict = {0: 'backround',
                10: 'left_thalamus', 11: 'left_caudate', 12: 'left_putamen',
                13: 'left_pallidum', 16: 'brain_stem', 17: 'left_hippocampus',
                18: 'left_amygdala', 26: 'left_accumbens', 49: 'right_thalamus',
                50: 'right_caudate', 51: 'right_putamen', 52: 'right_pallidum',
                53: 'right_hippocampus', 54: 'right_amygdala', 58: 'right_accumbens'}

# predict
with torch.no_grad():
    moved, warp = model(moving.image.tensor.unsqueeze(dim=0).cuda(),
                        fixed.image.tensor.unsqueeze(dim=0).cuda(),
                        registration=True)

    # save moved image
    if args.moved:
        moved = moved.detach().cpu().numpy().squeeze()
        vxm.py.utils.save_volfile(moved, args.moved)

    if args.moving_seg:
        transformer = vxm.layers.SpatialTransformer(size=args.inshape, mode='nearest').to(device)
        morphed = transformer(moving.label.tensor.unsqueeze(dim=0).cuda(), warp)
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

if args.use_probs and args.moving_seg:
    transformer = vxm.layers.SpatialTransformer(size=args.inshape, mode='nearest').to(device)
    mask_values = list(structures_dict.keys())
    input_ = [moving.image.tensor.unsqueeze(dim=0).cuda(),
                fixed.image.tensor.unsqueeze(dim=0).cuda(), moving.label.tensor.unsqueeze(dim=0).cuda()]
    y_true = [fixed.label.tensor.unsqueeze(dim=0).cuda()]
    with torch.no_grad():
        dice_score, hd_score, asd_score, dice_std, hd_std, asd_std = \
            calc_scores(device, mask_values, model, transformer=transformer, inputs=input_,
                        y_true=y_true, num_statistics_runs=args.num_statistics_runs, calc_statistics=True)

    print('---------------------------------------')
    print('stats')
    print('---------------------------------------')
    for i, (d, d_std, a, a_std) in enumerate(zip(dice_score.squeeze(), dice_std.squeeze(),
                                                 asd_score.squeeze(), asd_std.squeeze())):
        print(f'Dice, {structures_dict[int(mask_values[i])]}, {d}, {d_std}')
        print(f'MSD, {structures_dict[int(mask_values[i])]}, {a}, {a_std}')




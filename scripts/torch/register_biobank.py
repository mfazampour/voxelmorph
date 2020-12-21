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
    CropOrPad,
    Resample
)
import SimpleITK as sitk
# from tvtk.api import tvtk, write_data

from monai.metrics import compute_meandice, compute_hausdorff_distance, compute_average_surface_distance

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm

from scripts.torch.utils import calc_scores
from scripts.torch.utils import create_toy_sample


def biobank_transform(target_shape=None, min_value=0, max_value=1, target_spacing=None):
    if min_value is None:
        transforms = []
    else:
        rescale = RescaleIntensity((min_value, max_value))
        transforms = [rescale]
    if target_spacing is not None:
        transforms.append(Resample(target_spacing))
    if target_shape is not None:
        transforms.append(CropOrPad(target_shape=target_shape))
    return Compose(transforms)

# def save_field_to_disk(field, file_path):
#     field = field.cpu().numpy()
#
#     field_x = field[0]
#     field_y = field[1]
#     field_z = field[2]
#
#     dim = field.shape[1:]  # FIXME: why is the no. cells < no. points?
#
#     vectors = np.empty(field_x.shape + (3,), dtype=float)
#     vectors[..., 0] = field_x
#     vectors[..., 1] = field_y
#     vectors[..., 2] = field_z
#
#     vectors = vectors.transpose(2, 1, 0, 3).copy()
#     vectors.shape = vectors.size // 3, 3
#
#     im_vtk = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0), dimensions=dim)
#     im_vtk.point_data.vectors = vectors
#     im_vtk.point_data.vectors.name = 'field'
#
#     write_data(im_vtk, file_path)


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
parser.add_argument('--target-spacing', type=float, default=1, help='target spacing of the inputs to the network')
parser.add_argument('--use-probs', action='store_true', help='enable probabilities')
parser.add_argument('--num-statistics-runs', type=int, default=50,
                        help='number of runs to get each statistic')
parser.add_argument('--use-toy', action='store_true', help='create a toy example out of moving before registration')
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

trasform = biobank_transform(target_shape=args.inshape, target_spacing=args.target_spacing)
full_size = biobank_transform(target_spacing=1.0, min_value=None)
repad = biobank_transform(target_shape=moving.shape[-3:], min_value=None)

moving = trasform(moving)
fixed = trasform(fixed)

moving_fs = full_size(moving)
fixed_fs = full_size(fixed)

full_size_vxm = vxm.layers.ResizeTransform(0.5, 3)

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
    if args.use_toy:
        toy = create_toy_sample(moving.image.tensor, mask=moving.label.tensor, method='noise', num_changes=5)
        moved, warp = model(toy.unsqueeze(dim=0).cuda(),
                            fixed.image.tensor.unsqueeze(dim=0).cuda(),
                            registration=True)
    else:
        moved, warp = model(moving.image.tensor.unsqueeze(dim=0).cuda(),
                            fixed.image.tensor.unsqueeze(dim=0).cuda(),
                            registration=True)

    # save moved image
    if args.moved:
        img = torchio.ScalarImage(tensor=moved[0, ...].cpu())
        img = repad(img)
        img.save(args.moved)

    if args.moving_seg:
        transformer = vxm.layers.SpatialTransformer(size=moving.shape[-3:], mode='nearest').to(device)
        morphed = transformer(moving.label.tensor.unsqueeze(dim=0).cuda(), warp)
        morphed_im = torchio.LabelMap(tensor=morphed.cpu().squeeze(0), affine=moving.image.affine)
        morphed = full_size(morphed_im).data.to(device).unsqueeze(0)
        mask_values = list(structures_dict.keys())
        shape = list(fixed_fs.label.tensor.unsqueeze(dim=0).shape)
        shape[1] = len(mask_values)
        one_hot_fixed = torch.zeros(shape, device=device)
        one_hot_morphed = torch.zeros(shape, device=device)
        seg_fixed = fixed_fs.label.tensor.unsqueeze(dim=0)
        for i, (val) in enumerate(mask_values):
            one_hot_fixed[:, i, seg_fixed[0, 0, ...] == val] = 1
            one_hot_morphed[:, i, morphed[0, 0, ...] == val] = 1
            seg_fixed[:, 0, seg_fixed[0, 0, ...] == val] = i
            morphed[:, 0, morphed[0, 0, ...] == val] = i
        dice_score = compute_meandice(one_hot_fixed, one_hot_morphed, to_onehot_y=False).cpu()

        hd_score = torch.zeros_like(dice_score)
        asd_score = torch.zeros_like(dice_score)
        for i in range(len(mask_values)):
            # hd_score[0, i] = compute_hausdorff_distance(morphed, seg_fixed, i)
            asd_score[0, i] = compute_average_surface_distance(morphed, seg_fixed, i)

        for i, (dice, asd) in enumerate(zip(dice_score.squeeze(), asd_score.squeeze())):
            print(f'dice score for {structures_dict[int(mask_values[i])]} is {dice}')
            print(f'asd for {structures_dict[int(mask_values[i])]} is {asd}')

    # save warp
    if args.warp:
        img = torchio.ScalarImage(tensor=warp[0, ...].cpu())
        img = repad(img)
        img.save(args.warp)

if args.use_probs and args.moving_seg:
    transformer = vxm.layers.SpatialTransformer(size=args.inshape, mode='nearest').to(device)
    mask_values = list(structures_dict.keys())
    if args.use_toy:
        input_ = [toy.unsqueeze(dim=0).cuda(),
                  fixed.image.tensor.unsqueeze(dim=0).cuda(), moving.label.tensor.unsqueeze(dim=0).cuda()]
    else:
        input_ = [moving.image.tensor.unsqueeze(dim=0).cuda(),
                    fixed.image.tensor.unsqueeze(dim=0).cuda(), moving.label.tensor.unsqueeze(dim=0).cuda()]
    y_true = [fixed.label.tensor.unsqueeze(dim=0).cuda()]
    with torch.no_grad():
        dice_score, hd_score, asd_score, dice_std, hd_std, asd_std, seg_maps, dvfs = \
            calc_scores(device, mask_values, model, transformer=transformer, inputs=input_,
                        y_true=y_true, num_statistics_runs=args.num_statistics_runs, calc_statistics=True,
                        affine=moving.image.affine)

    print('---------------------------------------')
    print('stats')
    print('---------------------------------------')
    for i, (d, d_std, a, a_std) in enumerate(zip(dice_score.squeeze(), dice_std.squeeze(),
                                                 asd_score.squeeze(), asd_std.squeeze())):
        print(f'Dice, {structures_dict[int(mask_values[i])]}, {d}, {d_std}')
        print(f'MSD, {structures_dict[int(mask_values[i])]}, {a}, {a_std}')

    for i, (ddf) in enumerate(dvfs):
        jacob = vxm.py.utils.jacobian_determinant(ddf[0, ...].permute(*range(1, len(ddf.shape) - 1), 0).cpu().numpy())
        print(f'jacob/negative count, {jacob[jacob < 0].size}, sample, {i}')
        print(f'jacob/negative ratio, {jacob[jacob < 0].size / jacob.size}, sample, {i}')
        img = sitk.GetImageFromArray(ddf[0, ...].cpu().permute(1, 2, 3, 0))
        sitk.WriteImage(img, f'/tmp/ddf{i}.mhd')
        # save_field_to_disk(ddf[0, ...].cpu(), f'/tmp/ddf{i}.nii.gz')
        # remove the flag for BinaryData from the mhd file :( :strange:







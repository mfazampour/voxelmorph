#!/usr/bin/env python

import os
import argparse
import sys
import pathlib

import numpy as np
import pandas as pd
import torch

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.absolute()))  # add vxm path
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.absolute()
                    / 'voxelmorph' / 'torch' / 'learnsim'))  # add learnsim path
# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm

from scripts.torch.utils_scripts import calc_scores
from scripts.torch.utils_scripts import create_toy_sample

def main():
    parser = parse_args()
    args = parser.parse_args()
    args.affine = np.diag([*[args.target_spacing] * 3, 1])

    # device handling
    if args.gpu and (args.gpu != '-1'):
        device = 'cuda'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        device = 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # load and set up model
    model_base = vxm.networks.VxmDense.load(args.model_base, device)
    model_base.to(device)
    model_base.eval()

    model_test = vxm.networks.VxmDense.load(args.model_test, device)
    model_test.to(device)
    model_test.eval()

    transformer = vxm.layers.SpatialTransformer(size=args.inshape, mode='nearest').to(device)
    resizer = vxm.layers.ResizeTransform(vel_resize=1 / args.target_spacing, ndims=3)

    structures_dict = {0: 'backround',
                       10: 'left_thalamus', 11: 'left_caudate', 12: 'left_putamen',
                       13: 'left_pallidum', 16: 'brain_stem', 17: 'left_hippocampus',
                       18: 'left_amygdala', 26: 'left_accumbens', 49: 'right_thalamus',
                       50: 'right_caudate', 51: 'right_putamen', 52: 'right_pallidum',
                       53: 'right_hippocampus', 54: 'right_amygdala', 58: 'right_accumbens'}

    generator = create_data_generator(args, is_train=False)

    df_DSC = pd.DataFrame(columns=["DSC", "im_pair", "structure", 'method'])
    df_ASD = pd.DataFrame(columns=["ASD", "im_pair", "structure", 'method'])
    df_HD = pd.DataFrame(columns=["HD", "im_pair", "structure", 'method'])
    df_Jac = pd.DataFrame(columns=["count", 'ratio', "im_pair", 'method'])


    with torch.no_grad():
        for im_num in range(args.num_test_imgs):
            inputs, y_true = next(generator)
            if not isinstance(inputs[0], torch.Tensor):
                inputs = [torch.from_numpy(d).float().permute(0, 4, 1, 2, 3) for d in inputs]
                y_true = [torch.from_numpy(d).float().permute(0, 4, 1, 2, 3) for d in y_true]
            inputs = [t.to(device) for t in inputs]
            y_true = [t.to(device) for t in y_true]

            moving = inputs[0]
            fixed = y_true[0]
            moving_label = inputs[-1]
            fixed_label = y_true[-1]

            if args.use_toy:
                toy = create_toy_sample(moving, mask=moving_label, method='noise', num_changes=5)

            mask_values = list(structures_dict.keys())
            if args.use_toy:
                input_ = [toy, fixed, moving_label]
            else:
                input_ = [moving, fixed, moving_label]

            df_ASD, df_DSC, df_HD, df_Jac = get_stats(input_, fixed_label, mask_values, model_base, args, device, df_ASD,
                                                      df_DSC, df_HD, df_Jac, im_num, resizer, structures_dict,
                                                      transformer, args.method_base)

            df_ASD, df_DSC, df_HD, df_Jac = get_stats(input_, fixed_label, mask_values, model_test, args, device,
                                                      df_ASD, df_DSC, df_HD, df_Jac, im_num, resizer, structures_dict,
                                                      transformer, args.method_test)

            torch.cuda.empty_cache()


    os.makedirs(args.output_dir, exist_ok=True)
    df_Jac.to_csv(os.path.join(args.output_dir, 'jac.csv'))
    df_DSC.to_csv(os.path.join(args.output_dir, 'dsc.csv'))
    df_ASD.to_csv(os.path.join(args.output_dir, 'asd.csv'))
    df_HD.to_csv(os.path.join(args.output_dir, 'hd.csv'))


def get_stats(input_, fixed_label, mask_values, model, args, device, df_ASD, df_DSC, df_HD, df_Jac, im_num, resizer,
              structures_dict, transformer, method):
    dice_scores, hd_scores, asd_scores, dice_std, hd_std, asd_std, seg_maps, dvfs = \
        calc_scores(device, mask_values, model, transformer=transformer, inputs=input_,
                    y_true=[fixed_label], num_statistics_runs=args.num_statistics_runs, calc_statistics=True,
                    affine=args.affine, resize_module=resizer)
    dice_score = dice_scores.mean(dim=0, keepdim=True)
    hd_score = hd_scores.mean(dim=0, keepdim=True)
    asd_score = asd_scores.mean(dim=0, keepdim=True)
    df_DSC = add_to_data_frame(dice_scores.cpu(), df_DSC, 'DSC', im_num=im_num, structures_dict=structures_dict, method=method)
    df_HD = add_to_data_frame(hd_scores.cpu(), df_HD, 'HD', im_num=im_num, structures_dict=structures_dict, method=method)
    df_ASD = add_to_data_frame(asd_scores.cpu(), df_ASD, 'ASD', im_num=im_num, structures_dict=structures_dict, method=method)
    print('---------------------------------------')
    print('stats')
    print('---------------------------------------')
    for i, (d, d_std, a, a_std) in enumerate(zip(dice_score.squeeze(), dice_std.squeeze(),
                                                 asd_score.squeeze(), asd_std.squeeze())):
        print(f'Dice, {structures_dict[int(mask_values[i])]}, {d}, {d_std}')
        print(f'MSD, {structures_dict[int(mask_values[i])]}, {a}, {a_std}')

    j_count = []
    j_ratio = []
    for i, (ddf) in enumerate(dvfs):
        jacob = vxm.py.utils.jacobian_determinant(
            ddf[0, ...].permute(*range(1, len(ddf.shape) - 1), 0).cpu().numpy())
        c = jacob[jacob < 0].size
        r = c / jacob.size
        print(f'jacob negative count, {c}, sample, {i}')
        print(f'jacob negative ratio, {r}, sample, {i}')
        j_count.append(c)
        j_ratio.append(r)
    n = args.num_statistics_runs
    scores_dict = {'im_pair': [im_num] * n, "count": j_count, 'ratio': j_ratio, 'method': [method] * n}
    tmp_df = pd.DataFrame.from_dict(scores_dict)
    df_Jac = df_Jac.append(tmp_df, ignore_index=True)
    return df_ASD, df_DSC, df_HD, df_Jac


def parse_args():
    parser = argparse.ArgumentParser()

    # data organization parameters
    parser.add_argument('datadir', help='base data directory')
    parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
    parser.add_argument('--model-base', required=True, help='baseline model for nonlinear registration')
    parser.add_argument('--model-test', required=True, help='test model for nonlinear registration')
    parser.add_argument('--multichannel', action='store_true', help='specify that data has multiple channels')
    parser.add_argument("--output-dir", required=True, help="directory to dave the results")

    # training parameters
    parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
    parser.add_argument('--epochs', type=int, default=1500, help='number of training epochs (default: 1500)')
    parser.add_argument('--steps-per-epoch', type=int, default=100, help='frequency of model saves (default: 100)')
    parser.add_argument('--load-model', help='optional model file to initialize with')
    parser.add_argument('--initial-epoch', type=int, default=0, help='initial epoch number (default: 0)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--cudnn-nondet', action='store_true', help='disable cudnn determinism - might slow down training')

    # network architecture parameters
    parser.add_argument('--inshape', type=int, nargs='+', help='after cropping shape of input. default is equal to image size. specify if the input can\'t path through UNet')
    parser.add_argument('--target-spacing', type=float, default=1, help='target spacing of the inputs to the network')
    parser.add_argument('--final-spacing', type=float, default=2.43, help='final spacing of the saved images')

    # loading and saving parameters
    parser.add_argument('--loader-name', type=str, default='biobank', help='volume generator function to use')
    parser.add_argument('--patient-list-src', type=str, default='/tmp/', help='directory to store patient list for training and testing')
    parser.add_argument('--load-segmentation', action='store_true', default=False, help='use segmentation data for training the network (torch functionality seems to be missing)')
    parser.add_argument('--num-statistics-runs', type=int, default=5, help='number of runs to get each statistic')
    parser.add_argument('--num-test-imgs', type=int, default=10, help='number of test images to compare the results with previous one')

    parser.add_argument('--use-probs', action='store_true', help='enable probabilities')
    parser.add_argument('--use-toy', action='store_true', help='create a toy example out of moving before registration')
    parser.add_argument("--sampling-speed", action='store_true', help='measure the sampling through 10k runs')
    parser.add_argument('--method-base', type=str, default='voxelmorph(baseline, SSD)', help='saved name in the csv file')
    parser.add_argument('--method-test', type=str, default='voxelmorph(learnt)', help='saved name in the csv file')
    return parser


def create_data_generator(args, is_train=True):
    train_vol_names = args.datadir
    # no need to append an extra feature axis if data is multichannel
    add_feat_axis = not args.multichannel
    if 'inshape' not in args:
        args.inshape = None
    if args.atlas:
        generator = vxm.generators.scan_to_atlas_biobank(source_folder=train_vol_names, atlas=args.atlas,
                                                         batch_size=args.batch_size,
                                                         bidir=False, add_feat_axis=add_feat_axis,
                                                         target_shape=args.inshape,
                                                         return_segs=not is_train, target_spacing=args.target_spacing,
                                                         patient_list_src=args.patient_list_src, is_train=is_train)
        args.mask = next(generator)

    else:
        # scan-to-scan generator
        return_segs = args.load_segmentation
        if is_train is False:
            # we want to compare segmentation in evaluation/test
            return_segs = True
        generator = vxm.generators.scan_to_scan(train_vol_names, batch_size=args.batch_size,
                                                bidir=False, add_feat_axis=add_feat_axis,
                                                loader_name=args.loader_name, target_shape=args.inshape,
                                                return_segs=return_segs, target_spacing=args.target_spacing,
                                                patient_list_src=args.patient_list_src, is_train=is_train)
    return generator


def add_to_data_frame(scores: torch.Tensor, df: pd.DataFrame, score_type: str, im_num: int, structures_dict, method: str):
    n = scores.shape[0]
    for i, (key) in enumerate(structures_dict):
        if 'background' in structures_dict[key]:
            continue
        scores_dict = {score_type: scores[:, i].numpy(), 'im_pair': [im_num] * n,
                       'structure': [structures_dict[key]] * n, 'method': [method] * n}
        tmp_df = pd.DataFrame.from_dict(scores_dict)
        df = df.append(tmp_df, ignore_index=True)
    return df


if __name__ == '__main__':
    main()
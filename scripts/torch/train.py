#!/usr/bin/env python

"""
Example script to train a VoxelMorph model.

For the CVPR and MICCAI papers, we have data arranged in train, validate, and test folders. Inside each folder
are normalized T1 volumes and segmentations in npz (numpy) format. You will have to customize this script slightly
to accommodate your own data. All images should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed. Otherwise,
registration will be scan-to-scan.
"""

import os
import random
import argparse
import glob
import time
from datetime import datetime
import pathlib

import numpy as np
import torch
# import napari
from GPUtil import GPUtil
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from matplotlib import gridspec

import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.absolute()))  # add vxm path
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.absolute()
                    / 'voxelmorph' / 'torch' / 'learnsim'))  # add learnsim path

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
from scripts.torch.utils_scripts import apply_model
from scripts.torch.utils_scripts import calc_scores


def main():
    parser = parse_args()
    args = parser.parse_args()

    # prepare model folder
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)

    print_options(parser, args)

    bidir = args.bidir

    # tensorboard
    if args.log_dir is None:
        args.log_dir = args.model_dir
    writer = SummaryWriter(log_dir=f'{args.log_dir}/{datetime.now().strftime("%d.%m.%Y_%H.%M")}')

    log_input_params(args, writer)

    generator = create_data_generator(args)
    args_test = argparse.Namespace(**vars(args))
    args_test.batch_size = 1
    test_generator = create_data_generator(args_test, is_train=False)

    # extract shape from sampled input
    inshape = args.inshape

    # device handling
    gpus = args.gpu.split(',')
    nb_gpus = len(gpus)
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # assert args.batch_size >= nb_gpus, 'Batch size (%d) should be no less than the number of gpus (%d)' % (
    #     args.batch_size, nb_gpus)

    model = create_model(args, bidir, device, inshape, nb_gpus)

    losses, optimizer, weights, loss_names = create_optimizers(args, bidir, model, device)
    # training loops
    train(args, device, generator, losses, model, model_dir, optimizer, weights, writer, loss_names, test_generator)


def parse_args() -> argparse.ArgumentParser:
    # parse the commandline
    parser = argparse.ArgumentParser()

    # data organization parameters
    parser.add_argument('datadir', help='base data directory')
    parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
    parser.add_argument('--model-dir', default='models', help='model output directory (default: models)')
    parser.add_argument('--multichannel', action='store_true', help='specify that data has multiple channels')

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
    parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters (default: 16 32 32 32)')
    parser.add_argument('--dec', type=int, nargs='+', help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
    parser.add_argument('--int-steps', type=int, default=7, help='number of integration steps (default: 7)')
    parser.add_argument('--int-downsize', type=int, default=2, help='flow downsample factor for integration (default: 2)')
    parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')
    parser.add_argument('--inshape', type=int, nargs='+', help='after cropping shape of input. default is equal to image size. specify if the input can\'t path through UNet')
    parser.add_argument('--target-spacing', type=float, default=1, help='target spacing of the inputs to the network')

    # loss hyperparameters
    parser.add_argument('--image-loss', default='mse', help='image reconstruction loss - can be mse or ncc (default: mse)')
    parser.add_argument('--image-sigma', default=1.0, type=float, help='sigma for the mse loss')
    parser.add_argument('--lambda', type=float, dest='weight', default=0.01, help='weight of deformation loss (default: 0.01)')
    parser.add_argument('--use-probs', action='store_true', help='enable probabilities')
    parser.add_argument('--kl-lambda', type=float, default=10, help='prior lambda regularization for KL loss (default: 10)')
    parser.add_argument('--flow-logsigma-bias', type=float, default=-10, help='negative value for initialization of the logsigma layer bias value')

    # loading and saving parameters
    parser.add_argument('--log-dir', type=str, default=None, help='folder for tensorboard logs')
    parser.add_argument('--loader-name', type=str, default='default', help='volume generator function to use')
    parser.add_argument('--patient-list-src', type=str, default='/tmp/', help='directory to store patient list for training and testing')
    parser.add_argument('--load-segmentation', action='store_true', default=False, help='use segmentation data for training the network (torch functionality seems to be missing)')
    parser.add_argument('--display-freq', type=int, default=20, help='frequency of plotting results in tensorboard')
    parser.add_argument('--statistics-freq', type=int, default=10, help='frequency/period of plotting the statistics per number of displaying results')
    parser.add_argument('--num-statistics-runs', type=int, default=5, help='number of runs to get each statistic')
    parser.add_argument('--num-test-imgs', type=int, default=10, help='number of test images to compare the results with previous one')
    return parser


def print_options(parser, opt):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    expr_dir = os.path.join(opt.model_dir)
    file_name = os.path.join(expr_dir, 'params.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


def create_data_generator(args, is_train=True):
    if args.loader_name is not 'default':
        train_vol_names = args.datadir
    else:
        # load and prepare training data
        train_vol_names = glob.glob(os.path.join(args.datadir, '*.npz'))
        random.shuffle(train_vol_names)  # shuffle volume list
        assert len(train_vol_names) > 0, 'Could not find any training data'
    # no need to append an extra feature axis if data is multichannel
    add_feat_axis = not args.multichannel
    if 'inshape' not in args:
        args.inshape = None
    if args.atlas:
        # scan-to-atlas generator
        atlas = vxm.py.utils.load_volfile(args.atlas, np_var='vol', add_batch_axis=True, add_feat_axis=add_feat_axis)
        generator = vxm.generators.scan_to_atlas(train_vol_names, atlas, batch_size=args.batch_size, bidir=args.bidir,
                                                 add_feat_axis=add_feat_axis)
    else:
        # scan-to-scan generator
        return_segs = args.load_segmentation
        if is_train is False:
            # we want to compare segmentation in evaluation/test
            return_segs = True
        generator = vxm.generators.scan_to_scan(train_vol_names, batch_size=args.batch_size,
                                                bidir=args.bidir, add_feat_axis=add_feat_axis,
                                                loader_name=args.loader_name, target_shape=args.inshape,
                                                return_segs=return_segs, target_spacing=args.target_spacing,
                                                patient_list_src=args.patient_list_src, is_train=is_train)
    return generator


def create_model(args, bidir, device, inshape, nb_gpus):
    # enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = not args.cudnn_nondet
    # unet architecture
    enc_nf = args.enc if args.enc else [16, 32, 32, 32]
    dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]
    if args.load_model:
        # load initial model (if specified)
        model = vxm.networks.VxmDense.load(args.load_model, device)
    else:
        # otherwise configure new model
        model = vxm.networks.VxmDense(
            inshape=inshape,
            nb_unet_features=[enc_nf, dec_nf],
            bidir=bidir,
            int_steps=args.int_steps,
            int_downsize=args.int_downsize,
            use_probs=args.use_probs,
            flow_logsigma_bias=args.flow_logsigma_bias
        )
    # if nb_gpus > 1:
    #     # use multiple GPUs via DataParallel
    model = torch.nn.DataParallel(model)
    model.save = model.module.save
    # prepare the model for training and send to device
    model.to(device)
    model.train()
    return model


def create_optimizers(args, bidir, model, device):
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # prepare image loss
    loss_names = []
    available_loss_images = ['ncc', 'mse', 'ssim', 'mind', 'learnsim', 'lcc']
    if args.image_loss == 'ncc':
        image_loss_func = vxm.losses.NCC().loss
    elif args.image_loss == 'mse':
        image_loss_func = vxm.losses.MSE(args.image_sigma).loss
    elif args.image_loss == 'ssim':
        image_loss_func = vxm.losses.SSIM().loss
    elif args.image_loss == 'mind':
        image_loss_func = vxm.losses.MIND().loss
    elif args.image_loss == 'lcc':
        image_loss_func = vxm.losses.LCC(s=4, device=device).loss
    else:
        raise ValueError(f'Image loss should be among {available_loss_images}, but found {args.image_loss}')
    # need two image loss functions if bidirectional
    if bidir:
        losses = [image_loss_func, image_loss_func]
        loss_names += [args.image_loss, args.image_loss]
        weights = [0.5, 0.5]
    else:
        losses = [image_loss_func]
        loss_names += [args.image_loss]
        weights = [1]

    # prepare deformation loss
    if args.use_probs:
        losses += [vxm.losses.KL(args.kl_lambda).loss]
        loss_names += ['KL']
    else:
        losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
        loss_names += ['Regularization']
    weights += [args.weight]
    return losses, optimizer, weights, loss_names


def train(args: argparse.Namespace, device, generator, losses, model, model_dir, optimizer, weights, writer, loss_names,
          test_generator):
    ssim = vxm.losses.SSIM()
    transformer = vxm.layers.SpatialTransformer(size=args.inshape, mode='nearest').to(device)
    display_count = 0

    for epoch in range(args.initial_epoch, args.epochs):
        # save model checkpoint
        model.save(os.path.join(model_dir, '%04d.pt' % epoch))

        for step in range(args.steps_per_epoch):
            step_start_time = time.time()

            loss, loss_list, _, _, y_pred = apply_model(model=model, generator=generator, device=device,
                                                        losses=losses, weights=weights)
            loss_info = 'loss: %.6f  (%s)' % (loss.item(), ', '.join(loss_list))

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print step info
            epoch_info = 'epoch: %04d' % (epoch + 1)
            step_info = ('step: %d/%d' % (step + 1, args.steps_per_epoch)).ljust(14)
            time_info = 'time: %.2f sec' % (time.time() - step_start_time)
            print('  '.join((epoch_info, step_info, time_info, loss_info)), flush=True)

            # tensorboard logging & evaluation
            global_step = (epoch) * args.steps_per_epoch + step + 1
            if global_step % args.display_freq == 1:
                calc_statistics = False
                if (display_count % args.statistics_freq == 0) and args.use_probs:
                    calc_statistics = True

                display_count += 1
                model.eval()
                if args.use_probs:
                    mean = y_pred[-1][0:1, :3, ...].detach().cpu()
                    log_sigma = y_pred[-1][0:1, 3:, ...].detach().cpu()
                else:
                    log_sigma = None
                    mean = None
                tensorboard_log(model, test_generator, loss_names, device, loss_list, writer, ssim=ssim,
                                log_sigma=log_sigma, mean=mean, global_step=global_step)
                evaluate_with_segmentation(model, test_generator, device=device, args=args, writer=writer,
                                           global_step=global_step, transformer=transformer,
                                           calc_statistics=calc_statistics)
                model.train()
        torch.cuda.empty_cache()
        GPUtil.showUtilization()
    # final model save
    model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))


def tensorboard_log(model, test_generator, loss_names, device, loss_list,
                    writer: SummaryWriter, ssim: vxm.losses.SSIM, log_sigma=None, mean=None, global_step=0):
    with torch.no_grad():
        inputs, y_true, y_pred = apply_model(model=model, generator=test_generator, device=device,
                                             is_test=True, has_seg=True)
    ddf = y_pred[-1].detach()
    y_pred = y_pred[0]
    jacob = vxm.py.utils.jacobian_determinant(ddf[0, ...].permute(*range(1, len(ddf.shape) - 1), 0).cpu().numpy())
    figure = vxm.torch.utils.create_figure(y_true[0].cpu(), inputs[0].cpu(), y_pred.cpu(),
                                           jacob=torch.tensor(jacob).view_as(y_true[0]),
                                           deformation=ddf.cpu(), log_sigma=log_sigma, mean=mean)
    writer.add_figure(tag='volumes',
                      figure=figure,
                      global_step=global_step)
    for name, value in zip(loss_names, list(map(float, loss_list))):
        writer.add_scalar(f'loss/{name}', value, global_step=global_step)

    writer.add_scalar(f'jacob/negative count', jacob[jacob < 0].size, global_step=global_step)
    writer.add_scalar(f'jacob/negative ratio', jacob[jacob < 0].size / jacob.size, global_step=global_step)

    fix_to_mov = torch.mean((y_true[0][y_true[0] != 0] - inputs[0][y_true[0] != 0]) ** 2).cpu()
    fix_to_reg = torch.mean((y_true[0][y_true[0] != 0] - y_pred[y_true[0] != 0]) ** 2).cpu()
    ssim_mov = ssim.loss(y_true[0], inputs[0]).item()
    ssim_reg = ssim.loss(y_true[0], y_pred).item()
    ssim_increment = ssim_reg / (ssim_mov + 0.001) - 1
    diff_dict = {'Fix. to Mov.': fix_to_mov.item(),
                 'Fix. to Reg.': fix_to_reg.item(),
                 'SSD improvement': 1 - (fix_to_reg.item() / (fix_to_mov.item() + 1e-9)),
                 'SSIM Mov.': ssim_mov,
                 'SSIM Reg.': ssim_reg,
                 'SSIM increment': ssim_increment}
    for key in diff_dict:
        writer.add_scalar(f'diffs/{key}', diff_dict[key], global_step=global_step)


def evaluate_with_segmentation(model, test_generator, device, args: argparse.Namespace, writer: SummaryWriter,
                               transformer, global_step=0, calc_statistics=False):
    list_dice = []
    list_hd = []
    list_asd = []
    list_dice_std = []
    list_hd_std = []
    list_asd_std = []
    # mask_values = [0, 10, 11, 12, 13, 16, 17, 18, 26, 49, 50, 51, 52, ]
    structures_dict = {0: 'backround',
                       10: 'left_thalamus', 11: 'left_caudate', 12: 'left_putamen',
                       13: 'left_pallidum', 16: 'brain_stem', 17: 'left_hippocampus',
                       18: 'left_amygdala', 26: 'left_accumbens', 49: 'right_thalamus',
                       50: 'right_caudate', 51: 'right_putamen', 52: 'right_pallidum',
                       53: 'right_hippocampus', 54: 'right_amygdala', 58: 'right_accumbens'}
    mask_values = list(structures_dict.keys())
    seg_maps = []
    for step in range(args.num_test_imgs):
        print(step)
        # generate scores for logging on tensorboard
        dice_scores, hd_scores, asd_scores, dice_std, hd_std, asd_std, seg_maps, _ = calc_scores(device, mask_values,
                                                                                              model,
                                                                                              transformer,
                                                                                              test_generator=test_generator,
                                                                                              num_statistics_runs=args.num_statistics_runs,
                                                                                              calc_statistics=calc_statistics)
        dice_score = dice_scores.mean(dim=0, keepdim=True)
        hd_score = hd_scores.mean(dim=0, keepdim=True)
        asd_score = asd_scores.mean(dim=0, keepdim=True)
        list_dice.append(dice_score.cpu())
        list_hd.append(hd_score.cpu())
        list_asd.append(asd_score.cpu())
        list_dice_std.append(dice_std.cpu())
        list_hd_std.append(hd_std.cpu())
        list_asd_std.append(asd_std.cpu())
        torch.cuda.empty_cache()

    mean_dice = torch.cat(list_dice).mean(dim=0)
    for i, (val) in enumerate(mean_dice):
        writer.add_scalar(f'dice/{structures_dict[int(mask_values[i])]}', scalar_value=val, global_step=global_step)

    mean_hd = torch.cat(list_hd).mean(dim=0)
    for i, (val) in enumerate(mean_hd):
        writer.add_scalar(f'HD/{structures_dict[int(mask_values[i])]}', scalar_value=val, global_step=global_step)

    mean_asd = torch.cat(list_asd).mean(dim=0)
    for i, (val) in enumerate(mean_asd):
        writer.add_scalar(f'ASD/{structures_dict[int(mask_values[i])]}', scalar_value=val, global_step=global_step)

    if len(seg_maps) > 0:
        seg_maps = seg_maps[0]
        figure = vxm.torch.utils.create_seg_figure(*seg_maps)
        writer.add_figure(tag='seg_maps',
                          figure=figure,
                          global_step=global_step)

    if calc_statistics:
        log_statistics(torch.cat(list_hd_std), structures_dict.values(), writer=writer,
                       global_step=global_step, title='Housedorf Distance std')
        log_statistics(torch.cat(list_dice_std), structures_dict.values(), writer=writer,
                       global_step=global_step, title='Dice Score std')
        log_statistics(torch.cat(list_hd_std), structures_dict.values(), writer=writer,
                       global_step=global_step, title='Average Surface Distance std')


def log_statistics(scores_std: torch.Tensor, labels, writer: SummaryWriter, title: str, global_step=0):
    data = scores_std.numpy()
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.boxplot(data, vert=False, showfliers=False)
    ax.set_yticklabels(labels)
    fig.set_size_inches(12, 8)
    writer.add_figure(f'statistics/{title}', figure=fig, global_step=global_step)
    print(f'created statistic figure for {title}')


def log_input_params(args: argparse.Namespace, writer: SummaryWriter):
    cell_data = [[key, f'{value}'] for key, value in zip(vars(args).keys(), vars(args).values())]
    fig, ax = plt.subplots(1, 1)
    ax.table(cellText=cell_data,
             loc='center')
    fig.set_size_inches(6, 8)
    ax.set_axis_off()
    writer.add_figure('Params', fig)
    param_text = "\n\n".join(["{key:<20}:{value:>40}".format(key=key, value=f'{value}')
                              for key, value in zip(vars(args).keys(), vars(args).values())])
    writer.add_text('params', param_text)


if __name__ == "__main__":
    main()

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

import numpy as np
import torch
# import napari
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from matplotlib import gridspec
from monai.metrics import compute_meandice
from monai.metrics import compute_hausdorff_distance
from monai.metrics import compute_average_surface_distance

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm


def main():
    parser = parse_args()
    args = parser.parse_args()

    bidir = args.bidir

    # tensorboard
    if args.log_dir is None:
        args.log_dir = args.model_dir
    writer = SummaryWriter(log_dir=f'{args.log_dir}/{datetime.now().strftime("%d.%m.%Y_%H.%M")}')

    generator = create_data_generator(args)

    test_generator = create_data_generator(args, is_train=False)

    # extract shape from sampled input
    inshape = args.inshape
    # prepare model folder
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    # device handling
    gpus = args.gpu.split(',')
    nb_gpus = len(gpus)
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert args.batch_size >= nb_gpus, 'Batch size (%d) should be no less than the number of gpus (%d)' % (
        args.batch_size, nb_gpus)

    model = create_model(args, bidir, device, inshape, nb_gpus)

    losses, optimizer, weights, loss_names = create_optimizers(args, bidir, model)
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
    parser.add_argument('--cudnn-nondet', action='store_true',
                        help='disable cudnn determinism - might slow down training')
    # network architecture parameters
    parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters (default: 16 32 32 32)')
    parser.add_argument('--dec', type=int, nargs='+',
                        help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
    parser.add_argument('--int-steps', type=int, default=7, help='number of integration steps (default: 7)')
    parser.add_argument('--int-downsize', type=int, default=2,
                        help='flow downsample factor for integration (default: 2)')
    parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')
    # loss hyperparameters
    parser.add_argument('--image-loss', default='mse',
                        help='image reconstruction loss - can be mse or ncc (default: mse)')
    parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                        help='weight of deformation loss (default: 0.01)')
    parser.add_argument('--inshape', type=int, nargs='+',
                        help='after cropping shape of input. '
                             'default is equal to image size. specify if the input can\'t path through UNet')
    parser.add_argument('--log-dir', type=str, default=None, help='folder for tensorboard logs')
    parser.add_argument('--display_freq', type=int, default=20, help='frequency of plotting results in tensorboard')
    parser.add_argument('--loader_name', type=str, default='default', help='volume generator function to use')
    parser.add_argument('--patient_list_src', type=str, default='/tmp/',
                        help='directory to store patient list for training and testing')
    parser.add_argument('--load_segmentation', action='store_true', default=False,
                        help='use segmentation data for training the network (torch functionality seems to be missing)')
    parser.add_argument('--use-probs', action='store_true', help='enable probabilities')
    parser.add_argument('--flow-logsigma-bias', type=float, default=-10,
                        help='negative value for initialization of the logsigma layer bias value')
    parser.add_argument('--kl-lambda', type=float, default=10,
                        help='prior lambda regularization for KL loss (default: 10)')
    return parser


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
                                                return_segs=return_segs,
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
    if nb_gpus > 1:
        # use multiple GPUs via DataParallel
        model = torch.nn.DataParallel(model)
        model.save = model.module.save
    # prepare the model for training and send to device
    model.to(device)
    model.train()
    return model


def create_optimizers(args, bidir, model):
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # prepare image loss
    loss_names = []
    available_loss_images = ['ncc', 'mse', 'ssim', 'mind']
    if args.image_loss == 'ncc':
        image_loss_func = vxm.losses.NCC().loss
    elif args.image_loss == 'mse':
        image_loss_func = vxm.losses.MSE().loss
    elif args.image_loss == 'ssim':
        image_loss_func = vxm.losses.SSIM().loss
    elif args.image_loss == 'mind':
        image_loss_func = vxm.losses.MIND().loss
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
        loss_names += 'KL'
    else:
        losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
        loss_names += 'Regularization'
    weights += [args.weight]
    return losses, optimizer, weights, loss_names


def train(args, device, generator, losses, model, model_dir, optimizer, weights, writer, loss_names, test_generator):
    ssim = vxm.losses.SSIM()
    transformer = vxm.layers.SpatialTransformer(size=args.inshape, mode='nearest').to(device)

    for epoch in range(args.initial_epoch, args.epochs):
        # save model checkpoint
        model.save(os.path.join(model_dir, '%04d.pt' % epoch))

        for step in range(args.steps_per_epoch):
            step_start_time = time.time()

            loss, loss_list, _, _, _ = apply_model(model, generator, device, losses, weights)
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
                model.eval()
                tensorboard_log(model, test_generator, loss_names, device, loss_list, writer, ssim=ssim,
                                global_step=global_step)
                evaluate_with_segmentation(model, test_generator, device=device, writer=writer,
                                           global_step=global_step, transformer=transformer)
                model.train()
    # final model save
    model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))


def apply_model(model, generator, device, losses=None, weights=None, is_test=False, has_seg=False):
    # generate inputs (and true outputs) and convert them to tensors
    inputs, y_true = next(generator)
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


def tensorboard_log(model, test_generator, loss_names, device, loss_list,
                    writer: SummaryWriter, ssim: vxm.losses.SSIM, global_step=0):
    with torch.no_grad():
        inputs, y_true, y_pred = apply_model(model=model, generator=test_generator, device=device,
                                             is_test=True, has_seg=True)
    ddf = y_pred[-1].detach()
    y_pred = y_pred[0]
    figure = vxm.torch.utils.create_figure(y_true[0].cpu(), inputs[0].cpu(), y_pred.cpu(),
                                           ddf.cpu())
    writer.add_figure(tag='volumes',
                      figure=figure,
                      global_step=global_step)
    for name, value in zip(loss_names, list(map(float, loss_list))):
        writer.add_scalar(f'loss/{name}', value, global_step=global_step)


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


def evaluate_with_segmentation(model, test_generator, device, writer: SummaryWriter,
                                transformer, num_of_vol=10, global_step=0, calc_statistics=True):
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
    for step in range(num_of_vol):
        print(step)
        # generate scores for logging on tensorboard
        dice_score, hd_score, asd_score, dice_std, hd_std, asd_std = calc_scores(device, model, test_generator, transformer, mask_values,
                                                      calc_statistics=calc_statistics)
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

    if calc_statistics:
        log_statistics(torch.cat(list_hd_std), structures_dict.values(), writer=writer,
                       global_step=global_step, title='Housedorf Distance std')
        log_statistics(torch.cat(list_dice_std), structures_dict.values(), writer=writer,
                       global_step=global_step, title='Dice Score std')
        log_statistics(torch.cat(list_hd_std), structures_dict.values(), writer=writer,
                       global_step=global_step, title='Average Surface Distance std')


def calc_scores(device, model, test_generator, transformer, mask_values, calc_statistics=False):
    reps = 1
    if calc_statistics:
        reps = 10
    dice_scores = []
    hd_scores = []
    asd_scores = []

    with torch.no_grad():
        for n in range(reps):
            inputs, y_true, y_pred = apply_model(model=model, generator=test_generator, device=device,
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


def log_statistics(scores_std: torch.Tensor, labels, writer: SummaryWriter, title: str, global_step=0):
    data = scores_std.numpy()
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.boxplot(data, vert=False, showfliers=False)
    writer.add_figure(f'statistics', figure=fig, global_step=global_step)



if __name__ == "__main__":
    main()

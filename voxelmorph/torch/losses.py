import torch
import torch.nn.functional as F
import numpy as np
import math

from torch import nn
from torch.autograd import Variable
from math import exp


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        I = y_true
        J = y_pred

        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """
    def __init__(self, image_sigma=1.0):
        self.image_sigma = image_sigma

    def loss(self, y_true, y_pred):
        return 1.0 / (self.image_sigma**2) * torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


#################################################
## SSIM loss for 2D and 3D
## base repo: https://github.com/Po-Hsun-Su/pytorch-ssim
#################################################


class SSIM:
    def __init__(self, window_size=11, size_average=True):
        self.window_size = window_size
        self.size_average = size_average
        self.window = None
        pass

    def loss(self, y_fixed: torch.Tensor, y_pred: torch.Tensor):
        ndims = len(y_fixed.shape) - 2
        channel = y_fixed.shape[1]

        if self.window is None:
            window = create_window(self.window_size, channel, ndims=ndims)

            if y_fixed.is_cuda:
                window = window.cuda(y_fixed.device)
            window = window.type_as(y_fixed)
            self.window = window

        return _ssim(y_fixed, y_pred, self.window, self.window_size, channel, self.size_average)


def gaussian(window_size, sigma):
    gauss = torch.tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel, ndims=3):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    assert ndims in [2, 3], 'can create window only for 2D and 3D images'
    if ndims == 2:
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    else:
        _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                      window_size).float().unsqueeze(0).unsqueeze(0)
        window = _3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
    return window


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window, window_size, channel, size_average=True):
    ndims = len(list(img1.size())) - 2
    conv_fn = getattr(F, 'conv%dd' % ndims)

    mu1 = conv_fn(img1, window, padding=window_size // 2, groups=channel)
    mu2 = conv_fn(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv_fn(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = conv_fn(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = conv_fn(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map: torch.Tensor = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.view((ssim_map.shape[0], -1)).mean(1)


##################################################
## Added MIND loss from this pull request:
## https://github.com/voxelmorph/voxelmorph/pull/145#issue-372397584
##################################################

def pdist_squared(x):
    xx = (x ** 2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist


def MINDSSC(img, radius=2, dilation=2):
    # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

    # kernel size
    kernel_size = radius * 2 + 1

    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.tensor([[0, 1, 1],
                                      [1, 1, 0],
                                      [1, 0, 1],
                                      [1, 1, 2],
                                      [2, 1, 1],
                                      [1, 2, 1]]).long()

    # squared distances
    dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

    # define comparison mask
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
    mask = ((x > y).view(-1) & (dist == 2).view(-1))

    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
    mshift1 = torch.zeros(12, 1, 3, 3, 3).cuda()
    mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3).cuda()
    mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    rpad1 = nn.ReplicationPad3d(dilation)
    rpad2 = nn.ReplicationPad3d(radius)

    # compute patch-ssd
    ssd = F.avg_pool3d(rpad2(
        (F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2),
                       kernel_size, stride=1)

    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, (mind_var.mean() * 0.001).item(), (mind_var.mean() * 1000).item())
    mind /= mind_var
    mind = torch.exp(-mind)

    # permute to have same ordering as C++ code
    mind = mind[:, torch.tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

    return mind


class MIND:

    def loss(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        return torch.mean((MINDSSC(y_true) - MINDSSC(y_pred)) ** 2)


class KL:
    """
    Kullback–Leibler divergence for probabilistic flows.
    """

    def __init__(self, prior_lambda):
        self.prior_lambda = prior_lambda
        self.D = None

    def _adj_filt(self, ndims):
        """
        compute an adjacency filter that, for each feature independently,
        has a '1' in the immediate neighbor, and 0 elsewhere.
        so for each filter, the filter has 2^ndims 1s.
        the filter is then setup such that feature i outputs only to feature i
        """

        # inner filter, that is 3x3x...
        filt_inner = np.zeros([3] * ndims)
        for j in range(ndims):
            o = [[1]] * ndims
            o[j] = [0, 2]
            filt_inner[np.ix_(*o)] = 1

        # full filter, that makes sure the inner filter is applied
        # ith feature to ith feature
        filter_ = np.zeros([ndims, ndims] + [3] * ndims)
        for i in range(ndims):
            filter_[i, i, ...] = filt_inner

        return filter_

    def _degree_matrix(self, vol_shape):
        # get shape stats
        ndims = len(vol_shape)
        sz = [ndims, *vol_shape]

        # prepare conv kernel
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # prepare tf filter
        z = torch.ones([1] + sz)
        filt_ = torch.tensor(self._adj_filt(ndims), dtype=torch.float32)
        strides = [1] * ndims

        return conv_fn(z, filt_, stride=strides, padding=1)

    def prec_loss(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        a more manual implementation of the precision matrix term
                mu * P * mu    where    P = D - A
        where D is the degree matrix and A is the adjacency matrix
                mu * P * mu = 0.5 * sum_i mu_i sum_j (mu_i - mu_j) = 0.5 * sum_i,j (mu_i - mu_j) ^ 2
        where j are neighbors of i

        Note: could probably do with a difference filter,
        but the edges would be complicated unless tensorflow allowed for edge copying
        """
        vol_shape = list(y_pred.shape)[2:]
        ndims = len(vol_shape)

        sm = torch.tensor([0.0], device=y_pred.device)
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = y_pred.permute([0, *range(2, ndims + 2), 1]).permute(r)  # first change to TF convention then perm
            df = y[1:, ...] - y[:-1, ...]
            sm += (df * df).mean()

        return 0.5 * sm / ndims

    def loss(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """
        KL loss
        y_pred is assumed to be D*2 channels: first D for mean, next D for logsigma
        D (number of dimensions) should be 1, 2 or 3

        y_true is only used to get the shape
        """

        # prepare inputs
        ndims = len(y_pred.shape) - 2
        mean = y_pred[0, ...]
        log_sigma = y_pred[1, ...]

        # compute the degree matrix (only needs to be done once)
        # we usually can't compute this until we know the ndims,
        # which is a function of the data
        if self.D is None:
            self.D = self._degree_matrix(y_true.shape[-3:]).to(y_pred.device)

        # sigma terms
        sigma_term = self.prior_lambda * self.D * torch.exp(log_sigma) - log_sigma
        sigma_term = sigma_term.mean()

        # precision terms
        # note needs 0.5 twice, one here (inside self.prec_loss), one below
        prec_term = self.prior_lambda * self.prec_loss(mean)

        # combine terms
        return 0.5 * ndims * (sigma_term + prec_term)  # ndims because we averaged over dimensions as well
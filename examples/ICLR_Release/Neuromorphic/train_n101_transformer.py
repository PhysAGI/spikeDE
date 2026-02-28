import datetime
import os
import time
import torch
import torch.utils.data
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import math
from torch.cuda import amp
import model as spikformer_models
import utils
from spikingjelly.clock_driven import functional
from spikingjelly.datasets import cifar10_dvs
from timm.models import create_model
from timm.data import Mixup
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.loss import SoftTargetCrossEntropy
import autoaugment
from torch.nn import functional as F
from spikingjelly.datasets.n_caltech101 import NCaltech101 
from torch.utils.data import random_split         
_seed_ = 2021
import random
random.seed(2021)
root_path = os.path.abspath(__file__)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(_seed_)
# Keep this reference so local timm model registration side effects are preserved.
_ = spikformer_models

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--model', default='spikformer', help='model')
    parser.add_argument('--dataset', default='DVS128', help='dataset')
    parser.add_argument('--num-classes', type=int, default=101, metavar='N',
                        help='number of label classes (default: 1000)')
    parser.add_argument('--data-path', default='./data/DVS128/101', help='dataset')

    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--print-freq', default=256, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./logs', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        # default=True,
        help="Only test the model",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument('--amp', default=True, action='store_true',
                        help='Use AMP training')


    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--tb', default=True,  action='store_true',
                        help='Use TensorBoard to record logs')
    parser.add_argument('--T', default=16, type=int, help='simulation steps')
    # parser.add_argument('--adam', default=True, action='store_true',
    #                     help='Use Adam')

    # Optimizer Parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar="OPTIMIZER", help='Optimizer (default: "adamw")')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, metavar='BETA', help='Optimizer Betas')
    parser.add_argument('--weight-decay', default=0.06, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='Momentum for SGD. Adam will not use momentum')

    parser.add_argument('--connect_f', default='ADD', type=str, help='element-wise connect function')
    parser.add_argument('--T_train', default=None, type=int)

    #Learning rate scheduler
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit')
    parser.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                        help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=float, default=20, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation & regularization parameters
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--mixup', type=float, default=0.5,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix', type=float, default=0.,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=0.5,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                        help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    parser.add_argument('--integrator', type=str, default='fdeint',
                        choices=['odeint_adjoint', 'odeint', 'fdeint_adjoint', 'fdeint'],
                        help='differential equation integrator type (default: odeint_adjoint)')

    parser.add_argument('--alpha', type=float, nargs='+', default=None,
                        help='Fractional order(s). Single value for single-term FDE, '
                             'multiple values for multi-term FDE. '
                             'Examples: --alpha 0.5 (single) or --alpha 0.3 0.7 0.9 (multi)')
    parser.add_argument('--multi_coefficient', type=float, nargs='+', default=None,
                        help='Coefficients for multi-term FDE. Required when --alpha has multiple values. '
                             'Example: --multi_coefficient 1.0 0.5 0.2')
    parser.add_argument('--learn_coefficient', action='store_true',
                        help='Make multi-term coefficients learnable')

    parser.add_argument('--method', type=str, default='pred', choices=[
        'euler',  # odeint only supports euler method
        'gl-f', 'gl-o', 'trap-f', 'trap-o',
        'l1-f', 'l1-o', 'pred-f', 'pred-o',  # fdeint_adjoint methods
        'pred', 'gl', 'trap', 'l1', 'glmulti'  # fdeint methods
    ], help='method for euler solver')
    parser.add_argument('--step_size', type=float, default=0.1,
                        help='Integration step size (default: 1.0)')
    parser.add_argument('--time_interval', type=float, default=0.1,
                        help='Time interval between steps in ms (default: 1.0) for each input event')
    parser.add_argument('--memory', type=int, default=-1,
                        help='if cost much gpu memory, set save_memory to small positive int, '
                             'default: -1, which means full memory')
    parser.add_argument('--glmulti_chunk', type=int, default=96,
                        help='Chunk size for CuPy glmulti history convolution (default: 96)')
    parser.add_argument('--glmulti_detach_conv', type=int, default=1, choices=[0, 1],
                        help='Detach conv history for CuPy glmulti to reduce memory (default: 1)')
    parser.add_argument('--cupy', action='store_true',
                        help='use CuPy backend for fractional ODE solver (requires CUDA + CuPy)')
    
    args = parser.parse_args()
    return args


def process_alpha_args(args):
    if getattr(args, "alpha", None) is None:
        if getattr(args, "multi_coefficient", None) is not None:
            print("Warning: --multi_coefficient is ignored for single-term FDE (--alpha).")
        return args.alpha, None

    alpha = args.alpha
    if len(alpha) == 1:
        if args.multi_coefficient is not None:
            print("Warning: --multi_coefficient is ignored for single-term FDE (single alpha value)")
        return alpha[0], None

    multi_coefficient = args.multi_coefficient
    if multi_coefficient is None:
        raise ValueError(
            f"--multi_coefficient is required when --alpha has multiple values. "
            f"Got --alpha {alpha}"
        )
    if len(alpha) != len(multi_coefficient):
        raise ValueError(
            f"--alpha (len={len(alpha)}) and --multi_coefficient "
            f"(len={len(multi_coefficient)}) must have the same length"
        )
    print("[Multi-term FDE] Enabled:")
    print(f"  - Fractional orders (alpha): {alpha}")
    print(f"  - Coefficients: {multi_coefficient}")
    print(f"  - Learnable coefficients: {args.learn_coefficient}")
    return alpha, multi_coefficient

def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):
    '''
    :param train_ratio: split the ratio of the origin dataset as the train set
    :type train_ratio: float
    :param origin_dataset: the origin dataset
    :type origin_dataset: torch.utils.data.Dataset
    :param num_classes: total classes number, e.g., ``10`` for the MNIST dataset
    :type num_classes: int
    :param random_split: If ``False``, the front ratio of samples in each classes will
            be included in train set, while the reset will be included in test set.
            If ``True``, this function will split samples in each classes randomly. The randomness is controlled by
            ``numpy.randon.seed``
    :type random_split: int
    :return: a tuple ``(train_set, test_set)``
    :rtype: tuple
    '''
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(origin_dataset):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    train_idx = []
    test_idx = []
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, scaler=None, T_train=None, aug=None, trival_aug=None, mixup_fn=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)

    data_time = torch.linspace(0, args.time_interval * (args.T),
                               args.T+1, device=args.device).float()
    options = {
        'step_size': args.step_size,
        'memory': args.memory,
        'glmulti_chunk': args.glmulti_chunk,
        'glmulti_detach_conv': bool(args.glmulti_detach_conv),
    }

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        image = image.float()  # [N, T, C, H, W]
        N,T,C,H,W = image.shape

        image = image.permute(1, 0, 2, 3, 4).contiguous()

        image = F.interpolate(
                image.reshape(-1, C, H, W),  # 展平到 [(T * N), C, H, W]
                size=(128, 128),  # 目标大小
                mode='bilinear',  # 使用双线性插值
                align_corners=False  # 避免插值边界问题
            )
        image = image.reshape(T, N, C, 128, 128)
        # print(image.shape)
        if aug != None:
            image = torch.stack([(aug(image[i])) for i in range(T)])
        if trival_aug != None:
            image = torch.stack([(trival_aug(image[i])) for i in range(T)])

        if mixup_fn is not None:
            image, target = mixup_fn(image, target)
            target_for_compu_acc = target.argmax(dim=-1)

        if T_train:
            sec_list = np.random.choice(image.shape[1], T_train, replace=False)
            sec_list.sort()
            image = image[:, sec_list]

        if scaler is not None:
            with amp.autocast():
                #print(image.shape)
                output = model(image, data_time, output_time=data_time, method=args.method,
                                 options=options)

                loss = criterion(output, target)
        else:
            output = model(image, data_time, output_time=data_time, method=args.method,
                             options=options)
            loss = criterion(output, target)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        functional.reset_net(model)
        if mixup_fn is not None:
            acc1, acc5 = utils.accuracy(output, target_for_compu_acc, topk=(1, 5))
        else:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        loss_s = loss.item()
        if math.isnan(loss_s):
            raise ValueError('loss is Nan')
        acc1_s = acc1.item()
        acc5_s = acc5.item()

        metric_logger.update(loss=loss_s, lr=optimizer.param_groups[0]["lr"])

        metric_logger.meters['acc1'].update(acc1_s, n=batch_size)
        metric_logger.meters['acc5'].update(acc5_s, n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg


def evaluate(model, criterion, data_loader, device, print_freq=100, header='Test:'):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    data_time = torch.linspace(0, args.time_interval * (args.T),
                               args.T+1, device=args.device).float()
    options = {
        'step_size': args.step_size,
        'memory': args.memory,
        'glmulti_chunk': args.glmulti_chunk,
        'glmulti_detach_conv': bool(args.glmulti_detach_conv),
    }
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image, target = image.to(device), target.to(device)
            # print('(---test---)' * 4, target.shape)

            image = image.float()  # [N, T, C, H, W]
            N, T, C, H, W = image.shape
            image = image.permute(1, 0, 2, 3, 4).contiguous()

            image = F.interpolate(
                image.reshape(-1, C, H, W),  # 展平到 [(T * N), C, H, W]
                size=(128, 128),  # 目标大小
                mode='bilinear',  # 使用双线性插值
                align_corners=False  # 避免插值边界问题
            )
            image = image.reshape(T, N, C, 128, 128)

            output = model(image, data_time, output_time=data_time, method=args.method,
                                 options=options)
            loss = criterion(output, target)
            functional.reset_net(model)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    loss, acc1, acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    print(f' * Acc@1 = {acc1}, Acc@5 = {acc5}, loss = {loss}')
    return loss, acc1, acc5


def load_data(dataset_dir, distributed, T):
    # Data loading code
    print("Loading data")

    st = time.time()

    origin_set = cifar10_dvs.CIFAR10DVS(root=dataset_dir, data_type='frame', frames_number=T, split_by='number')
    dataset_train, dataset_test = split_to_train_test_set(0.9, origin_set, 10)
    print("Took", time.time() - st)

    print("Creating data loaders")
    if distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
        sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_train, dataset_test, sampler_train, sampler_val


def main(args):

    max_test_acc1 = 0.
    test_acc5_at_max_test_acc1 = 0.

    train_tb_writer = None
    te_tb_writer = None

    utils.init_distributed_mode(args)
    print(args)

    if isinstance(alpha, (list, tuple)):
        alpha_tag = "-".join(f"{a:.3f}" for a in alpha)
    else:
        alpha_tag = f"{alpha:.3f}"
    output_dir = os.path.join(
        args.output_dir,
        f'{args.model}_b{args.batch_size}_T{args.T}_method{args.method}_alpha{alpha_tag}_step{args.step_size}'
    )

    if args.T_train:
        output_dir += f'_Ttrain{args.T_train}'

    if args.weight_decay:
        output_dir += f'_wd{args.weight_decay}'

    if args.opt == 'adamw':
        output_dir += '_adamw'
    else:
        output_dir += '_sgd'

    if args.connect_f:
        output_dir += f'_cnf_{args.connect_f}'

    if args.cupy:
        output_dir += '_cupy'

    if not os.path.exists(output_dir):
        utils.mkdir(output_dir)

    output_dir = os.path.join(output_dir, f'lr{args.lr}')
    if not os.path.exists(output_dir):
        utils.mkdir(output_dir)

    device = torch.device(args.device)
    
    # 创建DVS128Gesture数据集，替换原来的build_dataset
    data_set = NCaltech101(root=args.data_path, data_type='frame', frames_number=args.T, split_by='number')
    #test_set = NCaltech101(root=args.data_dir, data_type='frame', frames_number=args.T, split_by='number')

    train_ratio = 0.8

    # 计算具体数量
    train_size = int(len(data_set) * train_ratio)
    test_size = len(data_set) - train_size

    # 执行随机划分（保持数据分布）
    dataset_train, dataset_test = random_split(
        data_set,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 设置随机种子保证可复现
    )


    sampler_train = None
    if args.distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
        sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=(sampler_train is None),
        sampler=sampler_train,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        sampler=sampler_val,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True)

    model = create_model(
        args.model,
        pretrained=False,
        drop_rate=0.,
        drop_path_rate=0.1,
        drop_block_rate=None,
        integrator=args.integrator,
        alpha=alpha,
        multi_coefficient=multi_coefficient,
        learn_coefficient=args.learn_coefficient,
        method=getattr(args, 'method'),
        step_size=getattr(args, 'step_size'),
        time_interval=getattr(args, 'time_interval'),
        memory=getattr(args, 'memory'),
        device=args.device,
    )
    if args.cupy:
        if device.type != 'cuda':
            raise RuntimeError("CuPy backend requires CUDA device.")
        try:
            import cupy  # type: ignore
        except ImportError as exc:
            raise ImportError("CuPy backend requested but CuPy is not installed.") from exc
        model.fde_backend = 'cupy'
        print(f"CuPy backend enabled (version {cupy.__version__}).")
    else:
        model.fde_backend = 'torch'
    print("Creating model")
    print(model)  # 添加这行来打印模型结构
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # criterion_train = LabelSmoothingCrossEntropy()
    criterion_train = SoftTargetCrossEntropy().cuda()
    criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer(args, model)
    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    print(num_epochs)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        # 修改后
        try:
            # 尝试使用 weights_only=False 加载
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        except:
            # 如果失败，回退到旧方法
            checkpoint = torch.load(args.resume, map_location='cpu')

        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        max_test_acc1 = checkpoint['max_test_acc1']
        test_acc5_at_max_test_acc1 = checkpoint['test_acc5_at_max_test_acc1']

    if args.test_only:

        evaluate(model, criterion, data_loader_test, device=device, header='Test:')
        return

    if args.tb and utils.is_main_process():
        purge_step_train = args.start_epoch
        purge_step_te = args.start_epoch
        tb_root = output_dir + '_logs'
        utils.mkdir(tb_root)
        utils.mkdir(os.path.join(tb_root, 'train'))
        utils.mkdir(os.path.join(tb_root, 'te'))
        train_tb_writer = SummaryWriter(os.path.join(tb_root, 'train'), purge_step=purge_step_train)
        te_tb_writer = SummaryWriter(os.path.join(tb_root, 'te'), purge_step=purge_step_te)
        with open(os.path.join(tb_root, 'args.txt'), 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))

        print(f'purge_step_train={purge_step_train}, purge_step_te={purge_step_te}')


    train_snn_aug = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5)
                    ])
    train_trivalaug = autoaugment.SNNAugmentWide()
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        mixup_fn = Mixup(**mixup_args)
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, num_epochs):
        save_max = False
        if args.distributed:
            sampler_train.set_epoch(epoch)
        if epoch >= 75:
            mixup_fn.mixup_enabled = False
        train_start_time = time.time()
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, criterion_train, optimizer, data_loader, device, epoch,
            args.print_freq, scaler, args.T_train,
            train_snn_aug, train_trivalaug, mixup_fn)
        train_time = time.time() - train_start_time
        train_samples = len(data_loader) * args.batch_size
        train_speed = train_samples / train_time if train_time > 0 else 0.0
        print(f"Epoch {epoch}: train_acc1={train_acc1:.4f}, train_acc5={train_acc5:.4f}")
        if utils.is_main_process():
            train_tb_writer.add_scalar('train_loss', train_loss, epoch)
            train_tb_writer.add_scalar('train_acc1', train_acc1, epoch)
            train_tb_writer.add_scalar('train_acc5', train_acc5, epoch)
        lr_scheduler.step(epoch + 1)

        test_start_time = time.time()
        test_loss, test_acc1, test_acc5 = evaluate(model, criterion, data_loader_test, device=device, header='Test:')
        test_time = time.time() - test_start_time
        test_samples = len(data_loader_test) * args.batch_size
        test_speed = test_samples / test_time if test_time > 0 else 0.0
        if te_tb_writer is not None:
            if utils.is_main_process():

                te_tb_writer.add_scalar('test_loss', test_loss, epoch)
                te_tb_writer.add_scalar('test_acc1', test_acc1, epoch)
                te_tb_writer.add_scalar('test_acc5', test_acc5, epoch)

        if utils.is_main_process():
            if hasattr(model_without_ddp, "multi_coefficient") and model_without_ddp.multi_coefficient is not None:
                coeff_tensor = model_without_ddp.multi_coefficient.detach().flatten().cpu()
                coeff_list = ", ".join(f"{c:.4f}" for c in coeff_tensor.tolist())
                multi_coeff_str = f"multi_coefficient: {coeff_list}"
            elif args.multi_coefficient is not None:
                coeff_list = ", ".join(f"{c:.4f}" for c in args.multi_coefficient)
                multi_coeff_str = f"multi_coefficient(init): {coeff_list}"
            else:
                multi_coeff_str = "multi_coefficient: None"
            print(multi_coeff_str)


        if max_test_acc1 < test_acc1:
            max_test_acc1 = test_acc1
            test_acc5_at_max_test_acc1 = test_acc5
            save_max = True


        if output_dir:

            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'max_test_acc1': max_test_acc1,
                'test_acc5_at_max_test_acc1': test_acc5_at_max_test_acc1,
            }

            if save_max:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(output_dir, 'checkpoint_max_test_acc1.pth'))
                
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        print(f'avg train speed = {train_speed:.4f} images/s, avg test speed = {test_speed:.4f} images/s')
        print('Training time {}'.format(total_time_str), 'max_test_acc1', max_test_acc1, 'test_acc5_at_max_test_acc1', test_acc5_at_max_test_acc1)
        print(output_dir)
    if output_dir:
        utils.save_on_master(
            checkpoint,
            os.path.join(output_dir, f'checkpoint_{epoch}.pth'))

    return max_test_acc1

if __name__ == "__main__":
    args = parse_args()
    alpha, multi_coefficient = process_alpha_args(args)
    if isinstance(alpha, (list, tuple)) and args.method != "glmulti":
        raise ValueError(
            f"Multi-term alpha requires method 'glmulti', got '{args.method}'. "
            "Please set --method glmulti."
        )
    args.alpha = alpha
    main(args)

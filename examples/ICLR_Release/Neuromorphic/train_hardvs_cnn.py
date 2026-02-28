import torch
import sys
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import os
import argparse
import datetime
import torch.nn as nn
from spikeDE import SNNWrapper
from spikeDE.neuron import LIFNeuron

from hardvs import HARDVS


def calculate_topk_correct(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """
    Return the number of correct predictions for each k in topk.
    """
    with torch.no_grad():
        maxk = min(max(topk), output.size(1))
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            k = min(k, maxk)
            if k == 0:
                res.append(0.0)
            else:
                res.append(correct[:k].reshape(-1).float().sum().item())
        return res


def process_alpha_args(args):
    """
    Process alpha arguments and validate multi-term configuration.
    Returns:
        tuple: (alpha, multi_coefficient) where alpha is float or list
    """
    if args.alpha is None:
        if args.multi_coefficient is not None:
            print("Warning: --multi_coefficient is ignored for single-term FDE (--beta).")
        return args.beta, None

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




#     # 假设 LIFNeuron 和 VotingLayer 已经定义
class CustomCNN(nn.Module):
    """
    自定义卷积神经网络，结合 LIFNeuron 和 VotingLayer
    """

    def __init__(self, args):
        super(CustomCNN, self).__init__()

        # 第一卷积模块
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=args.channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.lif1 = LIFNeuron(args.tau, args.threshold, args.surrogate_grad_scale) #nn.BatchNorm2d(128,track_running_stats=False, momentum=0.0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二卷积模块
        self.conv2 = nn.Conv2d(in_channels=args.channels, out_channels=args.channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.lif2 = LIFNeuron(args.tau, args.threshold, args.surrogate_grad_scale)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第三卷积模块
        self.conv3 = nn.Conv2d(in_channels=args.channels, out_channels=args.channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.lif3 = LIFNeuron(args.tau, args.threshold, args.surrogate_grad_scale)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第四卷积模块
        self.conv4 = nn.Conv2d(in_channels=args.channels, out_channels=args.channels, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.lif4 = LIFNeuron(args.tau, args.threshold, args.surrogate_grad_scale)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第五卷积模块
        self.conv5 = nn.Conv2d(in_channels=args.channels, out_channels=args.channels, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.lif5 = LIFNeuron(args.tau, args.threshold, args.surrogate_grad_scale)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(args.channels * 4 * 4, 512)  # 假设输入图像经过卷积后最终尺寸为 4x4
        self.lif6 = LIFNeuron(args.tau, args.threshold, args.surrogate_grad_scale)

        self.dropout2 = nn.Dropout(0.5)

        self.output_layer = nn.Linear(512, 300)


    def forward(self, x):
        # 第一卷积模块
        #print(x.shape)
        x = self.conv1(x)
        #print(x.shape)
        x = self.bn1(x)
        x = self.lif1(x)
        x = self.pool1(x)

        # 第二卷积模块
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lif2(x)
        x = self.pool2(x)

        # 第三卷积模块
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lif3(x)
        x = self.pool3(x)

        # 第四卷积模块
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.lif4(x)
        x = self.pool4(x)

        # 第五卷积模块
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.lif5(x)
        x = self.pool5(x)

        # 全连接层
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.lif6(x)

        x = self.dropout2(x)
        # 输出层
        x = self.output_layer(x)
        return x

def main():
    # Example: python train_hardvs_cnn.py -T 8 -device cuda:0 -b 32 -epochs 64 -data-dir /path/to/hardvs -amp -cupy -opt adam -lr 0.001 -j 8

    parser = argparse.ArgumentParser(description='Classify HarDVS  ')
    parser.add_argument('-T', default=8, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=32, type=int, help='batch size')
    parser.add_argument('-epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', default='/data/gcj/data1/DVS128/hardvs', type=str, help='root dir of hardvs dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use cupy backend')
    parser.add_argument('-opt', default='adam', type=str, help='use which optimizer. SDG or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_min', default=1e-6, type=float, help='minimum learning rate for cosine annealing')
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')
    parser.add_argument('--model', type=str, default='cnn', choices=['mlp', 'cnn'],
                        help='Model architecture: mlp or cnn (default: cnn)')
    parser.add_argument('--neuron_type', type=str, default='LIF',
                        help='Neuron type for SNN (default: LIF)')
    parser.add_argument('--tau', type=float, default=2.0,
                        help='Membrane time constant for LIF neurons (default: 2.0)')
    parser.add_argument('--split_by', type=str, default='number', choices=['number', 'time'],
                        help='dataset frame split method from events: number or time (default: number)')
    ## if split_by == time, the duration of each frame is set by time_interval
    ## if split_by == number, the duration of each frame is decided by T, i.e. time_of_each_sample/T
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='Firing threshold (default: 1.0)')
    parser.add_argument('--surrogate_grad_scale', type=float, default=5.0,
                        help='Scale parameter for surrogate gradient (default: 5.0)')
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
    parser.add_argument('--beta', type=float, default=0.9,
                        help='[Deprecated] single-term fractional order (use --alpha instead)')

    parser.add_argument('--method', type=str, default='pred', choices=[
        'euler',  # odeint only supports euler method
        'gl-f', 'gl-o', 'trap-f', 'trap-o',
        'l1-f', 'l1-o', 'pred-f', 'pred-o',  # fdeint_adjoint methods
        'pred', 'gl', 'trap', 'l1', 'glmulti'  # fdeint methods
    ], help='Method for solver (default: gl)')
    parser.add_argument('--step_size', type=float, default=0.1,
                        help='Integration step size (default: 1.0)')
    parser.add_argument('--time_interval', type=float, default=0.1,
                        help='Time interval between steps in ms (default: 1.0) for each input event')
    parser.add_argument('--memory', type=int, default=-1,
                        help='if cost much gpu memory, set save_memory to small positive int, '
                             'default: -1, which means full memory')

    args = parser.parse_args()
    print(args)
    alpha, multi_coefficient = process_alpha_args(args)
    if isinstance(alpha, (list, tuple)) and args.method != "glmulti":
        raise ValueError(
            f"Multi-term alpha requires method 'glmulti', got '{args.method}'. "
            "Please set --method glmulti."
        )
    args.alpha = alpha

    def print_total_parameters(self):
        total_params = 0
        for param in self.parameters():
            total_params += param.numel()  # Count the number of elements in each parameter tensor
        print(f"Total number of parameters: {total_params}")



    network = CustomCNN(args)

    net = SNNWrapper(
        network,
        integrator=args.integrator,
        alpha=alpha,
        multi_coefficient=multi_coefficient,
        learn_coefficient=args.learn_coefficient,
    ).to(args.device)
    net.fde_backend = 'cupy' if args.cupy else 'torch'

    print(net)
    print_total_parameters(net)

    net.to(args.device)
    net._set_neuron_shapes(input_shape=(1, 2, 128, 128))

    train_set = HARDVS(root=args.data_dir, train_test_val='train', data_type='frame', frames_number=args.T,
                  split_by='number')
    test_set = HARDVS(root=args.data_dir, train_test_val='test', data_type='frame', frames_number=args.T,
                    split_by='number')

    # 创建数据加载器
    trainloader = DataLoader(
        dataset=train_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )

    testloader = DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

    if isinstance(alpha, (list, tuple)):
        alpha_tag = "-".join(f"{a:.3f}" for a in alpha)
    else:
        alpha_tag = f"{alpha:.3f}"
    out_dir = os.path.join('DVS128GestureFDE', args.out_dir,
                           f'T{args.T}_s{args.step_size}_t{args.time_interval}_lr{args.lr}_c{args.channels}_alpha({alpha_tag})_method({args.method})')

    if args.amp:
        out_dir += '_amp'

    if args.cupy:
        out_dir += '_cupy'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    # data_time = torch.linspace(0, args.time_interval * (args.T - 1),
    #                            args.T, device=args.device).float()
    data_time = torch.linspace(0, args.time_interval * (args.T),
                               args.T+1, device=args.device).float()

    options = {
        'step_size': args.step_size,
        'memory': args.memory,
    }

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc1 = 0
        train_acc5 = 0
        train_samples = 0

        # Wrap your data loader with tqdm for a progress bar
       #progress_bar = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{args.epochs}")


        for frame, label in trainloader:
            optimizer.zero_grad()
            frame = frame.to(args.device)
            frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
            label = label.to(args.device)
            T, N, C, H, W = frame.shape
            frame = F.interpolate(
                frame.reshape(-1, C, H, W),  # 展平到 [(T * N), C, H, W]
                size=(128, 128),  # 目标大小
                mode='bilinear',  # 使用双线性插值
                align_corners=False  # 避免插值边界问题
            )
            frame = frame.reshape(T, N, C, 128, 128)
            # label_onehot = F.one_hot(label, 11).float()

            if scaler is not None:
                with amp.autocast():
                    out_fr = net(frame, data_time, output_time=data_time, method=args.method,
                                 options=options)  # Assuming correct function call
                    out_fr = out_fr.mean(0)  # Accessing specific output depending on your network design
                    # out_fr = out_fr[-1][-1]
                    loss = F.cross_entropy(out_fr, label)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = net(frame, data_time, output_time=data_time, method=args.method,
                             options=options)  # Assuming correct function call
                out_fr = out_fr.mean(0)
                loss = F.cross_entropy(out_fr, label)
                # loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()

            # print("-"*20)
            # print('logits_abs_mean=', out_fr.abs().mean().item(),
            #       'softmax_std=', torch.softmax(out_fr, -1).std(dim=1).mean().item())

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            acc1, acc5 = calculate_topk_correct(out_fr, label, topk=(1, 5))
            train_acc1 += acc1
            train_acc5 += acc5

            # 更新进度条描述信息，包括当前损失和准确率
            # progress_bar.set_description(
            #     f"Epoch {epoch + 1}/{args.epochs} - Loss: {train_loss / train_samples:.4f}, "
            #     f"Acc@1: {train_acc1 / train_samples:.4f}, Acc@5: {train_acc5 / train_samples:.4f}")
            # print('net.ode_func.nfe:', net.ode_func.nfe)
            # net.ode_func.nfe = 0

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc1 /= train_samples
        train_acc5 /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc1, epoch)
        writer.add_scalar('train_acc5', train_acc5, epoch)
        lr_scheduler.step()

        #progress_bar.close()

        print('train_time', train_time - start_time)

        net.eval()
        test_loss = 0
        test_acc1 = 0
        test_acc5 = 0
        test_samples = 0
        test_time1 = time.time()

        print("Starting inference...")

        with torch.no_grad():
            for frame, label in testloader:
                optimizer.zero_grad()
                frame = frame.to(args.device)
                frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
                label = label.to(args.device)
                T, N, C, H, W = frame.shape
                frame = F.interpolate(
                    frame.reshape(-1, C, H, W),  # 展平到 [(T * N), C, H, W]
                    size=(128, 128),  # 目标大小
                    mode='bilinear',  # 使用双线性插值
                    align_corners=False  # 避免插值边界问题
                )
                frame = frame.reshape(T, N, C, 128, 128)

                # 前向传播
                out_fr = net(frame, data_time, output_time=data_time, method=args.method, options=options)
                out_fr = out_fr.mean(0)

                # 计算损失
                loss = F.cross_entropy(out_fr, label)

                # 累计统计指标
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                acc1, acc5 = calculate_topk_correct(out_fr, label, topk=(1, 5))
                test_acc1 += acc1
                test_acc5 += acc5

                # print("-" * 20)
                # print('logits_abs_mean=', out_fr.abs().mean().item(),
                #       'softmax_std=', torch.softmax(out_fr, -1).std(dim=1).mean().item())

        # 计算平均损失和准确率
        test_time2 = time.time()
        test_loss /= test_samples
        test_acc1 /= test_samples
        test_acc5 /= test_samples
        test_speed = test_samples / (test_time2 - test_time1)

        # 输出测试结果
        print("\nTest Results:")
        print(f"{'Loss':<10} {'Acc@1':<10} {'Acc@5':<10}")
        print(f"{test_loss:<10.4f} {test_acc1:<10.4f} {test_acc5:<10.4f}")
        net.ode_func.nfe = 0
        # 记录到 TensorBoard
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc1, epoch)
        writer.add_scalar('test_acc5', test_acc5, epoch)

        save_max = False
        if test_acc1 > max_test_acc:
            max_test_acc = test_acc1
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'gesture_sj_checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'gesture_fde_checkpoint_max.pth'))

        # 获取网络中的alpha参数值
        alpha_values = []
        for name, param in net.named_parameters():
            if 'alpha' in name.lower():
                alpha_values.append(f"{name}: {param.data.mean().item():.4f}")

        if alpha_values:
            alpha_str = ", ".join(alpha_values)
        else:
            if isinstance(args.alpha, (list, tuple)):
                alpha_list = ", ".join(f"{a:.4f}" for a in args.alpha)
                alpha_str = f"initial_alpha: {alpha_list}"
            else:
                alpha_str = f"initial_alpha: {args.alpha:.4f}"
        if hasattr(net, "multi_coefficient") and net.multi_coefficient is not None:
            coeff_tensor = net.multi_coefficient.detach().flatten().cpu()
            coeff_list = ", ".join(f"{c:.4f}" for c in coeff_tensor.tolist())
            multi_coeff_str = f"multi_coefficient: {coeff_list}"
        elif args.multi_coefficient is not None:
            coeff_list = ", ".join(f"{c:.4f}" for c in args.multi_coefficient)
            multi_coeff_str = f"multi_coefficient(init): {coeff_list}"
        else:
            multi_coeff_str = "multi_coefficient: None"
        
        print(args)
        print(out_dir)
        print(
            f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc@1 ={train_acc1: .4f}, train_acc@5 ={train_acc5: .4f}, '
            f'test_loss ={test_loss: .4f}, test_acc@1 ={test_acc1: .4f}, test_acc@5 ={test_acc5: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'Alpha parameters: {alpha_str}')
        print(multi_coeff_str)
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(
            f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')


if __name__ == '__main__':
    main()

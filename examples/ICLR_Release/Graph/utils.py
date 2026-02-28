import torch, os
import torch.nn as nn
import numpy as np
import torch.optim as optim
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Planetoid
from torch_geometric.utils import to_dense_adj, add_self_loops, degree
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
from spikingjelly.clock_driven import encoding
from typing import Optional, Tuple, Dict, Any, List


def normalized_adj(data: Data, add_self_loop: Optional[bool] = True) -> torch.Tensor:
    if add_self_loop is True:
        edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.shape[0])
        return to_dense_adj(edge_index, max_num_nodes=data.x.shape[0])[0].to_sparse()
    else:
        return to_dense_adj(data.edge_index, max_num_nodes=data.x.shape[0])[
            0
        ].to_sparse()


def normalized_degree(
    data: Data, add_self_loop: Optional[bool] = True, exponent: Optional[float] = -0.5
) -> torch.Tensor:
    if add_self_loop is True:
        edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.shape[0])
    else:
        edge_index = data.edge_index

    deg = degree(edge_index[0], num_nodes=data.x.shape[0])
    deg = torch.pow(deg, exponent)
    deg[torch.isinf(deg)] = 0
    return torch.diag(deg).to_sparse()


def positional_encoding(
    data: Data, dim: int = 10, method: str = "random_walk"
) -> torch.Tensor:
    assert method in [
        "random_walk",
        "laplacian",
    ], 'Only support "random_walk" or "laplacian".'
    A = normalized_adj(data, add_self_loop=False)
    if method == "random_walk":
        D = normalized_degree(data, exponent=-1.0, add_self_loop=False)
        P = torch.sparse.mm(A, D)
        PE = torch.zeros(data.x.shape[0], dim)
        PE[:, 0] = 1.0
        for i in range(1, dim):
            PE[:, i] = torch.diag(torch.sparse.mm(P, PE[:, i - 1].unsqueeze(1)))
    elif method == "laplacian":
        D = normalized_degree(data, exponent=-0.5, add_self_loop=False)
        laplacian = torch.eye(data.x.shape[0]) - torch.sparse.mm(
            (torch.sparse.mm(D, A)), D
        )
        _, eigen_vectors = torch.linalg.eigh(laplacian)
        PE = eigen_vectors[:, 1 : dim + 1]
        sign = torch.randint(0, 2, (dim,)) * 2 - 1
        PE = PE * sign.unsqueeze(0)
    return PE


def graph_convolution(
    data: Data, add_self_loop: Optional[bool] = True, exponent: Optional[float] = -0.5
) -> torch.Tensor:
    A = normalized_adj(data, add_self_loop)
    D = normalized_degree(data, add_self_loop, exponent)
    S = torch.sparse.mm(torch.sparse.mm(D, A), D)
    H = torch.sparse.mm(S, data.x)
    return H.to_dense()


def split_idx(
    data,
    method: Optional[str] = "public",
    num_train_per_class: Optional[int] = 20,
    num_val: Optional[int] = 500,
    num_test: Optional[int] = 1000,
    ratio: Optional[List[float]] = [0.7, 0.2, 0.1],
) -> 'Data':
    total_nodes = data.x.shape[0]
    data.train_mask = torch.zeros(total_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(total_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(total_nodes, dtype=torch.bool)

    if method == "public":
        train_size, val_size, test_size = 20, 500, 1000
    elif method == "random":
        train_size, val_size, test_size = num_train_per_class, num_val, num_test
    elif method == "ratio":
        assert len(ratio) == 3, "Ratio must be a list of three floats."
        assert abs(sum(ratio) - 1.0) < 1e-6, "Ratios must sum to 1.0."

        # 获取标签
        labels = data.y
        num_classes = int(labels.max().item()) + 1

        train_indices = []
        val_indices = []
        test_indices = []

        for cls in range(num_classes):
            # 找出当前类的所有节点索引
            cls_mask = (labels == cls).nonzero(as_tuple=False).view(-1)
            cls_count = cls_mask.numel()

            if cls_count == 0:
                continue  # 跳过空类

            # 打乱当前类的索引
            perm = torch.randperm(cls_count)
            shuffled_cls_idx = cls_mask[perm]

            # 按比例分配：先算各集所需数量（向下取整）
            n_train_cls = int(cls_count * ratio[0])
            n_val_cls = int(cls_count * ratio[1])
            n_test_cls = cls_count - n_train_cls - n_val_cls  # 剩余给 test，避免舍入误差

            # 确保非负
            n_train_cls = max(1, n_train_cls)  # 至少保留1个，防止某类全丢
            if n_val_cls < 0:
                n_val_cls = 0
            if n_test_cls < 0:
                # 如果比例太小导致负数，从 train 中匀一点出来
                deficit = -n_test_cls
                n_test_cls = 0
                n_train_cls = max(1, n_train_cls - deficit)
                n_val_cls = cls_count - n_train_cls

            # 切分
            train_part = shuffled_cls_idx[:n_train_cls]
            val_part = shuffled_cls_idx[n_train_cls:n_train_cls + n_val_cls]
            test_part = shuffled_cls_idx[n_train_cls + n_val_cls:]

            train_indices.append(train_part)
            val_indices.append(val_part)
            test_indices.append(test_part)

        # 合并所有类
        def concat_if_not_empty(indices_list):
            if not indices_list:
                return torch.empty(0, dtype=torch.long)
            return torch.cat(indices_list)

        train_idx = concat_if_not_empty(train_indices)
        val_idx = concat_if_not_empty(val_indices)
        test_idx = concat_if_not_empty(test_indices)

        # 去重（理论上不需要，但保险）
        train_idx = torch.unique(train_idx)
        val_idx = torch.unique(val_idx)
        test_idx = torch.unique(test_idx)

        # 设置 mask
        data.train_mask[train_idx] = True
        data.val_mask[val_idx] = True
        data.test_mask[test_idx] = True

        return data

    else:
        raise NotImplementedError("Method not supported.")

    # --- 以下为原 public/random 逻辑（保持不变）---
    for i in range(data.y.max().item() + 1):
        idx = (data.y == i).nonzero(as_tuple=False).reshape(-1)
        num_class = idx.numel()
        num_train = min(train_size, num_class)
        perm = torch.randperm(num_class)
        train_idx = idx[perm[:num_train]]
        data.train_mask[train_idx] = True

    remaining_idx = (~data.train_mask).nonzero(as_tuple=False).reshape(-1)
    num_remaining = remaining_idx.numel()
    perm = torch.randperm(num_remaining)
    val_idx = remaining_idx[perm[:val_size]]
    data.val_mask[val_idx] = True

    test_start = val_size
    test_end = min(val_size + test_size, num_remaining)
    test_idx = remaining_idx[perm[test_start:test_end]]
    data.test_mask[test_idx] = True

    return data


def load_data(
    root: str,
    name: str,
    split: Optional[str] = "public",
    batch_size: Optional[int] = 32,
    shuffle: Optional[bool] = False,
    num_train_per_class: Optional[int] = 20,
    num_val: Optional[int] = 500,
    num_test: Optional[int] = 1000,
    ratio: Optional[List[float]] = [0.7, 0.2, 0.1],
    add_self_loop: Optional[bool] = True,
    exponent: Optional[float] = -0.5,
    add_positional_encoding: Optional[bool] = False,
    positional_encoding_dim: Optional[int] = 10,
    positional_encoding_method: Optional[str] = "random_walk",
) -> Tuple[int, int, DataLoader, DataLoader, DataLoader]:
    if name.lower() in ["cora", "citeseer", "pubmed"]:
        data = (
            Planetoid(root, name, split, num_train_per_class, num_val, num_test)[0]
            if split != "ratio"
            else split_idx(
                Planetoid(root, name)[0],
                split,
                num_train_per_class,
                num_val,
                num_test,
                ratio,
            )
        )
    elif name.lower() in ["computers", "photo"]:
        data = split_idx(
            Amazon(root, name)[0], split, num_train_per_class, num_val, num_test, ratio
        )
    elif name.lower() in ["arxiv", "ogbn-arxiv"]:
        data = split_idx(
            PygNodePropPredDataset("ogbn-arxiv", root)[0],
            split,
            num_train_per_class,
            num_val,
            num_test,
            ratio,
        )
        data.y = data.y.squeeze(-1)

    features = graph_convolution(data, add_self_loop, exponent)
    features = (features - features.mean(dim=0, keepdim=True)) / (
        features.std(dim=0, keepdim=True) + 1e-8
    )

    if add_positional_encoding is True:
        pe = positional_encoding(
            data, positional_encoding_dim, positional_encoding_method
        )
        features = torch.concat((features, pe), dim=1)

    train_features = features[data.train_mask]
    train_labels = data.y[data.train_mask]
    val_features = features[data.val_mask]
    val_labels = data.y[data.val_mask]
    test_features = features[data.test_mask]
    test_labels = data.y[data.test_mask]

    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    test_dataset = TensorDataset(test_features, test_labels)

    return (
        features.shape[-1],
        data.y.max().item() + 1,
        DataLoader(train_dataset, batch_size, shuffle),
        DataLoader(val_dataset, batch_size),
        DataLoader(test_dataset, batch_size),
    )


def generate_spikes(
    data: torch.Tensor,
    encoder: encoding.StatelessEncoder,
    time_steps: Optional[int] = 100,
    time_interval: Optional[float] = 0.1,
    device: Optional[torch.device] = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    spikes = torch.full(
        (time_steps, data.shape[0], data.shape[1]),
        0,
        dtype=torch.float32,
        device=device,
    )
    for t in range(time_steps):
        spikes[t] = encoder(data)
    data_time = torch.linspace(0, time_interval * (time_steps - 1), time_steps).to(
        device
    )
    return spikes, data_time


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    encoder: encoding.StatelessEncoder,
    method: Optional[str] = "euler",
    time_steps: Optional[int] = 100,
    time_interval: Optional[float] = 0.1,
    device: Optional[torch.device] = torch.device("cpu"),
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for data, labels in train_loader:
        labels = labels.long().to(device)

        spikes, time_points = generate_spikes(
            data, encoder, time_steps, time_interval, device
        )
        output = model(spikes, time_points, method=method).mean(0)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        _, predicted = torch.max(output, 1)
        total_samples += labels.shape[0]
        total_correct += (predicted == labels).sum().item()
        total_loss += loss.item() * data.shape[0]

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    encoder: encoding.StatelessEncoder,
    method: Optional[str] = "euler",
    time_steps: Optional[int] = 100,
    time_interval: Optional[float] = 0.1,
    device: Optional[torch.device] = torch.device("cpu"),
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for data, labels in val_loader:
        labels = labels.long().to(device)

        spikes, time_points = generate_spikes(
            data, encoder, time_steps, time_interval, device
        )
        output = model(spikes, time_points, method=method).mean(dim=0)
        loss = criterion(output, labels)
        total_loss += loss.item() * data.shape[0]

        _, predicted = torch.max(output, 1)
        total_samples += labels.shape[0]
        total_correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def save_model(
    model: nn.Module,
    epoch: int,
    optimizer: optim.Optimizer,
    path: Optional[str] = "checkpoints",
    filename: Optional[str] = None,
    scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
    best_val_acc: float = 0.0,
) -> str:
    if not os.path.exists(path):
        os.makedirs(path)
    if filename is None:
        filename = f"model_epoch_{epoch}.pth"
    checkpoint_path = os.path.join(path, filename)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_model(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    epoch, model_state_dict, optimizer_state_dict, best_val_acc = (
        checkpoint["epoch"],
        checkpoint["model_state_dict"],
        checkpoint["optimizer_state_dict"],
        checkpoint["best_val_acc"],
    )
    model.load_state_dict(model_state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return epoch + 1, best_val_acc


def train_snn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    encoder: encoding.StatelessEncoder,
    scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
    scheduler_metric: Optional[str] = "val_loss",
    epochs: Optional[int] = 100,
    early_stop: Optional[bool] = False,
    patience: Optional[int] = 10,
    method: Optional[str] = "euler",
    time_steps: Optional[int] = 100,
    time_interval: Optional[float] = 0.1,
    model_save: Optional[bool] = True,
    model_save_dir: Optional[str] = "checkpoints",
    resume_from: Optional[str] = None,
    device: Optional[torch.device] = torch.device("cpu"),
) -> Dict[str, Any]:
    best_val_acc = 0.0
    no_improvement_count = 0
    start_epoch = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "alphas": [],
        "coefficients": [],
        "learning_rates": [],
        "best_val_acc": 0.0,
        "best_model_checkpoint_path": None,
    }

    if resume_from is not None:
        start_epoch, best_val_acc = load_model(
            model, resume_from, optimizer, scheduler, device
        )
        history["best_val_acc"] = best_val_acc
        print(f"Resuming training from epoch {start_epoch}")

    print("=" * 80)
    print(
        f"{'Epoch':^6} | {'Train Loss':^10} | {'Train Acc':^9} | {'Val Loss':^10} | {'Val Acc':^9} | {'LR':^8}"
    )
    print("-" * 80)

    for epoch in range(start_epoch, epochs):
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            encoder,
            method,
            time_steps,
            time_interval,
            device,
        )
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            encoder,
            method,
            time_steps,
            time_interval,
            device,
        )

        alphas = {}
        coefficients = {}
        for index, alpha_param in enumerate(model.get_per_layer_alpha()):
            alphas[f"Layer {index}"] = alpha_param.detach().cpu().numpy().tolist()
        for index, coefficient_param in enumerate(model.get_per_layer_coefficient()):
            coefficients[f"Layer {index}"] = (
                coefficient_param.detach().cpu().numpy().tolist()
            )
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["alphas"].append(alphas)
        history["coefficients"].append(coefficients)
        history["learning_rates"].append(current_lr)

        print(
            f"{epoch+1:6d}/{epochs:<4d} | {train_loss:^10.4f} | {train_acc*100:^9.2f}% | {val_loss:^10.4f} | {val_acc*100:^9.2f}% | {current_lr:.2e}"
        )

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_metrics = {
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
                scheduler.step(scheduler_metrics[scheduler_metric])
            else:
                scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history["best_val_acc"] = best_val_acc
            if model_save is True:
                save_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_model(
                    model,
                    epoch,
                    optimizer,
                    model_save_dir,
                    f"best_model_{save_time}.pth",
                    scheduler,
                    best_val_acc,
                )
                print(f"New best validation accuracy: {best_val_acc*100:.2f}%")
                history["best_model_checkpoint_path"] = os.path.join(
                    model_save_dir, f"best_model_{save_time}.pth"
                )
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if early_stop and no_improvement_count >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch+1} (no improvement for {patience} epochs)"
                )
                break

    print("=" * 80)
    print(f"Training completed. Best validation accuracy: {best_val_acc*100:.2f}%")
    if history["best_model_checkpoint_path"]:
        print(f"Best model saved at: {history['best_model_checkpoint_path']}")
    print("=" * 80 + "\n")

    return history


@torch.no_grad()
def test_snn(
    model: nn.Module,
    test_loader: DataLoader,
    encoder: encoding.StatelessEncoder,
    method: str = "euler",
    time_steps: int = 100,
    time_interval: float = 0.1,
    device: torch.device = torch.device("cpu"),
    num_runs: int = 20,
) -> Tuple[float, float]:
    model.eval()
    all_accuracies = []

    for run in range(num_runs):
        total_correct = 0
        total_samples = 0

        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.long().to(device)

            spikes, time_points = generate_spikes(
                data, encoder, time_steps, time_interval, device
            )

            output = model(spikes, time_points, method=method).mean(dim=0)  # [B, C]

            _, predicted = torch.max(output, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        acc = total_correct / total_samples
        all_accuracies.append(acc)
        print(f"Test Run {run + 1}/{num_runs}: Accuracy = {acc * 100:.2f}%")

    all_accuracies = np.array(all_accuracies)
    mean_acc = all_accuracies.mean()
    std_acc = all_accuracies.std()

    print(f"Final Test Results over {num_runs} runs:")
    print(f"Mean Accuracy: {mean_acc * 100:.2f}%")
    print(f"Standard Deviation: {std_acc * 100:.2f}%")

    return mean_acc, std_acc

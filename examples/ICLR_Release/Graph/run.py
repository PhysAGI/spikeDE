import torch, json, argparse, os, random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from spikeDE import SNNWrapper
from spikingjelly.activation_based.encoding import PoissonEncoder
from models import SpikingGCN, DynamicReactiveSpikingGNN
from utils import load_data, train_snn, load_model, test_snn
from typing import Union, List, Optional

_seed_ = 6372
random.seed(_seed_)
torch.manual_seed(_seed_)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)


def setup_model(
    backbone: str,
    in_features: int,
    out_features: int,
    bias: bool,
    tau: float,
    tau_learnable: bool,
    threshold: float,
    integrator: str,
    alpha: Optional[Union[float, List[float], List[List[float]]]],
    multi_coefficient: Optional[Union[List[float], List[List[float]]]],
    alpha_mode: str,
    learn_alpha: bool,
    learn_coefficient: bool,
):
    if backbone == "SGCN":
        base_model = SpikingGCN(
            in_features, out_features, bias, tau, tau_learnable, threshold
        )
    elif backbone == "DRSGNN":
        base_model = DynamicReactiveSpikingGNN(
            in_features, out_features, bias, tau, tau_learnable, threshold
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    snn_net = SNNWrapper(
        base_model,
        integrator=integrator,
        alpha=alpha,
        multi_coefficient=multi_coefficient,
        alpha_mode=alpha_mode,
        learn_alpha=learn_alpha,
        learn_coefficient=learn_coefficient,
    )
    snn_net._set_neuron_shapes(input_shape=(1, in_features))
    return snn_net


def parse_list_or_float(value):
    try:
        return float(value)
    except ValueError:
        import ast

        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, float, int)):
                return parsed
            else:
                raise ValueError
        except Exception:
            raise argparse.ArgumentTypeError(
                f"Invalid format for alpha or coefficient: {value}"
            )


def main():
    parser = argparse.ArgumentParser(description="Run Spiking GNN Experiment")
    parser.add_argument(
        "--backbone", type=str, default="SGCN", choices=["SGCN", "DRSGNN"]
    )
    parser.add_argument("--dataset", type=str, default="Cora", help="Dataset name")
    parser.add_argument(
        "--split", type=str, default="ratio", choices=["public", "random", "ratio"]
    )
    parser.add_argument("--ratio", nargs=3, type=float, default=[0.7, 0.2, 0.1])
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--time_steps", type=int, default=100)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--tau_learnable", action="store_true")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument(
        "--integrator", type=str, default="odeint", choices=["odeint", "fdeint"]
    )
    parser.add_argument(
        "--method",
        type=str,
        default="euler",
        choices=[
            "euler",
            "gl",
            "trap",
            "pred",
            "l1",
            "glmulti"
        ],
    )
    parser.add_argument(
        "--alpha",
        type=parse_list_or_float,
        default=0.5,
        help="Alpha value: float, list of floats, or list of lists of floats",
    )
    parser.add_argument(
        "--coefficients",
        type=parse_list_or_float,
        default=None,
        help="Coefficients: list of floats or list of lists of floats",
    )
    parser.add_argument("--positional_encoding_dim", type=int, default=32)
    parser.add_argument(
        "--positional_encoding_method",
        type=str,
        default="laplacian",
        choices=["laplacian", "random_walk"],
    )
    parser.add_argument("--learn_alpha", action="store_true")
    parser.add_argument("--learn_coefficient", action="store_true")
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else (
                torch.device("mps")
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        )
    else:
        device = torch.device(args.device)

    experiment_name = f"{args.backbone}_{args.integrator}_{args.method}_{args.dataset}"
    experiment_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("checkpoints", experiment_name, experiment_time)
    os.makedirs(log_dir, exist_ok=True)

    in_features, num_classes, train_loader, val_loader, test_loader = load_data(
        root="data",
        name=args.dataset,
        split=args.split,
        batch_size=args.batch_size,
        shuffle=False,
        ratio=args.ratio,
        add_self_loop=True,
        exponent=-0.5,
        add_positional_encoding=True if args.backbone == "DRSGNN" else False,
        positional_encoding_dim=args.positional_encoding_dim,
        positional_encoding_method=args.positional_encoding_method,
    )

    model = setup_model(
        backbone=args.backbone,
        in_features=in_features,
        out_features=num_classes,
        bias=args.bias,
        tau=args.tau,
        tau_learnable=args.tau_learnable,
        threshold=args.threshold,
        integrator=args.integrator,
        alpha=args.alpha,
        multi_coefficient=args.coefficients,
        alpha_mode="auto",
        learn_alpha=args.learn_alpha,
        learn_coefficient=args.learn_coefficient,
    )
    model.to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    encoder = PoissonEncoder()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=25, min_lr=1e-6
    )

    history = train_snn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        encoder=encoder,
        scheduler=scheduler,
        scheduler_metric="val_loss",
        epochs=args.epochs,
        early_stop=args.early_stop,
        patience=args.patience,
        method=args.method,
        time_steps=args.time_steps,
        time_interval=0.1,
        model_save=True,
        model_save_dir=log_dir,
        resume_from=None,
        device=device,
    )

    best_path = history.get("best_model_checkpoint_path")
    if best_path and os.path.exists(best_path):
        load_model(model, best_path, device=device)
        mean_acc, std_acc = test_snn(
            model, test_loader, encoder, args.method, args.time_steps, 0.1, device
        )
        history["test_mean_accuracy"] = mean_acc
        history["test_std_accuracy"] = std_acc

    with open(os.path.join(log_dir, "log.json"), "w") as f:
        json.dump(history, f, indent=4)


if __name__ == "__main__":
    main()

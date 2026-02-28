import torch
import math
import torch.nn as nn
from typing import Tuple


# ========== 1. Sigmoid Surrogate ==========
class SigmoidSurrogate(torch.autograd.Function):
    r"""
    Sigmoid-based surrogate gradient function for Spiking Neural Networks (SNNs).

    This class implements a custom autograd function where the forward pass uses a hard
    Heaviside step function to generate discrete spikes, while the backward pass approximates
    the undefined gradient using the derivative of a scaled sigmoid function.

    Forward Pass:

    $$ S(x) = H(x) = \begin{cases} 1 & \text{if } x \geq 0 \\ 0 & \text{if } x < 0 \end{cases} $$

    Backward Pass (Surrogate Gradient):

    $$ \sigma'(x) = \kappa \cdot \text{sigmoid}(\kappa x) \cdot (1 - \text{sigmoid}(\kappa x)) $$

    Where: $x$ is the input (membrane potential minus threshold, $U - \theta$), $\kappa$ (scale) controls the sharpness of the approximation.

    Attributes:
        scale (float): The scaling factor $\kappa$. Larger values approximate the true
                       step function more closely but may lead to vanishing gradients.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, input: torch.Tensor, scale: float
    ) -> torch.Tensor:
        r"""
        Performs the forward pass using a hard threshold (Heaviside step function).

        Args:
            ctx: Context object to save tensors for the backward pass.
            input: Input tensor representing the membrane potential minus threshold ($U - \theta$).
            scale: Scaling factor ($\kappa$) controlling the sharpness of the surrogate gradient.

        Returns:
            A binary tensor of spikes (0.0 or 1.0).
        """
        ctx.save_for_backward(input)
        ctx.scale = scale
        return (input >= 0).float()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        r"""
        Computes the gradient using the sigmoid derivative as a surrogate.

        Args:
            ctx: Context object containing saved tensors from the forward pass.
            grad_output: Gradient of the loss with respect to the output of the forward pass.

        Returns:
            A tuple containing the gradient with respect to the input
                                       and None for the non-differentiable scale parameter.
        """
        (input,) = ctx.saved_tensors
        scale = ctx.scale
        sigmoid = torch.sigmoid(scale * input)
        surrogate_grad = scale * sigmoid * (1 - sigmoid)
        return grad_output * surrogate_grad, None


def sigmoid_surrogate(input: torch.Tensor, scale: float = 5.0) -> torch.Tensor:
    r"""
    Functional wrapper for the Sigmoid surrogate gradient.

    Allows gradients to flow through the non-differentiable spiking operation during
    backpropagation by replacing the step function's derivative with a smooth sigmoid derivative.

    Args:
        input: Input tensor representing membrane potential minus threshold ($U - \theta$).
        scale: Scaling factor ($\kappa$). Higher values make the surrogate sharper.

    Returns:
        A tensor of binary spikes (0.0 or 1.0) with custom gradient flow.
    """
    return SigmoidSurrogate.apply(input, scale)


# ========== 2. Arctan Surrogate ==========
class ArctanSurrogate(torch.autograd.Function):
    r"""
    Arctangent-based surrogate gradient function for SNNs.

    This method uses the derivative of the arctangent function as the surrogate gradient.
    It features heavier tails compared to the sigmoid, allowing gradients to propagate
    even when the membrane potential is far from the threshold.

    Forward Pass:

    $$ S(x) = H(x) $$

    Backward Pass (Surrogate Gradient):
        
    $$ \sigma'(x) = \frac{\kappa}{1 + (\frac{\pi}{2} \kappa x)^2} $$

    Note: 
        The implementation includes a normalization factor involving $\pi/2$ to ensure stable gradient magnitudes, slightly modifying the standard arctan derivative.

    Attributes:
        scale (float): The scaling factor $\kappa$.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, input: torch.Tensor, scale: float
    ) -> torch.Tensor:
        r"""
        Performs the forward pass using a hard threshold.

        Args:
            ctx: Context object to save tensors for the backward pass.
            input: Input tensor ($U - \theta$).
            scale: Scaling factor ($\kappa$).

        Returns:
            Binary spike tensor.
        """
        ctx.save_for_backward(input)
        ctx.scale = scale
        return (input >= 0).float()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        r"""
        Computes the gradient using the normalized arctangent derivative.

        Args:
            ctx: Context object containing saved tensors.
            grad_output: Upstream gradient from the loss function.

        Returns:
            Gradient w.r.t input and None for scale.
        """
        (input,) = ctx.saved_tensors
        scale = ctx.scale
        return (
            scale / 2 / (1 + (math.pi / 2 * scale * input).pow_(2)) * grad_output,
            None,
        )


def arctan_surrogate(input: torch.Tensor, scale: float = 2.0) -> torch.Tensor:
    r"""
    Functional wrapper for the Arctan surrogate gradient.

    Ideal for deep networks where gradient vanishing is a concern due to its heavy-tailed
    gradient distribution.

    Args:
        input: Input tensor ($U - \theta$).
        scale: Scaling factor ($\kappa$).

    Returns:
        Binary spike tensor with arctan-based gradient flow.
    """
    return ArctanSurrogate.apply(input, scale)


# ========== 3. Piecewise Linear Surrogate ==========
class PiecewiseLinearSurrogate(torch.autograd.Function):
    r"""
    Piecewise Linear (Triangular) surrogate gradient function.

    A computationally efficient approximation that defines a triangular window around
    the threshold. Gradients are constant within the window and zero outside.

    Forward Pass:

    $$ S(x) = H(x) $$

    Backward Pass (Surrogate Gradient):

    $$ \sigma'(x) = \begin{cases} \frac{1}{2\gamma} & \text{if } |x| \leq \gamma \\ 0 & \text{otherwise} \end{cases} $$

    Where $\gamma$ (gamma) defines the width of the active region.

    Attributes:
        gamma (float): Half-width of the linear region.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, input: torch.Tensor, gamma: float
    ) -> torch.Tensor:
        r"""
        Performs the forward pass using a hard threshold.

        Args:
            ctx: Context object to save tensors.
            input: Input tensor ($U - \theta$).
            gamma: Width parameter ($\gamma$).

        Returns:
            Binary spike tensor.
        """
        ctx.save_for_backward(input)
        ctx.gamma = gamma
        return (input >= 0).float()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        r"""
        Computes the gradient using a rectangular window function.

        Args:
            ctx: Context object containing saved tensors.
            grad_output: Upstream gradient.

        Returns:
            Gradient w.r.t input and None for gamma.
        """
        (input,) = ctx.saved_tensors
        gamma = ctx.gamma
        surrogate_grad = ((input >= -gamma) & (input <= gamma)).float() / (2 * gamma)
        return grad_output * surrogate_grad, None


def piecewise_linear_surrogate(input: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    r"""
    Functional wrapper for the Piecewise Linear surrogate gradient.

    Best for high-speed training on resource-constrained hardware or models requiring
    sparse gradient updates.

    Args:
        input: Input tensor ($U - \theta$).
        gamma: Width of the active region ($\gamma$).

    Returns:
        Binary spike tensor with linear-based gradient flow.
    """
    return PiecewiseLinearSurrogate.apply(input, gamma)


# ========== 4. Gaussian Surrogate ==========
class GaussianSurrogate(torch.autograd.Function):
    r"""
    Gaussian-based surrogate gradient function.

    Uses a normalized Gaussian function to approximate the derivative. It offers the
    smoothest profile with exponential decay, providing very localized gradient updates.

    Forward Pass:

    $$ S(x) = H(x) $$

    Backward Pass (Surrogate Gradient):

    $$ \sigma'(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{x^2}{2\sigma^2}} $$

    Where $\sigma$ (sigma) controls the spread (standard deviation) of the gradient.

    Attributes:
        sigma (float): Standard deviation of the Gaussian.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, input: torch.Tensor, sigma: float
    ) -> torch.Tensor:
        r"""
        Performs the forward pass using a hard threshold.

        Args:
            ctx: Context object to save tensors.
            input: Input tensor ($U - \theta$).
            sigma: Standard deviation parameter ($\sigma$).

        Returns:
            Binary spike tensor.
        """
        ctx.save_for_backward(input)
        ctx.sigma = sigma
        return (input >= 0).float()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        r"""
        Computes the gradient using the Gaussian PDF.

        Args:
            ctx: Context object containing saved tensors.
            grad_output: Upstream gradient.

        Returns:
            Gradient w.r.t input and None for sigma.
        """
        (input,) = ctx.saved_tensors
        sigma = ctx.sigma
        surrogate_grad = torch.exp(-(input**2) / (2 * sigma**2)) / (
            sigma * torch.sqrt(torch.tensor(2 * torch.pi))
        )
        return grad_output * surrogate_grad, None


def gaussian_surrogate(input: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    r"""
    Functional wrapper for the Gaussian surrogate gradient.

    Best for precision tasks where only neurons very close to firing should receive updates.

    Args:
        input: Input tensor ($U - \theta$).
        sigma: Spread of the gradient ($\sigma$).

    Returns:
        Binary spike tensor with Gaussian-based gradient flow.
    """
    return GaussianSurrogate.apply(input, sigma)


# =================================The following is NoisyThresholdSpike  ========================================


def noisy_threshold_spike(
    input: torch.Tensor, scale: float = 5.0, training: bool = True, sample: bool = True
) -> torch.Tensor:
    r"""
    Stochastic spiking function using a noisy threshold.

    Instead of a hard spike in the forward pass, this method injects logistic noise
    into the threshold, creating a stochastic soft spike during training. During
    inference (eval mode), it reverts to a hard spike. This acts as both the forward
    mechanism and its own differentiable path (real backward), unlike the surrogate
    methods above.

    Training Mode:

    $$ S(t) = \text{sigmoid}(\kappa(U(t) - \theta) + \epsilon) $$
    
    Where $\epsilon \sim \text{Logistic}(0, 1)$ sampled via inverse CDF:
    
    $$ \epsilon = \log(u) - \log(1-u), \quad u \sim \text{Uniform}(0, 1) $$

    Inference Mode:
    
    $$ S(t) = H(U(t) - \theta) $$

    Args:
        input: Input tensor ($U - \theta$).
        scale: Sharpness parameter ($\kappa$). Higher values make the sigmoid sharper.
        training: If True, applies noise and soft sigmoid. If False, uses hard threshold.
        sample: If True, samples noise per element. If False, uses the mean-field
                approximation (standard sigmoid without noise).

    Returns:
        Soft probabilities during training, binary spikes during eval.
    """
    if not training:
        # Evaluation: hard threshold (same as before)
        return (input >= 0).float()

    # z = (v - threshold) * scale = (v - threshold) / τ
    z = input * scale

    if sample:
        # Sample g ~ Logistic(0, 1) via inverse CDF
        u = torch.rand_like(input).clamp(1e-6, 1 - 1e-6)
        g = torch.log(u) - torch.log1p(-u)
        # Soft spike with noise: S = σ(z + g)
        spike = torch.sigmoid(z + g)
    else:
        # Mean-field: S = σ(z)
        spike = torch.sigmoid(z)

    return spike


class NoisyThresholdSpikeModule(nn.Module):
    r"""
    PyTorch Module wrapper for `noisy_threshold_spike`.

    Automatically tracks the model's training state (`self.training`) to switch between
    stochastic soft spikes and deterministic hard spikes.

    Attributes:
        scale (float): Sharpness parameter ($\kappa$).
        sample (bool): Whether to sample noise or use mean-field.
    """

    def __init__(self, scale=5.0, sample=True):
        super().__init__()
        self.scale = scale
        self.sample = sample

    def forward(self, input):
        return noisy_threshold_spike(
            input, scale=self.scale, training=self.training, sample=self.sample
        )

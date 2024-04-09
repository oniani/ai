import torch


class LeakyReLU(torch.autograd.Function):
    """The Leaky Rectified Linear Unit (Leaky ReLU) activation function."""

    @staticmethod
    def forward(ctx, data: torch.Tensor, alpha: float = 1e-2) -> torch.Tensor:
        """Performs a forward pass."""

        ctx.save_for_backward(data, torch.tensor(alpha).double())
        return torch.where(data < 0.0, alpha * data, data)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Performs a backpropagation."""

        data, alpha = ctx.saved_tensors
        grad = torch.where(data <= 0.0, alpha, 1.0)
        return grad_output * grad


# Alternative impl {{{
#
# Can your activation function be expressed as a combination of existing PyTorch functions?
# If yes, no need to implement the `backward` method.
#
# import torch
# import torch.nn as nn
#
#
# class LeakyReLU(nn.Module):
#     """The Rectified Linear Unit (ReLU) activation function."""
#
#     def __init__(self) -> None:
#         """Inherits from `nn.Module`."""
#
#         super().__init__()
#
#     def forward(self, data: torch.Tensor, alpha: float = 1e-2) -> torch.Tensor:
#         """Performs a forward pass."""
#
#         return torch.where(data < 0.0, alpha * data, data)
#
# }}}


# Testing (gradcheck) {{{

if __name__ == "__main__":
    # Sets the manual seed for reproducible experiments
    torch.manual_seed(0)

    relu = LeakyReLU.apply
    data = torch.randn(4, dtype=torch.double, requires_grad=True)

    # `torch.autograd.gradcheck` takes a tuple of tensors as input, check if your gradient evaluated
    # with these tensors are close enough to numerical approximations and returns `True` if they all
    # verify this condition.
    if torch.autograd.gradcheck(relu, data, eps=1e-8, atol=1e-7):
        print("gradcheck successful")
    else:
        print("gradcheck unsuccessful")

# }}}

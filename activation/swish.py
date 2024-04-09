import torch


class Swish(torch.autograd.Function):
    """The Swish activation function."""

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass."""

        sigmoid = torch.sigmoid(data)
        result = data * sigmoid
        ctx.save_for_backward(sigmoid, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Performs a backpropagation."""

        sigmoid, result = ctx.saved_tensors
        grad = sigmoid + result * (1 - sigmoid)
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
# class Swish(nn.Module):
#     """The Swish activation function."""
#
#     def __init__(self) -> None:
#         """Inherits from `nn.Module`."""
#
#         super().__init__()
#
#     def forward(self, data: torch.Tensor) -> torch.Tensor:
#         """Performs a forward pass."""
#
#         return data * torch.sigmoid(data)
#
# }}}


# Testing (gradcheck) {{{

if __name__ == "__main__":
    # Sets the manual seed for reproducible experiments
    torch.manual_seed(0)

    swish = Swish.apply
    data = torch.randn(4, dtype=torch.double, requires_grad=True)

    # `torch.autograd.gradcheck` takes a tuple of tensors as input, check if your gradient evaluated
    # with these tensors are close enough to numerical approximations and returns `True` if they all
    # verify this condition.
    if torch.autograd.gradcheck(swish, data, eps=1e-8, atol=1e-7):
        print("gradcheck successful")
    else:
        print("gradcheck unsuccessful")

# }}}

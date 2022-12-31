import torch
import torch.nn as nn


class SERF(torch.autograd.Function):
    """The SERF activation function."""

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass."""

        result = data * torch.erf(torch.log(1 + data.exp()))
        ctx.save_for_backward(data, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Performs a backpropagation."""

        data, result = ctx.saved_tensors
        p = 2.0 / torch.pi**0.5 * (-(1 + data.exp()).log().square()).exp()
        swish = nn.SiLU()
        grad = p * swish(data) + result / data
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
# class SERF(nn.Module):
#     """The SERF activation function."""
#
#     def __init__(self) -> None:
#         """Inherits from `nn.Module`."""
#
#         super().__init__()
#
#     def forward(self, data: torch.Tensor) -> torch.Tensor:
#         """Performs a forward pass."""
#
#         return data * torch.erf(torch.log(1 + data.exp()))
#
# }}}


# Testing (gradcheck) {{{

if __name__ == "__main__":
    # Sets the manual seed for reproducible experiments
    torch.manual_seed(0)

    serf = SERF.apply
    data = torch.randn(4, dtype=torch.double, requires_grad=True)

    # `torch.autograd.gradcheck` takes a tuple of tensors as input, check if your gradient evaluated
    # with these tensors are close enough to numerical approximations and returns `True` if they all
    # verify this condition.
    if torch.autograd.gradcheck(serf, data, eps=1e-8, atol=1e-7):  # type: ignore
        print("gradcheck successful")
    else:
        print("gradcheck unsuccessful")

# }}}

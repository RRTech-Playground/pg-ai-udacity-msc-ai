import torch

x = torch.randn(2,2, requires_grad=True)
print(x)

y = x**2
print(y)

## Below we can see the operation that created y, a power operation PowBackward0.
# grad_fn shows the function that generated this variable
print(y.grad_fn)

## The autgrad module keeps track of these operations and knows how to calculate the gradient for each one. In this way,
# it's able to calculate the gradients for a chain of operations, with respect to any one tensor. Let's reduce the
# tensor y to a scalar value, the mean.

z = y.mean()
print(z)

## You can check the gradients for x and y but they are empty currently.
print(x.grad)

## To calculate the gradients, you need to run the `.backward` method on a Variable, `z` for example. This will calculate
# the gradient for `z` with respect to `x`
z.backward()
print(x.grad)
print(x/2)
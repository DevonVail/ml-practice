import torch
import math

dtype = torch.float
device = torch.device("cuda:0")


# Approximate  y = sin(x) with the polynomial y = a + bx + cx^2 + dx^3

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

lr = 1e-6
initial_loss = 1.
for t in range(2000):
    # Forward pass
    y_pred = a + b * x + c * x ** 2 + d * x ** 3
    
    loss = (y_pred-y).pow(2).sum()

    if t % 100 == 99:
        print(f'Iteration t = {t:4d}  loss(t)/loss(0) = {round(loss.item()/initial_loss, 6):10.6f}  a = {a.item():10.6f}  b = {b.item():10.6f}  c = {c.item():10.6f}  d = {d.item():10.6f}')

    loss.backward()

    with torch.no_grad():
        a -= lr * a.grad
        b -= lr * b.grad
        c -= lr * c.grad
        d -= lr * d.grad

        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f"y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3")

import torch
import torch.nn as nn
import numpy as np


def cross_entropy(actual, predicted):
    loss = np.sum(actual * np.log(predicted))
    return loss


# y must be one hot encoder

Y = np.array([1, 0, 0])

# y_pred has probabilities
y_pred_good = np.array([0.7, 0.2, 0.1])
y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, y_pred_good)
l2 = cross_entropy(Y, y_pred_bad)

print(f"Loss1 good_pred numpy :{l1:.4f}")
print(f"Loss1 bad_pred numpy :{l2:.4f}")


loss = nn.CrossEntropyLoss()
Y = torch.tensor([0])
# nsamples * nclasses = 1x3
y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])

l1 = loss(y_pred_good, Y)
l2 = loss(y_pred_bad, Y)

print(f"Loss1 good_pred numpy :{l1:.4f}")
print(f"Loss2 bad_pred numpy :{l2:.4f}")

_, pred1 = torch.max(y_pred_good, 1)
_, pred2 = torch.max(y_pred_bad, 1)

print(f"pred1 :{pred1}")
print(f"pred2 :{pred2}")


##* 3 samples
Y = torch.tensor([2, 0, 1])
# nsamples * nclasses = 1x3

y_pred_good = torch.tensor([[2.0, 1.0, 2.9], [2.0, 1.0, 0.1], [2.0, 3.0, 0.1]])
y_pred_bad = torch.tensor([[0.5, 2.0, 0.3], [0.5, 2.0, 0.3], [0.5, 2.0, 0.3]])

l1 = loss(y_pred_good, Y)
l2 = loss(y_pred_bad, Y)

print(f"Loss1 good_pred numpy :{l1:.4f}")
print(f"Loss2 bad_pred numpy :{l2:.4f}")

_, pred1 = torch.max(y_pred_good, 1)
_, pred2 = torch.max(y_pred_bad, 1)

print(f"pred1 :{pred1}")
print(f"pred2 :{pred2}")

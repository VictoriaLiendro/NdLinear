# NdLinear
### NdLinear vs. Standard Linear Layers: MNIST 

This project compares the performance of **NdLinear** layers ([arXiv:2503.17353](https://arxiv.org/abs/2503.17353)) against standard linear layers with BatchNorm, replicating the experimental framework from the original Batch Normalization paper ([arXiv:1502.03167](https://arxiv.org/abs/1502.03167)).

## Experiment Setup
Data: MNIST dataset
Parameters: 64 batch size, 50 epochs.
Structure: 3 hidden layers, Sigmoid activation function, SGD optimizer

### Models
```python
# NdLinear Model
model_ndlinear = nn.Sequential(
    NdLinear(input_dims=(28, 28,1), hidden_size=(14, 14,4)),
    nn.Sigmoid(),
    NdLinear(input_dims=(14, 14,4), hidden_size=( 7, 7,8)),
    nn.Sigmoid(),
    nn.Flatten(),
    nn.Linear(in_features = 7*7*8, out_features = 10),
    nn.LogSoftmax(dim=1) 

)
# Baseline model
model_base = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(in_features = 28*28, out_features = 100),
    torch.nn.Sigmoid(),
    torch.nn.Linear(in_features = 100, out_features = 100),
    torch.nn.Sigmoid(),
    torch.nn.Linear(in_features = 100, out_features = 100),
    torch.nn.Sigmoid(),
    torch.nn.Linear(in_features = 100, out_features = 10),
    )

# BatchNorm Model 
model_BN = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(in_features = 28*28, out_features = 100),
    torch.nn.BatchNorm1d(num_features=100),
    torch.nn.Sigmoid(),
    torch.nn.Linear(in_features = 100, out_features = 100),
    torch.nn.BatchNorm1d(num_features=100),
    torch.nn.Sigmoid(),
    torch.nn.Linear(in_features = 100, out_features = 100),
    torch.nn.BatchNorm1d(num_features=100),
    torch.nn.Sigmoid(),
    torch.nn.Linear(in_features = 100, out_features = 10),
)

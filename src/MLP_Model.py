
import torch
import torch.nn as nn

def build_mlp_model(input_size, hidden_size, output_size):
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_size, output_size),
        nn.Sigmoid()
    )
    return model

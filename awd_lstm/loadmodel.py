import torch

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f, map_location='cpu')

model_load("awd_lstm/test_v2.pt")

for m in model.modules():
    for param in m.parameters():
        pass

import torch

def setup_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    return device

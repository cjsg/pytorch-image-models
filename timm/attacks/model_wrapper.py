import torch

def model_with_normalization(model, mean, std):
    model.forward_without_normalization = model.forward
    # Assuming that images arrive in [0,255] format
    model.mean = torch.tensor([x * 255 for x in mean]).view(1,3,1,1).cuda()
    model.std = torch.tensor([x * 255 for x in std]).view(1,3,1,1).cuda()
    def forward(x):
        x = x - model.mean
        x = x / model.std
        return model.forward_without_normalization(x)
    setattr(model, 'forward', forward)
    return model

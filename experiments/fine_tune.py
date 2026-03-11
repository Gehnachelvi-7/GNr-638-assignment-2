import torch

def linear_probe(model):

    for name, param in model.named_parameters():

        if "fc" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def last_block_finetune(model):

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():

        if "layer4" in name or "classifier" in name or "fc" in name:
            param.requires_grad = True


def full_finetune(model):

    for param in model.parameters():
        param.requires_grad = True


def selective_20_percent(model):

    params = list(model.parameters())
    total = len(params)

    for p in params:
        p.requires_grad = False

    for p in params[int(0.8 * total):]:
        p.requires_grad = True
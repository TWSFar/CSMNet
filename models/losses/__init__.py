from .focal_loss import FocalLoss
from .cross_entropy_loss import CrossEntropyLoss
from .mse_loss import MSELoss
from .sasc_loss import SASCLoss, SALoss, SCLoss


def build_loss(args):
    obj_type = args.pop('type')

    if obj_type == "FocalLoss":
        return FocalLoss(**args)

    elif obj_type == "CrossEntropyLoss":
        return CrossEntropyLoss(**args)

    elif obj_type == "MSELoss":
        return MSELoss(**args)

    elif obj_type == "SASCLoss":
        return SASCLoss(**args)

    elif obj_type == "SALoss":
        return SALoss(**args)

    elif obj_type == "SCLoss":
        return SCLoss()

    else:
        raise NotImplementedError

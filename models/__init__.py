from .backbones import *
from .transformer import *
from .pet import build_pet
from .crowd_pet import build_CrowdPET

def build_model(args):
    # return build_pet(args)
    return build_CrowdPET(args)


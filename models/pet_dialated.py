"""
PET model and criterion classes
"""

from typing import Callable

from .pet_base import PET_Base
from .transformer.dialated_prog_win_transformer import build_encoder, build_decoder


class PET(PET_Base):
    """
    Point quEry Transformer
    """

    def __init__(self, backbone, num_classes, args=None):
        super().__init__(backbone, num_classes, args)

    def get_build_enc_dec_func(self) -> Callable:
        return build_encoder, build_decoder


def build_pet(args, backbone, num_classes):
    # build model
    model = PET(
        backbone,
        num_classes=num_classes,
        args=args,
    )
    return model

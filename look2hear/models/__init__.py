from .tiger import TIGER
from .causal_tiger import CausalTIGER
from .causal_tiger2 import CausalTIGER2
from .causal_tiger3 import CausalTIGER3
from .tiger_tse import TSE_TIGER
from .tiger_tse_FiLM1 import TSE_TIGER_FiLM1
from .tiger_tse_cross1 import TSE_TIGER_Cross1
from .tiger_tse_SelfCross import TSE_TIGER_SelfCross
from .tiger_dnr import TIGERDNR
from .base_model import BaseModel

__all__ = [
    "TIGER",
    "CausalTIGER",
    "CausalTIGER2",
    "CausalTIGER3",
    "TSE_TIGER",
    "TSE_TIGER_FiLM1",
    "TSE_TIGER_Cross1",
    "TSE_TIGER_SelfCross"
]


def register_model(custom_model):
    """Register a custom model, gettable with `models.get`.

    Args:
        custom_model: Custom model to register.

    """
    if (
        custom_model.__name__ in globals().keys()
        or custom_model.__name__.lower() in globals().keys()
    ):
        raise ValueError(
            f"Model {custom_model.__name__} already exists. Choose another name."
        )
    globals().update({custom_model.__name__: custom_model})


def get(identifier):
    """Returns an model class from a string (case-insensitive).

    Args:
        identifier (str): the model name.

    Returns:
        :class:`torch.nn.Module`
    """
    if isinstance(identifier, str):
        to_get = {k.lower(): v for k, v in globals().items()}
        cls = to_get.get(identifier.lower())
        if cls is None:
            raise ValueError(f"Could not interpret model name : {str(identifier)}")
        return cls
    raise ValueError(f"Could not interpret model name : {str(identifier)}")

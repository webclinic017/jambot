from lightgbm import Sequence

from jambot.ml.dataloading import DiskSequence


class SequenceWrapper(DiskSequence, Sequence):
    """Wrapper class to init with Sequence later
    - importing lightgbm triggers scikit/scipy + 350mb memory
    """
    pass

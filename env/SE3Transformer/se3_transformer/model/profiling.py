import contextlib

import torch


def maybe_nvtx_range(msg, *args, **kwargs):
    """Creates an NVTX range context if not compiling, but skips it under compilation.

    nvtx.range causes a graph break under torch.compile, so we disable it.
    """
    if torch.compiler.is_compiling():
        return contextlib.nullcontext()
    else:
        return torch.cuda.nvtx.range(msg, *args, **kwargs)

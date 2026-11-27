# [Deprecated] Re-export shim for backward compatibility. Use dsa.utils instead.
import warnings

warnings.warn(
    "sglang.srt.layers.attention.nsa.utils is deprecated; "
    "use sglang.srt.layers.attention.dsa.utils instead.",
    DeprecationWarning,
    stacklevel=2,
)
from sglang.srt.layers.attention.dsa.utils import *  # noqa: F401, F403


# --- Ported/stubbed for the v0.5.10 compressed attention backend ---
# These context-parallel helpers are no-ops at tp/cp size 1 (the supported
# config here); v0.5.15 removed them from nsa.utils.
def is_nsa_prefill_cp_round_robin_split(forward_batch=None) -> bool:
    return False


def assert_tensor_identical_across_cp_ranks(t, tag, forward_batch) -> None:
    return None

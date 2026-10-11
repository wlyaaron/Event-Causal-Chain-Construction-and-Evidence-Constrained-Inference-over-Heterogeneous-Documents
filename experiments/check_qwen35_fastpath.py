"""Check that Qwen3.5's optional training kernels are selected on a GPU host.

Exit codes: 0 = fast path available, 1 = fallback/error, 2 = no CUDA GPU.
This does not measure throughput or assert a successful backward pass.
"""

from __future__ import annotations

import importlib.metadata


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def main() -> int:
    import torch

    for package in ("torch", "transformers", "flash-linear-attention",
                    "fla-core", "causal-conv1d", "liger-kernel"):
        print(f"{package}: {_version(package)}")

    if not torch.cuda.is_available():
        print("UNVERIFIED: no CUDA GPU is attached; Triton kernels cannot be tested.")
        return 2

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    except Exception as exc:
        print(f"FALLBACK: Qwen3.5 module import failed: {type(exc).__name__}: {exc}")
        return 1

    required = (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
    )
    missing = [name for name in required if getattr(qwen, name, None) is None]
    if missing:
        print("FALLBACK: unavailable Qwen3.5 fast-path operators: " + ", ".join(missing))
        return 1

    print("FAST_PATH_IMPORTS_OK: all four Qwen3.5 optional operators are available.")
    print("Run a short forward/backward benchmark to verify execution and speed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

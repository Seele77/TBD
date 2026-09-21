# FlashVID component

This directory contains the FlashVID-based, training-free spatiotemporal token compression used by Token-Budget Distillation (TBD).

The implementation is integrated into the parent repository rather than used as a standalone training project. The main entry point is `flashvid.flashvid(...)`; TBD enables it from `llava/train/train.py` with the `--enable_flashvid` and `--flashvid_*` arguments.

The bundled `lmms-eval` directory provides the evaluation harness used by the paper. See the repository-level [README](../README.md) for installation, training, merging, and evaluation commands.

The FlashVID component is distributed under the MIT license; see [`LICENSE`](LICENSE). Upstream notices in the source files must be preserved.

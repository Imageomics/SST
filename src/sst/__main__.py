"""Command line entry point for SST.

Each subcommand is implemented in its own module exposing ``add_arguments(parser)`` and ``run(args)``.
Dispatch is done in two phases so the heavy imports (torch, transformers, the subcommand modules) only
happen for the subcommand actually being run. ``sst`` and ``sst --help`` list the subcommands straight
from ``SUBCOMMANDS`` without importing anything, and ``sst <command> ...`` imports only that command's
module.
"""

import argparse
import importlib
import sys

from sst import __version__

SUBCOMMANDS = {
    "segment": ("sst.segment", "Propagate a support mask across query images."),
    "segment-and-crop": ("sst.segment_and_crop", "Segment specimens and crop each to the mask."),
    "retrieve": ("sst.trait_retrieval", "Rank query images by trait cycle-consistency."),
    "prepare-mask": ("sst.prepare_starter_mask", "Convert a white mask to an object-id mask."),
    "mask-from-crop": ("sst.get_mask_from_crop", "Build a full-image mask from a transparent crop."),
}


def _top_level_parser():
    """Parser used only to render top-level help and usage. It does not import subcommand modules."""
    parser = argparse.ArgumentParser(prog="sst", description="Static Segmentation by Tracking.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for name, (_module, help_text) in SUBCOMMANDS.items():
        subparsers.add_parser(name, help=help_text, add_help=False)
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help", "--version"):
        _top_level_parser().parse_args(argv)
        return

    command = argv[0]
    if command not in SUBCOMMANDS:
        _top_level_parser().error(f"invalid choice: {command!r}")

    module_name, help_text = SUBCOMMANDS[command]
    module = importlib.import_module(module_name)
    parser = argparse.ArgumentParser(prog=f"sst {command}", description=help_text)
    module.add_arguments(parser)
    module.run(parser.parse_args(argv[1:]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rename all files in a selected folder to lowercase using a GUI."""

import os
from tkinter import Tk, filedialog


def rename_files_to_lowercase(directory: str) -> None:
    """Recursively rename files in *directory* to lowercase."""
    for root, _dirs, files in os.walk(directory):
        for name in files:
            lower_name = name.lower()
            if name != lower_name:
                src = os.path.join(root, name)
                dst = os.path.join(root, lower_name)
                os.rename(src, dst)


def run_gui() -> None:
    """Launch a folder picker and rename files inside it to lowercase."""
    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select folder to lowercase")
    root.destroy()
    if folder:
        rename_files_to_lowercase(folder)


if __name__ == "__main__":  # pragma: no cover
    run_gui()

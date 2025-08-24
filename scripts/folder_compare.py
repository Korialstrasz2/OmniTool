#!/usr/bin/env python3
"""Compare file names between two folders ignoring extensions."""

import os
from tkinter import Tk, filedialog, messagebox


def gather_stems(directory: str) -> set[str]:
    """Return a set of file name stems for files in *directory* (non-recursive)."""
    stems = set()
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            stem, _ = os.path.splitext(name)
            stems.add(stem)
    return stems


def compare_folders(dir1: str, dir2: str) -> tuple[list[str], list[str]]:
    """Return names present only in dir1 and only in dir2."""
    stems1 = gather_stems(dir1)
    stems2 = gather_stems(dir2)
    only1 = sorted(stems1 - stems2)
    only2 = sorted(stems2 - stems1)
    return only1, only2


def run_gui() -> None:
    """Prompt for two folders and show file name differences."""
    root = Tk()
    root.withdraw()
    first = filedialog.askdirectory(title="Select first folder")
    second = filedialog.askdirectory(title="Select second folder")
    root.destroy()
    if not first or not second:
        return
    only1, only2 = compare_folders(first, second)
    if not only1 and not only2:
        message = "Both folders contain the same file names."
    else:
        parts = []
        if only1:
            parts.append("Only in first folder:\n" + "\n".join(only1))
        if only2:
            parts.append("Only in second folder:\n" + "\n".join(only2))
        message = "\n\n".join(parts)
    messagebox.showinfo("Folder Compare", message)


if __name__ == "__main__":  # pragma: no cover
    run_gui()

#!/usr/bin/env python3
"""Rename files in one folder based on selections from another using a GUI.

Open two windows listing files from two folders. Selecting a file in the
left window and then clicking a file in the right window renames the right
file to match the left file's stem while preserving the original extension.
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox


def choose_directory(title: str) -> str:
    """Prompt the user to select a directory and return its path."""
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder


class DualFileRenamer:
    """GUI application for renaming files across two folders."""

    def __init__(self, left_dir: str, right_dir: str):
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.selected_left: str | None = None

        self.left_root = tk.Tk()
        self.left_root.title(f"Left: {left_dir}")

        self.right_root = tk.Toplevel(self.left_root)
        self.right_root.title(f"Right: {right_dir}")

        self.left_list = tk.Listbox(self.left_root, width=40)
        self.left_list.pack(fill=tk.BOTH, expand=True)
        self.left_list.bind("<<ListboxSelect>>", self.on_left_select)

        self.right_list = tk.Listbox(self.right_root, width=40)
        self.right_list.pack(fill=tk.BOTH, expand=True)
        self.right_list.bind("<<ListboxSelect>>", self.on_right_select)

        self.refresh(self.left_list, self.left_dir)
        self.refresh(self.right_list, self.right_dir)

    def refresh(self, widget: tk.Listbox, directory: str) -> None:
        """Populate *widget* with file names from *directory*."""
        widget.delete(0, tk.END)
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if os.path.isfile(path):
                widget.insert(tk.END, name)

    def on_left_select(self, _event) -> None:
        """Store the currently selected file from the left list."""
        sel = self.left_list.curselection()
        if sel:
            self.selected_left = self.left_list.get(sel[0])

    def on_right_select(self, _event) -> None:
        """Rename the selected file in the right list using the left selection."""
        sel = self.right_list.curselection()
        if sel and self.selected_left:
            right_name = self.right_list.get(sel[0])
            left_stem, _left_ext = os.path.splitext(self.selected_left)
            _right_stem, right_ext = os.path.splitext(right_name)
            new_name = left_stem + right_ext
            src = os.path.join(self.right_dir, right_name)
            dst = os.path.join(self.right_dir, new_name)
            if os.path.exists(dst):
                messagebox.showerror("Error", f"File {new_name} already exists.")
                return
            os.rename(src, dst)
            self.refresh(self.right_list, self.right_dir)

    def run(self) -> None:
        """Start the GUI event loop."""
        self.left_root.mainloop()


def run_gui() -> None:
    """Launch directory pickers and start the renamer GUI."""
    left = choose_directory("Select left folder")
    if not left:
        return
    right = choose_directory("Select right folder")
    if not right:
        return
    DualFileRenamer(left, right).run()


if __name__ == "__main__":  # pragma: no cover
    run_gui()

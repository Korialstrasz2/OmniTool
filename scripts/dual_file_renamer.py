#!/usr/bin/env python3
"""Rename files in one folder based on selections from another using a GUI.

Two windows display the contents of two folders. Each file is shown with a
thumbnail preview so images, videos and other common media types can be easily
identified. Selecting a thumbnail in the left window and then clicking one in
the right window renames the right file to match the left file's stem while
preserving the original extension.
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

# Thumbnails and treeview rows are displayed three times larger than the
# previous defaults so previews are easier to view.
THUMB_SIZE = 64 * 3


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

        style = ttk.Style()
        style.configure("Thumb.Treeview", rowheight=THUMB_SIZE)

        self.placeholder = ImageTk.PhotoImage(
            Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), "gray")
        )
        self.left_images: dict[str, ImageTk.PhotoImage] = {}
        self.right_images: dict[str, ImageTk.PhotoImage] = {}

        # Left window widgets
        self.left_list = ttk.Treeview(self.left_root, show="tree", style="Thumb.Treeview")
        self.left_list.pack(fill=tk.BOTH, expand=True)
        self.left_list.bind("<<TreeviewSelect>>", self.on_left_select)

        # Right window widgets
        self.right_list = ttk.Treeview(
            self.right_root, show="tree", style="Thumb.Treeview"
        )
        self.right_list.pack(fill=tk.BOTH, expand=True)
        self.right_list.bind("<<TreeviewSelect>>", self.on_right_select)

        self.refresh(self.left_list, self.left_dir, self.left_images)
        self.refresh(self.right_list, self.right_dir, self.right_images)

    def refresh(
        self, widget: ttk.Treeview, directory: str, images: dict[str, ImageTk.PhotoImage]
    ) -> None:
        """Populate *widget* with file names and thumbnails from *directory*."""
        widget.delete(*widget.get_children())
        images.clear()
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if os.path.isfile(path):
                thumb = self.make_thumbnail(path)
                photo = ImageTk.PhotoImage(thumb) if thumb else self.placeholder
                images[name] = photo
                widget.insert("", tk.END, iid=name, text=name, image=photo)

    def on_left_select(self, _event) -> None:
        """Store the currently selected file from the left list."""
        sel = self.left_list.selection()
        if sel:
            self.selected_left = sel[0]

    def on_right_select(self, _event) -> None:
        """Rename the selected file in the right list using the left selection."""
        sel = self.right_list.selection()
        if not sel:
            return

        right_name = sel[0]
        right_path = os.path.join(self.right_dir, right_name)

        if self.selected_left:
            left_stem, _left_ext = os.path.splitext(self.selected_left)
            _right_stem, right_ext = os.path.splitext(right_name)
            new_name = left_stem + right_ext
            dst = os.path.join(self.right_dir, new_name)
            if os.path.exists(dst):
                messagebox.showerror("Error", f"File {new_name} already exists.")
                return
            os.rename(right_path, dst)
            self.refresh(self.right_list, self.right_dir, self.right_images)

    def make_thumbnail(self, path: str) -> Image.Image | None:
        """Return a thumbnail Image for *path* if possible."""
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}:
                img = Image.open(path)
            elif ext in {".mp4", ".mov", ".avi", ".mkv"}:
                try:
                    import cv2
                except Exception:
                    return None
                cap = cv2.VideoCapture(path)
                success, frame = cap.read()
                cap.release()
                if not success:
                    return None
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return None
            img.thumbnail((THUMB_SIZE, THUMB_SIZE))
            return img
        except Exception:
            return None

    def run(self) -> None:
        """Start the GUI event loop."""
        self.left_root.mainloop()


def run_gui() -> None:
    """Launch directory pickers and start the renamer GUI."""
    left = choose_directory("Select the folder whose file names are the source (not renamed)")
    if not left:
        return
    right = choose_directory("Select the folder whose files are to be renamed")
    if not right:
        return
    DualFileRenamer(left, right).run()


if __name__ == "__main__":  # pragma: no cover
    run_gui()

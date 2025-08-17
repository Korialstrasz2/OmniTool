#!/usr/bin/env python3
"""GUI tool to remove history for a domain from selected browser."""

import glob
import os
import shutil
import sqlite3
import tempfile
import tkinter as tk
from tkinter import messagebox


CHROMIUM_PATHS = {
    "Chrome": os.path.join(
        os.environ.get("LOCALAPPDATA", ""),
        "Google",
        "Chrome",
        "User Data",
        "Default",
        "History",
    ),
    "Edge": os.path.join(
        os.environ.get("LOCALAPPDATA", ""),
        "Microsoft",
        "Edge",
        "User Data",
        "Default",
        "History",
    ),
    "Brave": os.path.join(
        os.environ.get("LOCALAPPDATA", ""),
        "BraveSoftware",
        "Brave-Browser",
        "User Data",
        "Default",
        "History",
    ),
}

FIREFOX_GLOB = os.path.join(
    os.environ.get("APPDATA", ""),
    "Mozilla",
    "Firefox",
    "Profiles",
    "*",
    "places.sqlite",
)


def find_browsers():
    """Return mapping of detected browsers to history file paths."""
    browsers = {}
    for name, path in CHROMIUM_PATHS.items():
        if os.path.exists(path):
            browsers[name] = path
    for path in glob.glob(FIREFOX_GLOB):
        browsers.setdefault("Firefox", path)
    return browsers


def delete_chromium_history(db_path, domain):
    """Delete history entries for domain in Chromium-based browsers."""
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.close()
    shutil.copy2(db_path, temp.name)
    conn = sqlite3.connect(temp.name)
    cursor = conn.cursor()
    like = f"%{domain}%"
    cursor.execute("SELECT id FROM urls WHERE url LIKE ?", (like,))
    ids = [row[0] for row in cursor.fetchall()]
    if ids:
        marks = ",".join("?" * len(ids))
        cursor.execute(f"DELETE FROM visits WHERE url IN ({marks})", ids)
        cursor.execute(f"DELETE FROM urls WHERE id IN ({marks})", ids)
    conn.commit()
    conn.close()
    shutil.copy2(temp.name, db_path)
    os.remove(temp.name)


def delete_firefox_history(db_path, domain):
    """Delete history entries for domain in Firefox."""
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.close()
    shutil.copy2(db_path, temp.name)
    conn = sqlite3.connect(temp.name)
    cursor = conn.cursor()
    like = f"%{domain}%"
    cursor.execute("SELECT id FROM moz_places WHERE url LIKE ?", (like,))
    ids = [row[0] for row in cursor.fetchall()]
    if ids:
        marks = ",".join("?" * len(ids))
        cursor.execute(
            f"DELETE FROM moz_historyvisits WHERE place_id IN ({marks})", ids
        )
        cursor.execute(f"DELETE FROM moz_places WHERE id IN ({marks})", ids)
    conn.commit()
    conn.close()
    shutil.copy2(temp.name, db_path)
    os.remove(temp.name)


def run_gui():
    browsers = find_browsers()
    if not browsers:
        browsers = {k: v for k, v in CHROMIUM_PATHS.items()}
        browsers["Firefox"] = FIREFOX_GLOB

    root = tk.Tk()
    root.title("Browser History Cleaner")

    tk.Label(root, text="Browser:").grid(row=0, column=0, padx=5, pady=5)
    browser_var = tk.StringVar(root)
    browser_var.set(next(iter(browsers)))
    tk.OptionMenu(root, browser_var, *browsers.keys()).grid(
        row=0, column=1, padx=5, pady=5
    )

    tk.Label(root, text="Domain:").grid(row=1, column=0, padx=5, pady=5)
    domain_entry = tk.Entry(root, width=40)
    domain_entry.grid(row=1, column=1, padx=5, pady=5)

    def on_delete():
        browser = browser_var.get()
        domain = domain_entry.get().strip()
        if not domain:
            messagebox.showerror("Error", "Please enter a domain")
            return
        path = browsers.get(browser)
        if not path or not os.path.exists(path):
            messagebox.showerror(
                "Error", f"History database not found for {browser}."
            )
            return
        try:
            if browser == "Firefox" or path.endswith("places.sqlite"):
                delete_firefox_history(path, domain)
            else:
                delete_chromium_history(path, domain)
            messagebox.showinfo(
                "Success", f"Removed history for {domain} in {browser}."
            )
        except (sqlite3.Error, OSError) as exc:
            messagebox.showerror("Error", str(exc))

    tk.Button(root, text="Delete", command=on_delete).grid(
        row=2, column=0, columnspan=2, pady=10
    )

    root.mainloop()


if __name__ == "__main__":  # pragma: no cover
    run_gui()

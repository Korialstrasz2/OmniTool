#!/usr/bin/env python3
"""GUI tool to delete unwanted cookies using allowlist."""

import glob
import json
import os
import shutil
import sqlite3
import tempfile
import tkinter as tk
from tkinter import messagebox
from urllib.parse import urlparse

WHITELIST_FILE = os.path.join(os.path.dirname(__file__), "cookies_whitelist.txt")

CHROMIUM_ROOTS = {
    "Chrome": os.path.join(
        os.environ.get("LOCALAPPDATA", ""),
        "Google",
        "Chrome",
        "User Data",
        "Default",
    ),
    "Edge": os.path.join(
        os.environ.get("LOCALAPPDATA", ""),
        "Microsoft",
        "Edge",
        "User Data",
        "Default",
    ),
    "Brave": os.path.join(
        os.environ.get("LOCALAPPDATA", ""),
        "BraveSoftware",
        "Brave-Browser",
        "User Data",
        "Default",
    ),
}

FIREFOX_PROFILES = glob.glob(
    os.path.join(os.environ.get("APPDATA", ""), "Mozilla", "Firefox", "Profiles", "*")
)


def naive_etld1(host):
    if not host:
        return None
    parts = host.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


def load_whitelist():
    if os.path.exists(WHITELIST_FILE):
        with open(WHITELIST_FILE, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    return set()


def save_whitelist(domains):
    with open(WHITELIST_FILE, "w", encoding="utf-8") as f:
        for d in sorted(domains):
            f.write(d + "\n")


def append_whitelist(domains):
    existing = load_whitelist()
    to_add = domains - existing
    if not to_add:
        return
    with open(WHITELIST_FILE, "a", encoding="utf-8") as f:
        for d in sorted(to_add):
            f.write(d + "\n")


def extract_chromium_domains(root):
    domains = set()
    history_path = os.path.join(root, "History")
    if os.path.exists(history_path):
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.close()
        shutil.copy2(history_path, temp.name)
        conn = sqlite3.connect(temp.name)
        for (url,) in conn.execute("SELECT url FROM urls"):
            host = urlparse(url).hostname
            etld = naive_etld1(host)
            if etld:
                domains.add(etld)
        conn.close()
        os.remove(temp.name)
    bookmarks_path = os.path.join(root, "Bookmarks")
    if os.path.exists(bookmarks_path):
        try:
            with open(bookmarks_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            stack = [data]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    if node.get("type") == "url" and node.get("url"):
                        host = urlparse(node["url"]).hostname
                        etld = naive_etld1(host)
                        if etld:
                            domains.add(etld)
                    else:
                        stack.extend(node.values())
                elif isinstance(node, list):
                    stack.extend(node)
        except Exception:
            pass
    return domains


def extract_firefox_domains(profile):
    domains = set()
    places = os.path.join(profile, "places.sqlite")
    if os.path.exists(places):
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.close()
        shutil.copy2(places, temp.name)
        conn = sqlite3.connect(temp.name)
        for (url,) in conn.execute("SELECT url FROM moz_places"):
            host = urlparse(url).hostname
            etld = naive_etld1(host)
            if etld:
                domains.add(etld)
        conn.close()
        os.remove(temp.name)
    return domains


def enumerate_chromium_cookies(root):
    cookies_path = os.path.join(root, "Cookies")
    if not os.path.exists(cookies_path):
        cookies_path = os.path.join(root, "Network", "Cookies")
    cookies = []
    if os.path.exists(cookies_path):
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.close()
        shutil.copy2(cookies_path, temp.name)
        conn = sqlite3.connect(temp.name)
        for row in conn.execute("SELECT host_key, name FROM cookies"):
            cookies.append((row[0], row[1]))
        conn.close()
        os.remove(temp.name)
    return cookies, cookies_path


def enumerate_firefox_cookies(profile):
    cookies_path = os.path.join(profile, "cookies.sqlite")
    cookies = []
    if os.path.exists(cookies_path):
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.close()
        shutil.copy2(cookies_path, temp.name)
        conn = sqlite3.connect(temp.name)
        for row in conn.execute("SELECT host, name FROM moz_cookies"):
            cookies.append((row[0], row[1]))
        conn.close()
        os.remove(temp.name)
    return cookies, cookies_path


def delete_chromium_cookies(db_path, cookies):
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.close()
    shutil.copy2(db_path, temp.name)
    conn = sqlite3.connect(temp.name)
    for domain, name in cookies:
        conn.execute("DELETE FROM cookies WHERE host_key=? AND name=?", (domain, name))
    conn.commit()
    conn.close()
    shutil.copy2(temp.name, db_path)
    os.remove(temp.name)


def delete_firefox_cookies(db_path, cookies):
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.close()
    shutil.copy2(db_path, temp.name)
    conn = sqlite3.connect(temp.name)
    for domain, name in cookies:
        conn.execute("DELETE FROM moz_cookies WHERE host=? AND name=?", (domain, name))
    conn.commit()
    conn.close()
    shutil.copy2(temp.name, db_path)
    os.remove(temp.name)


def find_browsers():
    browsers = {}
    for name, root in CHROMIUM_ROOTS.items():
        cookies = os.path.join(root, "Cookies")
        if not os.path.exists(cookies):
            cookies = os.path.join(root, "Network", "Cookies")
        if os.path.exists(cookies):
            browsers[name] = {"root": root, "type": "chromium"}
    for profile in FIREFOX_PROFILES:
        if os.path.exists(os.path.join(profile, "cookies.sqlite")):
            browsers["Firefox"] = {"root": profile, "type": "firefox"}
            break
    return browsers


class CookieCleanerGUI:
    def __init__(self):
        self.browsers = find_browsers()
        if not self.browsers:
            self.browsers = {**{k: {"root": v, "type": "chromium"} for k, v in CHROMIUM_ROOTS.items()},
                             "Firefox": {"root": FIREFOX_PROFILES[0] if FIREFOX_PROFILES else "", "type": "firefox"}}
        self.root = tk.Tk()
        self.root.title("Cookie Cleaner")

        tk.Label(self.root, text="Browser:").grid(row=0, column=0, padx=5, pady=5)
        self.browser_var = tk.StringVar(self.root)
        self.browser_var.set(next(iter(self.browsers)))
        tk.OptionMenu(self.root, self.browser_var, *self.browsers.keys()).grid(row=0, column=1, padx=5, pady=5)

        tk.Button(self.root, text="Manage Whitelist", command=self.manage_whitelist).grid(row=0, column=2, padx=5, pady=5)
        tk.Button(self.root, text="Load Cookies", command=self.load_cookies).grid(row=1, column=0, columnspan=3, pady=5)

        self.listbox = tk.Listbox(self.root, selectmode=tk.MULTIPLE, width=80)
        self.listbox.grid(row=2, column=0, columnspan=3, padx=5, pady=5)

        tk.Button(self.root, text="Add domain to whitelist", command=self.add_selected_to_whitelist).grid(row=3, column=0, columnspan=3, pady=5)
        tk.Button(self.root, text="Select All", command=self.select_all).grid(row=4, column=0, columnspan=3, pady=5)
        tk.Button(self.root, text="Delete Selected", command=self.delete_selected).grid(row=5, column=0, columnspan=3, pady=5)

        self.cookies = []

        self.build_allowlist(self.browsers[self.browser_var.get()])

    def build_allowlist(self, info):
        domains = load_whitelist()
        if info["type"] == "chromium":
            history = extract_chromium_domains(info["root"])
        else:
            history = extract_firefox_domains(info["root"])
        new_domains = history - domains
        if new_domains and messagebox.askyesno(
            "Whitelist",
            "I found these cookies whose site is in the history. Should I add them to the whitelist?",
        ):
            append_whitelist(new_domains)
            domains.update(new_domains)
        return domains

    def load_cookies(self):
        browser = self.browser_var.get()
        info = self.browsers[browser]
        allow = self.build_allowlist(info)
        if info["type"] == "chromium":
            cookies, path = enumerate_chromium_cookies(info["root"])
        else:
            cookies, path = enumerate_firefox_cookies(info["root"])
        self.cookie_db_path = path
        self.cookies = []
        self.listbox.delete(0, tk.END)
        total = len(cookies)
        for domain, name in cookies:
            etld = naive_etld1(domain.lstrip('.'))
            if etld not in allow:
                self.cookies.append((domain, name))
                self.listbox.insert(tk.END, f"{domain}\t{name}")
        filtered = total - len(self.cookies)
        if filtered:
            messagebox.showinfo(
                "Info", f"{filtered} cookies are not shown as the domain is in the whitelist"
            )
        messagebox.showinfo("Info", f"Loaded {len(self.cookies)} cookies to delete")

    def selected_indices(self):
        return list(self.listbox.curselection())

    def select_all(self):
        self.listbox.select_set(0, tk.END)

    def add_selected_to_whitelist(self):
        indices = self.selected_indices()
        if not indices:
            messagebox.showerror("Error", "No cookies selected")
            return
        wl = {naive_etld1(self.cookies[i][0].lstrip('.')) for i in indices}
        append_whitelist(wl)
        self.load_cookies()

    def delete_selected(self):
        indices = self.selected_indices()
        if not indices:
            messagebox.showerror("Error", "No cookies selected")
            return
        to_delete = [self.cookies[i] for i in indices]
        if not messagebox.askyesno("Confirm", f"Delete {len(to_delete)} cookies?"):
            return
        info = self.browsers[self.browser_var.get()]
        try:
            backup_path = self.cookie_db_path + ".bak"
            shutil.copy2(self.cookie_db_path, backup_path)
            if info["type"] == "chromium":
                delete_chromium_cookies(self.cookie_db_path, to_delete)
            else:
                delete_firefox_cookies(self.cookie_db_path, to_delete)
            messagebox.showinfo(
                "Success",
                f"Deleted {len(to_delete)} cookies (backup saved to {backup_path})",
            )
            self.load_cookies()
        except OSError as exc:
            if getattr(exc, "winerror", None) in (32, 1224):
                messagebox.showerror("Error", "Please close the browser and retry")
            else:
                messagebox.showerror("Error", str(exc))

    def manage_whitelist(self):
        wl = load_whitelist()
        top = tk.Toplevel(self.root)
        top.title("Whitelist")
        listbox = tk.Listbox(top, selectmode=tk.SINGLE, width=40)
        listbox.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        for d in sorted(wl):
            listbox.insert(tk.END, d)

        entry = tk.Entry(top)
        entry.grid(row=1, column=0, padx=5, pady=5)

        def add_domain():
            d = entry.get().strip()
            if d:
                wl.add(d)
                listbox.insert(tk.END, d)
                entry.delete(0, tk.END)

        def remove_selected():
            sel = listbox.curselection()
            if sel:
                d = listbox.get(sel[0])
                wl.discard(d)
                listbox.delete(sel[0])

        tk.Button(top, text="Add", command=add_domain).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(top, text="Remove", command=remove_selected).grid(row=2, column=0, padx=5, pady=5)
        tk.Button(top, text="Save", command=lambda: (save_whitelist(wl), top.destroy())).grid(row=2, column=1, padx=5, pady=5)

    def run(self):
        self.root.mainloop()


def run_gui():
    CookieCleanerGUI().run()


if __name__ == "__main__":  # pragma: no cover
    run_gui()

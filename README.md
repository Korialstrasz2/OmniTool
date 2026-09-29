# OmniTool workspace — review build

A local Python toolbox with a searchable, paginated catalog, shared parameter forms, bounded jobs, and encrypted private bundles. This is a substantial migration, not a cosmetic patch. Read `docs/AUDIT.md`, `docs/VAULT_SECURITY.md`, and `docs/TEST_REPORT.md` before merging or entrusting private source to it.

## Start

Use Python 3.11 or newer in a virtual environment. From the repository root:

```sh
python -m pip install -r requirements-core.txt
python app.py
```

On Windows, `start.bat` creates `.venv` when necessary. On Unix-like systems, use `./start.sh`. The terminal prints a random **local access code**, used to sign into `http://127.0.0.1:5000`. That code is not an encryption passphrase. Keep the terminal private. The server binds to loopback, uses Waitress rather than Flask's debugger, and starts every bundle locked. Do not expose it through a tunnel or reverse proxy. It is not a multiuser or hosted service.

Core installation does not install every optional tool dependency. A card marked “Needs setup” means a Python module is missing, not that a tool has been health-checked. `python -m pip install -r requirements.txt` installs the original broader dependency set; review compatibility in a disposable virtual environment first. Tkinter, FFmpeg, browser binaries, and other external tools may still require separate installation. The retained NumPy/OpenCV constraints have not been comprehensively modernized.

## What changed

The new shell has search, folder/category filtering, availability filters, public-tool favorites, grid/list views, keyboard search, pagination, explicit empty/error states, and a common Jobs page. Public tools are discovered from `tools/**/tool.json`; existing tools are adapted through the root `tools.json`. Adding an ordinary Python tool no longer requires editing Flask routes or front-end code.

Hunyuan3D's manager, routes, and template are removed. Browser History Cleaner, Cookie Cleaner, and Lowercase Renamer remain on disk but are disabled in the launcher pending safety fixes. The Base64/PDF launcher now supplies actual parameters. CSV parsing is rebuilt for quoted multiline fields. Passive Media Capture is exposed as its own job-based tool. A new read-only Duplicate Finder groups equal-size files by SHA-256.

The old all-in-one backend is replaced by `omnitool_core/{catalog,jobs,locks,vault,web}.py`. Jobs use argument arrays, never a shell, with a bounded queue, concurrency, runtime, and output buffer. The old arbitrary-path lyrics log endpoint and hard-coded session key are gone. Optional AI metadata recovery is off by default in the new Lyrics form.

## Migration boundaries

Original scripts are retained; this does not rewrite every tool's internal UI or logic. Tkinter tools and Prompt Creator still open their own interfaces. Advanced legacy Media/Lyrics controls, saved presets, AI ping, and provider-specific screens are not all reproduced in the shared forms; remaining flags are available through the scripts' CLIs. These need an acceptance pass before this replaces a daily-use installation.

Catalog folders currently come from manifest metadata: they are not a drag-and-drop collection manager. Search returns paginated results, but discovery still scans manifests per request. There is no claim of infinite scale. Private tools remain in the separate encrypted-bundle view rather than being merged into the public search index. Jobs and their output are session-scoped and memory-only; restarting loses them. Ordinary jobs have a 30-minute execution timeout. Private access lasts ten minutes per unlock and expires even while a job is running.

The source-level audit did not test third-party services or run browser database modifications. A “Ready” badge means basic declared prerequisites are present, not that an external provider or desktop GUI has been verified. Three risky tools are deliberately unavailable from the new launcher; running their original scripts directly bypasses that precaution.

## Adding tools and private bundles

Read `docs/ADDING_TOOLS.md`. Private plaintext must live outside a Git working tree. A locked tool is one encrypted bundle containing one manifest. A locked folder is one encrypted bundle containing files and zero or more tool manifests. Public observers see opaque `.otvault` filenames and ciphertext, not internal paths or names. This does not hide the existence, padded size, or Git history of those blobs.

```sh
# Run from the OmniTool root. Replace the source path with a private local directory.
python -m omnitool_core.vault seal /private/my-tool --kind tool --name "My private tool"
```

The terminal prompts for a passphrase and confirmation. Nothing is uploaded automatically, and plaintext originals are unchanged. Commit only the generated `vaults/<random-id>.otvault` file. Never submit your passphrase in an issue, PR, chat, source file, command-line argument, or GitHub Actions job.

## Verify

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
node tests/csv.test.js
```

Encryption uses PyCA's AES-256-GCM and scrypt. The container and its integration are new and **not independently audited**. Confidentiality at rest is not an execution sandbox, reliable secure erasure, or a way to recall previously public source. Use non-sensitive test bundles first.

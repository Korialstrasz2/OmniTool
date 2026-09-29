# OmniTool

A local personal toolbox with a searchable manifest-driven library, shared workspace pages, explicit file-operation previews, and encrypted private bundles. Use Python 3.11 or newer. This is evolving software: test destructive operations on disposable copies and read the tool-specific limits before normal use.

## Start

```sh
python -m pip install -r requirements-core.txt
python app.py
```

Windows: `start.bat`. Unix-like systems: `./start.sh`. The terminal prints a random **local access code** for `http://127.0.0.1:5000`. This code is separate from an encrypted-bundle passphrase. Keep the terminal private. The application binds to loopback and is not a hosted, multiuser, or remotely accessible service. Do not expose it through a tunnel or reverse proxy.

Core startup does not install every tool dependency. For optional features, install deliberately into the same virtual environment:

```sh
python -m pip install -r requirements-conversion.txt
python -m pip install -r requirements-content.txt
```

The broad legacy `requirements.txt` remains for the download tools and older setups. It has not received a full dependency-isolation audit. External binaries and browser extensions require their own setup. A Ready badge means declared Python prerequisites are present, not that a provider, model, browser extension, or optional native binary was tested successfully.

## Current tools

| Tool | Current workflow |
| --- | --- |
| Browser History Cleaner / Cookie Cleaner | Profile-local extension, exact-host preview, verified encrypted backup, explicit deletion. Extension installation is required per profile. |
| Lowercase Renamer | File-only lowercase preview, no-overwrite apply, persistent undo/recovery journal. |
| Dual File Renamer | Two-pane matching, staged mappings, preview/apply, shared undo. |
| Folder Compare | Read-only recursive path/content comparison and JSON/CSV reports. |
| Base64 / PDF to PNG | Strict input inspection, bounded rendering, new output folder only. |
| Prompt Creator | In-workspace editor and one explicitly configured local model; no standalone FastAPI server or cloud fallback. |
| Lyrics Workbench | Inspect selected tracks, prepare sidecar/HTTPS-provider lyrics, review, then write tagged **copies** into a new folder. Originals are not modified. |
| CSV Workbench | Browser-only cell editing, undo, row filtering, text sorting, trim/deduplicate/remove-empty operations, and CSV/JSON export. |
| Media Harvester / Passive Media Capture | Existing scripts through tracked jobs; further provider/preset/dependency modernization remains. |
| Duplicate Finder | Read-only size/SHA-256 duplicate report; no automatic deletion. |

Hunyuan3D's old manager was removed. Existing tool IDs remain stable across these migrations. Public tool favorites, category/status filters, grid/list layouts, and pagination remain available. Categories are manifest metadata, not a drag-and-drop filesystem manager. Discovery rescans manifests; no infinite-capacity claim is made.

## Guides and migration boundaries

- `docs/SAFE_MAINTENANCE.md`: browser-extension permissions/backups, lowercase renaming, and recovery limits.
- `docs/FILE_WORKBENCH.md`: dual renaming, comparison, image/PDF conversion, and task limits.
- `docs/CONTENT_WORKBENCH.md`: Prompt Creator, non-destructive lyrics copies, CSV editing, and retired legacy options.
- `docs/ADDING_TOOLS.md`: add `tools/<name>/tool.json` plus a Python entrypoint; no per-tool launcher UI is needed for ordinary Python tools.
- `docs/VAULT_SECURITY.md`: encrypted bundles and their threat model. Earlier audit/test reports describe their particular implementation snapshots, not current CI results.

Some specialized tasks display their status/results in their own pages rather than the generic Jobs screen. Public job history and previews are memory-only. Restarting loses them; navigating away does not necessarily stop a running worker. Content tasks have bounded queues, a three-minute wall-clock limit, and session-scoped results usable for ten minutes after finishing. A completed filesystem action cannot be undone merely by clearing its displayed result.

## Private tools and folders

A public Git repository can store ciphertext bundles, but public observers still see their existence, padded sizes, count, and history. Keep private plaintext **outside Git working trees**, seal locally with the CLI, and commit only the generated opaque `vaults/*.otvault` file:

```sh
python -m omnitool_core.vault seal /private/my-tool --kind tool --name "My private tool"
```

Use `--kind folder` for files and multiple tools. The CLI prompts for the passphrase; never place it in a command-line argument, issue, PR, chat, or workflow. Plaintext originals are unchanged, and no automatic upload takes place. Store the passphrase and a recoverable backup independently. There is no reset or recovery backdoor.

**The vault container and integration are not independently audited.** Encryption does not retract previously public code, sandbox tool execution, secure a compromised host, or guarantee memory/file erasure. A running tool needs plaintext; temporary execution files, deliberate exports, crashes, backups, detached processes, and Windows permissions require careful review. Read the complete threat model before using valuable private code. Browser-maintenance backups use a separate format; they are not `.otvault` bundles.

## Tests

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
node tests/csv.test.js
node --test tests/csv_workbench.test.js tests/browser_maintenance.test.js
```

GitHub Actions runs Linux/Windows regression configurations. Read actual check results rather than interpreting a workflow file as a passing test. Browser-maintenance API tests are mocked, and neither provider availability nor the independent cryptographic/security review is established by a passing test suite.

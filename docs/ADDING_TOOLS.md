# Adding public and locked tools

## Public Python tool

Create `tools/my-tool/main.py` and `tools/my-tool/tool.json`. Discovery reads JSON but does not import or run the entrypoint. Paths are relative to the manifest folder; ordinary public tools run with that folder as their working directory.

```json
{
  "schema_version": 1,
  "id": "my-tool",
  "name": "My tool",
  "description": "Explain the actual operation and its limits.",
  "folder": "Files / Inspect",
  "tags": ["files", "inspect"],
  "kind": "python",
  "entrypoint": "main.py",
  "requires": [],
  "parameters": [
    {"name": "input", "label": "Input directory", "type": "path", "required": true},
    {"name": "limit", "label": "Limit", "type": "integer", "flag": "--limit", "default": 100, "min": 1, "max": 10000},
    {"name": "preview", "label": "Preview only", "type": "boolean", "flag": "--preview", "default": true}
  ]
}
```

Parameters appear in manifest order. A parameter without a flag is positional. Types are text, path, integer, boolean, and choice. Choice fields require `choices`. Only Python entrypoints are supported for automatic launch. `requires` lists import module names, not package installer commands. `platforms` can contain `win32`, `linux`, or `darwin`. A disabled tool must have `"disabled": true`; a status description alone does not disable it. For file-changing tools set `"risk": "writes-files"` to require confirmation; this declaration is not an OS permission sandbox.

Tool IDs must be unique across the public catalog. One broken manifest is reported without removing valid tools. Folder names are categories shown in the filter, not enforced disk boundaries. Keep meaningful tags and short, task-oriented descriptions. Prefer read-only/dry-run defaults and explicit output paths. Do not install dependencies automatically during discovery or ordinary execution. Tool code and dependencies are trusted local code.

## Locked tool

Build the same directory and manifest **outside every Git working tree**, for example `D:\PrivateTools\example` or `/home/you/private-tools/example`. The sealing CLI refuses a source under a `.git` ancestor. This is a precaution, not protection against all alternate Git configurations or manually force-adding files.

From the OmniTool repository root:

```sh
python -m omnitool_core.vault seal /home/you/private-tools/example --kind tool --name "Private example"
```

It asks for a passphrase through `getpass`, confirms it, encrypts the complete directory and metadata, and verifies the result before writing a randomly named `vaults/*.otvault`. There must be exactly one `tool.json`. No passphrase or private metadata is written beside the ciphertext. File bytes remain unchanged; executable file permissions are not preserved. The original source directory remains plaintext and is not deleted.

## Locked folder

A folder bundle may contain arbitrary regular files and zero or more tool folders. Put a `tool.json` next to each Python entrypoint to make it executable in the bundle view. Seal the parent with `--kind folder`. Nested tool paths, their manifests, internal file counts, and the folder name are encrypted together. Publicly there is still just one blob. To minimize visible tool counts, put related tools in one folder bundle rather than publishing one encrypted blob per tool.

Unlock through Locked bundles. Each bundle is scoped to the signed-in browser session and has a fixed ten-minute lease. Unlocking alone does not execute code. Launching requires separate consent, then materializes a private temporary copy for the subprocess. Locked entrypoints run from the bundle root, not their nested manifest directory: use `Path(__file__).resolve().parent` to locate files packaged beside the script. Modules needing a permanent external working directory should use explicit input/output parameters. External dependency installation remains local and manual.

Lock now, Lock all, sign-out, and expiry revoke access and stop associated jobs. Outputs and transient edits inside the temporary bundle are removed on cleanup; use an explicitly chosen external output directory for results you intend to keep. Such results are plaintext and are not protected by re-locking the bundle. Downloads from the file browser are also plaintext copies.

## Restore, edit, and reseal

```sh
python -m omnitool_core.vault restore vaults/OPAQUE_ID.otvault /home/you/private-tools/restored-copy
```

The destination must be new and outside a Git working tree. Edit locally, then seal again. Each seal creates a new random ciphertext filename. Replacing a bundle in Git does not revoke access to older ciphertext under its old passphrase. Store independent backups and the passphrase in a password manager. There is no reset, backdoor, or server-side recovery.

Limits in this review build: 32 MiB plaintext archive, at most 4,096 archive members, no symlinks, no compressed archive entries, portable relative filenames, and at most eight simultaneously unlocked bundles. These are deliberate resource/safety limits, not claims of arbitrary capacity. Folder snapshots do not preserve empty directories, permissions, or filesystem timestamps.

# OmniTool source audit and migration decisions

Scope: the public default-branch commit `a63c046bab06ae9d234d4a43ba9009b025bfe309` inspected through the connected GitHub repository. This is a source review, not a claim that all scripts, third-party services, or browser database schemas were exercised. Hunyuan removal follows the owner's request; no claim is made that every Hunyuan model or project is globally obsolete.

| Original tool | Decision | Evidence and next work |
| --- | --- | --- |
| Hunyuan3D 2.1 Manager | Remove | Delete its registry entry, manager routes/helpers, and template. Do not delete a user's downloaded model repository or weights. Git history is untouched. |
| Browser History Cleaner | Disable; rewrite | SQL matches `%domain%` against the whole URL; copied live SQLite databases are overwritten; the code can force-kill Edge. Replace with explicit browser/profile selection, host matching, closure checks, backups, preview, and schema-aware transactions. Firefox bookmark/history relationships require particular care. |
| Cookie Cleaner | Disable; rewrite | `naive_etld1` uses the last two labels, mishandling names such as `example.co.uk`. Live database replacement and limited profile discovery need redesign. Use Public Suffix List-aware matching, preview/backup/restore, and local-only settings. |
| Lowercase Renamer | Disable; rebuild | Calls `os.rename` recursively without collision planning. Implement two-phase rename plans, case-only rename handling, collision rejection, preview, and a recoverable operation journal before enabling. |
| Dual File Renamer | Keep; modernize | Has a destination-exists check, but selecting a right-hand item immediately changes the file. Keep thumbnail matching; add an explicit proposed mapping, confirmation, error handling, background thumbnails, and undo journal. Still uses Tkinter in this build. |
| Folder Compare | Keep; expand | Compares nonrecursive sets of stems, ignores extensions, and does not compare file bytes. Label its current behavior honestly; add recursive path/content modes and exportable reports. |
| Base64 PDF to PNG | Fix launcher; harden script next | The old registry supplied no required positional input. The new form supplies paths and bounded DPI/page controls. The script still needs strict Base64, input/render memory limits, and collision-safe outputs; CLI callers can bypass form bounds. |
| Comfy Prompt Builder | Keep as Prompt Creator; integrate later | The actual service calls KoboldCpp; it is not a general GGUF/ComfyUI execution manager. Its separate FastAPI UI, permissive CORS, arbitrary caller-supplied endpoint probes, and provider contracts need review. Not silently replaced with a newer model. |
| Lyrics Embedder | Keep; update | Uses provider fallbacks, including an HTTP ChartLyrics URL, and writes tags in place. Add backups/dry-run, provider health tests, bounded concurrency/rate limits, and explicit AI consent. New UI starts with lower concurrency and AI recovery off. Old defaults/presets and every advanced flag are not all migrated. |
| CSV Editor | Rebuild parser; expand later | Old parsing splits physical lines before interpreting quotes, breaking multiline fields. New local parser handles quoted newlines/doubled quotes/BOM; bounded preview and optional spreadsheet-formula neutralization are provided. Full cell editing, transformations, and large-file streaming remain future work. |
| Media Harvester | Keep; modularize | Active runs blocked a request for up to 30 minutes. Common job execution removes that blocking launch path. Retain authorized-use scope, audit credential/URL logging, isolate provider adapters, and add deliberate dependency management, presets, and integration tests. Do not assume every provider works today. |

`media_passive_capture.py` was already present; it is now a separate catalog entry rather than a newly invented downloader. The new Duplicate Finder is genuinely new and read-only: size grouping plus SHA-256, no deletion/linking/renaming. Hard-linked paths may appear as equal-content results; no disk-space recovery is claimed.

## Delete obsolete infrastructure, not useful capabilities

The old monolithic backend, unneeded Hunyuan page, and superseded Media/Lyrics templates are replaced. The original script implementations are retained rather than silently deleted. The old arbitrary-path lyrics log-reader route, `shell=True` generic launcher, debug server, and hard-coded `change-me` session secret are removed from the new app. This does not secure independently launched legacy services or direct CLI use of disabled scripts.

Runtime logs, browser allowlists, downloads, caches, and configuration files must not be tracked. The new ignore rules cover the known paths, but ignore rules do not protect already tracked files or Git history. Do not copy private source into a public checkout before encrypting it. The repository currently lacks a clearly declared license; decide a license for public tools separately from encrypted private content rather than assuming one.

## Expansion order

1. Finish acceptance/security testing and Windows cleanup behavior; fix the three disabled tools before exposing them again. Add secret scanning and a reproducible dependency/CI policy.
2. Build a unified File Workbench for rename, compare, and duplicates with preview/confirm/apply/undo as shared operations. The read-only duplicate scan is the first piece, not an automatic deduplication system.
3. Add dependency diagnostics, explicit installs, per-tool environments, saved presets, structured progress events, and durable public-job history. Keep private state out of public metadata and disk logs.
4. Add JSON/YAML validation/format conversion, checksums, safe archive inspection, and batch document/image conversion as small manifest-based tools with bounded resource usage. These are proposals, not implemented features.
5. Migrate the remaining Tkinter and separate Prompt Creator UIs into the shared workspace, then add saved collections, drag-and-drop organization, and schema-generated richer forms as needed. Optimize discovery with caching/indexing only after measuring realistic library sizes.

The UI has finite pagination and resource limits. It supports adding tools without per-tool UI code, not unlimited physical capacity. Encrypted folders are snapshots, not a transparent encrypted mount or a permission toggle on ordinary tracked folders.

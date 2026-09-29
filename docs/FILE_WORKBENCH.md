# File Workbench: second tool-repair batch

Follow-up to merged PR #119, based on main commit `e5838fec1c458f31e6be9b0858a1eb1308ae320b`. This batch replaces the unsafe/limited implementations of Dual File Renamer, Folder Compare, and Base64/PDF to PNG. Browser maintenance and encrypted-bundle code are not changed. The shared rename engine gains a second plan type, while retaining support for existing lowercase journals.

## Start and dependencies

Run the ordinary `app.py` entrypoint or `start.bat` / `start.sh`. It registers both the earlier maintenance pages and these new pages. The three existing catalog cards now open `/files/dual`, `/files/compare`, and `/files/convert`. No Tkinter windows remain for these tools. The original script paths remain CLI adapters; the immediate-on-selection rename implementation is removed, not just hidden.

Comparison and filename-based matching use the standard library. Still-image thumbnails and conversion require explicit installation into the workspace environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-conversion.txt
```

On Unix-like systems use `.venv/bin/python` instead. The optional file constrains Pillow to `>=12.1,<13` and PyMuPDF to `>=1.26.7,<2`; it is not a full dependency lock. Video thumbnails use a separately installed `ffmpeg` found on PATH. Missing/failed thumbnail support does not prevent filename selection. No ordinary tool launch installs or upgrades packages. The old conversion BAT no longer creates another environment or upgrades pip/dependencies automatically.

## Dual File Renamer

Choose separate, non-overlapping reference and target directories, then load them. Only top-level ordinary files are selectable. Each pane has name search and 24-item pages. A selected image can be previewed; supported video formats can request a first-frame preview through ffmpeg. Preview failures fall back to filenames. File bytes, URLs, and thumbnails are not sent to a remote service.

Select one reference and one target, then **Stage mapping**. Selection and staging do not rename anything. The target receives the reference's stem and keeps its own final suffix: `Holiday.jpg` paired with `DSC001.PNG` becomes `Holiday.PNG`. This retains the old suffix semantics; it does not treat compound suffixes such as `.tar.gz` as a single extension. Each reference and target may be used once per batch. Renaming is limited to 500 changes per operation.

**Preview staged changes** builds a server-side plan, displays all proposed changes/conflicts, and requires `RENAME N` before execution. Previews are session-scoped, fixed-ten-minute, and single-use for apply. Reloading changed folders requires staging again. The backend rechecks both inventories and the reviewed plan; changing the client request cannot substitute another target folder or arbitrary filename.

Occupied names, directory conflicts, duplicate destinations, and case/Unicode-folding collisions block the whole plan. Cycles/swaps are deliberately rejected even if a selected file would move away. Reference files, directory names, and file contents are never changed. Links, Windows reparse points, special files, nonportable names, and common development directories are excluded; scans are bounded to 20,000 entries per root.

The existing `Renamer` performs two-phase sibling staging using OS no-overwrite moves and a write-ahead journal under the user's OS state directory (outside Git). Ordinary errors attempt rollback. The shared **Undo / interrupted operations** view can recover applied or interrupted dual and lowercase operations. Recovery validates identities and refuses altered files or occupied original names. Old version-1 lowercase journals without a `kind` field remain supported. Journals are name/identity records, **not content backups**. Stop other writers first; this is not a sandbox against malicious filesystem races, guaranteed crash durability, or a substitute for independent backups.

CLI example (JSON pair file and paths are local):

```sh
python scripts/dual_file_renamer.py preview --reference /data/reference --target /data/targets --mapping /private/pairs.json
python scripts/dual_file_renamer.py apply --reference /data/reference --target /data/targets --mapping /private/pairs.json --fingerprint REVIEWED_SHA256 --confirm "RENAME 1"
python scripts/dual_file_renamer.py undo --operation OPERATION_ID --confirm UNDO
```

`pairs.json` is a list such as `[{"reference":"Holiday.jpg","source":"DSC001.PNG"}]`. Preview before apply. The old no-argument desktop behavior is intentionally retired.

## Folder Compare

Default **Content** mode recursively compares exact relative paths, including extensions. Matching files are first compared by size; equal-size candidates are streamed through SHA-256. Results distinguish same content, different content, left-only, right-only, directories on both sides, and file/directory conflicts. Matching paths are case-sensitive. It is not an automatic synchronization or duplicate-removal tool.

**Paths** mode compares names and types without claiming their bytes match. **Stems** mode preserves the older extension-insensitive concept, but keeps parent paths and reports ambiguous duplicate stems instead of collapsing them into a set. Empty directories are included in path/content mode, not stems mode. Recursion can be turned off explicitly.

Reports expose skipped entries and whether the requested scan scope was complete. Symlinks/junctions, special files, and excluded development directories are not silently treated as matching. Unreadable or concurrently changed data fails the scan rather than producing a misleading completed report. Inventories are rechecked after hashing, but an actively changing filesystem is still not an atomic snapshot.

The two roots are limited to 20,000 scanned entries each. The combined content-hash budget defaults to 1,024 MiB and can be set from 1 to 4,096 MiB; exceeding it fails with an instruction to narrow the scan or adjust the budget. Reports are paginated at 50 rows with path/status filtering. JSON export contains the complete report, counts, and exclusions. CSV exports the result rows and prefixes spreadsheet-formula-like values with an apostrophe; choose JSON for exact values and the full scope metadata.

```sh
python scripts/folder_compare.py /data/left /data/right --mode content --hash-budget-mib 1024
python scripts/folder_compare.py /data/left /data/right --mode paths --top-level
```

CLI reports are JSON on stdout. Neither comparison mode writes into either input tree.

## Base64 / PDF to PNG

Choose a local input file and a **new output directory whose parent already exists**. Inspect first: the preview reports source type/hash, selected/omitted pages, decoded size, and intended pixel dimensions. Type `CONVERT N` to publish the reviewed batch. Changing an option invalidates the browser preview; the backend separately stores the reviewed options and verifies the input SHA-256 again before rendering.

Strict standard Base64 is accepted, including whitespace-wrapped text and supported `data:...;base64,` prefixes. Invalid characters, noncanonical padding, unsupported data URI types, empty data, malformed/encrypted PDFs, and animated/multiframe images are rejected. Supported raw formats are PDF, PNG, JPEG, WEBP, BMP, GIF and TIFF; still-image conversion supports one frame only. EXIF orientation is applied and image metadata is stripped; alpha is preserved where present. PDF pages render to opaque RGB PNGs.

Limits apply in both CLI and library code: 32 MiB encoded/raw input, 24 MiB decoded bytes, DPI 36–600, 1–100 selected pages, 20 million pixels per image, 100 million pixels per batch, and 512 MiB rendered output. A page limit may deliberately omit later PDF pages, and that omission is reported. These are resource limits, not a guarantee against every native-decoder vulnerability or compressed-content workload.

Outputs are built in a new sibling `.omnitool-convert-*.partial` directory. After rendering, a no-overwrite directory rename publishes the entire batch. A pre-existing or concurrently created output path is never overwritten. The image output is `image.png`; PDF output is `page-001.png` etc. `conversion.json` records the source path/hash and rendered dimensions: it is local plaintext metadata, so inspect it before sharing output.

Ordinary failures attempt staging cleanup. Process termination, power loss, or cleanup errors can leave a partial directory beside the requested output. Check it manually; no broad automatic purge runs. Canceling at the moment publication finishes can leave a completed new output directory even if the task UI says canceled. Input files and pre-existing output files remain untouched. There is no conversion undo/delete endpoint.

```sh
python scripts/b64pdf2png.py /data/input.pdf --inspect --dpi 150 --max-pages 10
python scripts/b64pdf2png.py /data/input.pdf --out /data/new-output --dpi 150 --max-pages 10 --expected-sha256 REVIEWED_SHA256
```

Direct CLI conversion is an explicit command and does not ask for the UI confirmation phrase. Its content/size bounds still apply, but the workbench's outer subprocess wall-clock limit does not apply to direct library or CLI calls.

## Task and security boundaries

Comparison and conversion use a dedicated fixed worker command, not shell commands. Two worker tasks run concurrently, with at most 16 retained/running tasks, a 120-second wall-clock timeout, and a 16 MiB response check. On Linux the worker also applies a 1 GiB virtual-address-space and 90-second CPU limit. Windows/macOS do not get that same address-space/CPU enforcement. Thumbnail workers are separately limited to two concurrent requests and an eight-second outer timeout; ffmpeg has its own six-second timeout. This is process isolation for resilience, not an audited OS sandbox.

Status, reports, and errors appear in each tool's workbench page, **not yet in the shared Jobs & output screen**. Stay on the page while running. Navigating away loses the page's task handle but does not automatically cancel a worker. Finished task handles expire after ten minutes when checked or pruned; a workspace restart drops memory-only reports and previews. Task IDs, inventories, and results are owner-session checked. Ordinary filesystem paths/results are sensitive local plaintext; this feature does not turn them into locked-vault content.

The parent application's local sign-in, host/origin, CSRF, CSP, and no-store policies cover these routes. Browser and vault implementations are unchanged. The workspace access token is removed from worker environments. Workers still execute trusted local code with the user's permissions; do not expose the local server to the network or treat decoder subprocesses as permission sandboxes.

## Validation recorded for this batch

Local environment: Linux, Python 3.13.5, Pillow 12.3.0, PyMuPDF 1.26.7. Only generated/disposable fixtures were used. No real user directories, browser profiles, external APIs, or private source were modified.

- `python -m pytest -q tests/test_rename.py tests/test_file_workbench.py tests/test_conversion.py tests/test_file_web.py`: **84 passed, 1 skipped**. The skip is the entire Flask HTTP module because Flask is absent and installation was unavailable here. The unchanged lowercase regression file was fetched and hash-verified against main. This is not a run of the entire repository's test suite.
- Real filesystem tests cover dual preview/apply/undo, collision/staleness rejection, injected ordinary failure rollback, interrupted journal recovery, and old lowercase journal compatibility. Comparison tests cover recursion, modes, ambiguity, byte budgets, changing files and exclusions. Conversion tests exercise actual Pillow/PyMuPDF output, strict Base64, password rejection, limits, EXIF/alpha/metadata, output-race refusal, staging cleanup, source-change rejection, subprocess execution and owner checks. Still-image thumbnail generation and stale selection were tested; native video thumbnails were not.
- `python tests/file_ui_smoke.py`: Chromium smoke **passed using mocked APIs** with actual Jinja templates/CSS/JavaScript. Checks include 24-item dual pagination, selection/staging without apply, typed apply flow, 50-row comparison pagination/export links, conversion preview invalidation/apply, 390px layouts without horizontal overflow, and no JavaScript page errors. This does not verify real Flask HTTP, authentication, or an end-to-end UI/filesystem operation.
- Python compilation and JavaScript syntax checks passed.

The supplied GitHub Actions workflow runs Linux/Windows and Python 3.11/3.13 with Flask and conversion dependencies installed, then runs the targeted file/maintenance HTTP tests and retained browser/CSV JavaScript logic. **A workflow definition is not a passing run**; inspect its actual checks on the PR. Native browser-extension tests, real Windows/macOS acceptance, ffmpeg thumbnails, crash/power-loss behavior and an independent security review remain distinct acceptance work.

Run locally with `python -m pip install -r requirements-dev.txt`, then the pytest command above. UI smoke additionally needs `playwright` and an installed Chromium (`python -m playwright install chromium`). Start with copies of non-sensitive files and verify undo/recovery and conversion output publication on the actual computer before normal use.

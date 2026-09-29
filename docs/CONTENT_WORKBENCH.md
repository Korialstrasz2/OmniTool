# Content tools: Prompt Creator, Lyrics Workbench, CSV Workbench

This batch follows the file-workbench migration. It does not change the browser maintenance, rename, converter, download, or encrypted-bundle implementations. Existing catalog IDs remain stable.

## Prompt Creator

Open Prompt Creator from the library. Compose the idea, select/edit the system instruction, choose output-token and temperature limits, and explicitly generate. Results are editable text, never rendered as provider-controlled HTML. Copy and text export are explicit. No prompts are stored in localStorage or in workspace job logs.

Start the model server yourself. Configuration is read from the environment inherited when OmniTool starts:

```powershell
$env:OMNITOOL_PROMPT_URL = "http://127.0.0.1:5001"
$env:OMNITOOL_PROMPT_KIND = "kobold"
.\.venv\Scripts\python.exe app.py
```

`KOBOLD_HOST` is retained as a fallback environment-variable name. URLs must be HTTP loopback origins with an explicit port from 1024 to 65535; `localhost` is normalized to 127.0.0.1 without DNS. Paths, credentials, queries, fragments, external addresses, and wildcard bind addresses are rejected. The request body cannot select an endpoint. There are no proxy-environment lookups, redirects, automatic server scans, retry chains, or cloud fallbacks.

Kobold mode uses only `/api/v1/model` and `/api/v1/generate`, with explicit `max_length` and temperature. For a local chat-completions-compatible server, set `OMNITOOL_PROMPT_KIND=local-chat` and `OMNITOOL_PROMPT_MODEL` to the model ID it serves. This mode uses `/v1/models` and `/v1/chat/completions`; a model ID is required for generation. It is a local protocol adapter, not an integration with a paid cloud account. API-key-protected backends are not implemented in this batch. The backend itself is trusted local software and can retain or transmit data according to its own settings.

The previous standalone FastAPI server and its wildcard CORS policy are removed. `scripts/comfy_prompt_builder/start.py` is a non-mutating migration notice. The old static UI is removed; the original system-prompt text file is retained for reference, but the new editor uses its explicit editable instruction rather than reading `COMFY_SYSTEM_PROMPT` automatically. The default instruction is task-oriented; copy any desired custom instruction into the editor.

Generation is limited to 16–2048 output tokens, 0–2 temperature, 12 KiB idea and 12 KiB instruction, 512 KiB backend response, and 64 KiB generated text. A status check makes one explicit request. Cancel/timeout stops OmniTool's worker, not necessarily the model server's own generation. No automatic abort request is sent to a potentially shared model server.

## Lyrics Workbench

Install `requirements-content.txt` in the same environment. The workflow is **scan → select → prepare → read the lyrics → type WRITE N → create copies**. No original audio file is opened for writing.

Scan a specific music directory. Up to 5,000 filesystem entries, 50 supported tracks, 128 MiB per file, and 512 MiB of audio per batch are accepted. Linked files/junctions and common development directories are excluded. Unreadable or unsupported audio is reported instead of silently treated as writable. Supported containers are MP3, FLAC, M4A/MP4 audio, Ogg Vorbis, and Ogg Opus. Raw AAC is no longer advertised as taggable.

Nothing is selected automatically. Artist/title/album may be corrected in the table for lookup only. The code does not copy those corrections into the audio tags. Existing lyrics are protected unless replacement in the **new copies** is explicitly enabled. Source SHA-256 values bind preparation and apply to the inspected audio.

### Sources

**Local sidecars (default):** `Track.lrc` or `Track.txt` beside `Track.mp3`; LRC takes precedence. Sidecars must be UTF-8 and fit the 32 KiB text limit. LRC timestamps and recognized metadata lines are removed from the plain lyrics, while the original synchronized text is also exported as an LRC sidecar. This path performs no network request.

**LRCLIB:** only after explicit consent, send the selected artist/title/album/duration to `https://lrclib.net/api/get`. Audio, local paths, cookies, and credentials are not sent. Requests use verified HTTPS without environmental proxies or redirects, with connection/read deadlines and bounded responses. Processing is serial with a one-second delay after successful lookups; there are no automatic retries or speculative fallback requests. One prepare operation runs/queues at a time in the workspace. This is not a globally coordinated rate limiter across multiple independently launched OmniTool instances.

Provider artist/title must match after Unicode normalization and case folding; duration must be within two seconds. Missing, instrumental, mismatched, or failed results are itemized and skipped. Matching metadata is not a guarantee that returned lyrics are correct: review the full plain text. Provider availability was not established with a live public lookup in the implementation environment.

The old ChartLyrics HTTP fallback, lyrics.ovh fallback chain, automatic AI metadata recovery, in-place tag writing, and plaintext cache/log files are removed from this tool. This deliberately reduces legacy provider/options breadth. No paid API call occurs. Source/provider licensing and your authorization to use lyric text remain your responsibility; no copyrighted lyric fixture is included in the tests.

### Copy publication and recovery

Choose a **new** output directory outside the source tree, with an existing parent. Only ready selected tracks are copied. Relative subdirectories are retained. Plain lyrics are written through Mutagen and read back for verification. MP3 USLT is replaced on the copy; existing SYLT frames on that copy are removed when replacement is approved, and any reviewed synchronized lyrics are supplied as a sidecar, not native timed-tag embedding. MP3 ID3 representation may be normalized by Mutagen; original files remain the recovery source. MP4 uses the lyrics atom, while FLAC/Vorbis/Opus use lyrics comments. Other audio tags are not deliberately edited, but no byte-for-byte container-preservation claim is made for a rewritten copy.

All output is prepared in a randomly named sibling `.omnitool-lyrics-*.partial` directory, then published with an OS no-overwrite directory rename. Duplicate output names and source changes are rejected. An output path occupied before or during publication is never overwritten. A report in the output directory records filenames, provider labels, and source hashes, but not full lyrics. Output files/sidecars/report are plaintext; nothing is automatically committed or uploaded.

Ordinary failures attempt staging cleanup. Killing the process, a crash, power loss, permissions errors, or a filesystem race may leave staging output. Deletion is ordinary cleanup, not secure erasure. A final directory can be published just before cancellation or a lost HTTP response; inspect the destination before retrying. Independent backups are still appropriate. Discard unwanted copies manually; this tool has no destructive “restore originals” action because it never rewrites them.

The legacy script now provides read-only CLI scan/sidecar-preview compatibility:

```sh
python scripts/lyrics_embedder.py /music/album
python scripts/lyrics_embedder.py /music/album --out /music-tagged/new-album
```

These commands print JSON and do not write audio. Old mutation/AI/provider flags are rejected by argparse instead of silently mapped to changed behavior. Reviewed copy creation is available in the workspace.

## CSV Workbench

The selected file never goes to the Python backend. A same-origin Web Worker reads/decodes/parses it, keeps the table in memory, applies changes, and serializes exports. No formula or uploaded script is executed. Web Worker support is required; there is no main-thread parsing fallback.

Choose UTF-8, UTF-16 little-endian, or Windows-1252. Quoted newlines, doubled quote characters, BOMs, empty/trailing cells, and explicit delimiters are supported. Auto-detection is heuristic; check the chosen separator. NUL characters produce an encoding error rather than accepting likely misdecoded binary data. Changing input settings does not discard edits until an explicit import/reload, with confirmation when the table has been edited. A failed load disables export rather than leaving the previous file silently exportable under a new name.

The table displays at most 50 data rows and 20 columns per page. Click any cell, including a header cell, to edit its complete text in a dialog. Display previews may be shortened, but stored values are not truncated. Row filtering uses literal case-insensitive text and retains source row indexes so edits target the correct row. Sort is deliberately text order, not inferred numeric/date order; account numbers and leading zeroes stay strings.

Trim data-cell whitespace, remove empty data rows, remove exact duplicate rows, sort ascending/descending by a selected column, or append an empty row. Transformations apply to the whole table, not only visible matches. The first row is preserved when “header” is enabled. Ragged rows are reported; editing a missing cell explicitly pads that record, and undo restores its previous shape. There is no formula engine, automatic type conversion, cloud sync, or arbitrary transformation code.

Undo retains at most 20 steps within a 20 MiB serialized-history budget. Reset restores the original imported table and is itself undoable within that budget. Closing/reloading the page loses unsaved edits; the UI warns after edits. Input limits: 5 MiB file, 200,000 cells, 50,000 rows, 2,000 columns, 5 MiB text-character budget, and 65,536 characters per cell edit. These limits intentionally avoid unbounded browser memory; large-file streaming is not implemented.

CSV exports default to the full table; exporting filtered matches requires an explicit checkbox. Header inclusion follows the header setting. Spreadsheet-formula neutralization is enabled by default and intentionally changes formula-like/leading-control values, including negative numbers. Disable it for exact trusted-data preservation. This mitigation is not a universal spreadsheet-security guarantee. JSON export preserves arrays, avoiding loss from duplicate/empty header keys. Exports use UTF-8 and never overwrite the original through application code.

## Task/session boundaries

Prompt/lyrics actions use two worker slots, up to eight queued/running tasks globally, four per session, and 64 retained records. Results/previews become unavailable ten minutes after finishing and are purged on subsequent API activity; this is not secure memory wiping. Requests and worker responses are bounded to 8 MiB. Worker processes have a three-minute wall-clock limit; Linux additionally applies 1.5 GiB address-space and 120 CPU-second limits. Windows/macOS do not have the same OS-enforced resource limits. This is trusted-local-code isolation, not a security sandbox.

Mutation approval uses the exact server-side preview, belongs to the signed-in session, and is single-use. Changing browser fields requires a new review. Clearing results/sign-out requests cancellation and removes completed content results. A stopped process does not reverse a completed copy export or retract text sent to a local model/provider. Status/results live in the specific tool page rather than the generic Jobs screen in this batch. Existing vault behavior is not modified.

## Validation

Local Linux execution: 59 Python tests passed and the Flask integration module was skipped because Flask/package installation was unavailable. Tests used a real loopback HTTP fixture server, actual bounded subprocesses, and five generated silent audio fixtures with genuine Mutagen read/write/read-back checks. No real music library, external provider, private model, paid API, or user browser profile was touched. Fifteen new Node CSV workbench tests passed; Python compilation and JS syntax checks passed.

The local browser smoke attempt was blocked by managed Chromium policy (`ERR_BLOCKED_BY_ADMINISTRATOR`) before page navigation. It is **not** reported as a passing UI check. The included test and a separate GitHub Actions job run in a normal development runner: CSV uses the real Web Worker; prompt/lyrics APIs are mocked. This does not test a live model/provider. The full-regression workflow installs Flask and runs the complete Python suite on Linux/Windows, Python 3.11/3.13. Read the PR's actual current check results; adding a workflow is not itself a pass. Independent security review, real-provider/model acceptance, macOS, and forced-stop/power-loss behavior remain outside these test claims.

Primary API references: https://github.com/LostRuins/koboldcpp/wiki ; https://lrclib.net/docs ; https://mutagen.readthedocs.io/en/latest/user/id3.html .

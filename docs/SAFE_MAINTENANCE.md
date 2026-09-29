# First three tools: safe-maintenance replacement

This change replaces Browser History Cleaner, Cookie Cleaner, and Lowercase Renamer from the merged workspace build (base commit `a1896e4535d99cff433eb18dff33da7b2bb15d7b`). Other tools and vault encryption are unchanged. The earlier AUDIT.md describes the old implementations; the status below supersedes its first-three-tool decisions.

## Removed, not merely hidden

The two browser scripts no longer open/copy/overwrite SQLite databases, infer registered domains from the last two labels, delete cookies by host/name alone, or terminate a browser. They are non-mutating compatibility notices. The old lowercase script is now a CLI adapter to the same preview/journal engine used by the web interface; invoking it without arguments does not rename anything.

## Browser tools: native local extension

Browser History Cleaner and Cookie Cleaner now use a bundled WebExtension. The Python workspace provides setup instructions and builds a ZIP from an explicit list of public extension files. It does not receive history records, cookies, cookie values, backup passwords, or extension messages. There is no content script, background cleanup, native messaging, externally-connectable bridge, telemetry, or fetch call. CSP prohibits network connections. Local storage holds only a profile identifier and your explicit protected-host list; it is not sync storage.

This is an intentional architecture change, not a transparent automatic upgrade: install the extension in **each browser profile** to maintain. The extension acts only within its current profile and selected cookie store. There is no unreliable automatic profile-directory discovery or hard-coded Windows Default profile. Workspace cards say **Needs setup** because OmniTool cannot detect installation.

### Install

Start OmniTool normally (`start.bat` or `python app.py`), open either browser tool, and download the appropriate package. Extract it to a permanent local folder.

For Chrome, Edge, or Brave, open that browser's extension-management page, enable Developer mode, and choose **Load unpacked** with the extracted folder. The Chromium manifest requires version 130 or newer. Open the extension's toolbar popup and choose its workspace link.

For Firefox, the generated package requires Firefox 145 or newer. `about:debugging` > This Firefox > Load Temporary Add-on > select the extracted `manifest.json` provides a test installation. Temporary add-ons disappear when Firefox restarts. A persistent ordinary Firefox installation requires Mozilla signing; this change does not submit or sign the extension. Do not assume a backup's profile identifier will survive removing and reinstalling a temporary add-on. Keep the extension installed and complete any restore before removal. Sideloading may be forbidden by a managed browser policy; do not bypass that policy.

The source manifest is the Chromium variant. Download the generated Firefox variant rather than manually loading the Chromium manifest into Firefox. No store listing or automatic update mechanism is included.

### History workflow

Grant History access through the button, enter a hostname (not a URL), and choose whether to include its subdomains. Matching uses the parsed URL hostname and a dot-boundary suffix check, never arbitrary URL/title text. IDNA hostnames are normalized. Only HTTP(S) records are supported. IPv6 literals and non-web history entries are outside this implementation.

Preview selects nothing automatically and shows individual URLs. The search is capped at 20,000 URLs; reaching the cap displays an incomplete-results warning. At most 500 matching URLs can be processed per operation. Selecting a URL removes **all its visits**, not a date range or selected visits. Browser-native deletion handles internal relationships; the extension has no bookmarks permission and does not delete bookmark records.

### Cookie workflow

Grant cookie/HTTP(S) site access, explicitly select a cookie store, and enter protected hostnames one per line. Hostnames are exact by default; a separate checkbox permits their subdomains. **There is no guessed eTLD+1 grouping:** `example.co.uk` does not become `co.uk`, and `tenant.github.io` does not become `github.io`. This deliberately avoids a downloaded or stale Public Suffix List. It also means entering a public suffix with subdomains enabled protects that entire suffix; inspect the exact rules you enter.

Existing plaintext allowlists are not auto-imported. Copy only the intended hostnames into the new local allowlist after review. History/bookmarks are not silently added as trusted sites.

Preview displays host, name, path, store and partition identifiers but never cookie values. Cookies are differentiated by store, domain/hostOnly, name, path, first-party domain and partition key, including supported cross-site-ancestor metadata. The native API can select a shadowing cookie with the same name: identity/signature checks refuse that case instead of deleting a broader host/name group. Unsupported cookie paths or partition selectors fail rather than retrying with isolation removed. At most 500 unprotected cookies can appear in one operation; extend the allowlist or use browser settings for larger operations.

### Mandatory verified backup, then explicit apply

1. Select records and choose a strong generated passphrase (minimum 16 characters).
2. Save the encrypted `.otbrowser` download. Nothing has been deleted.
3. Reopen that saved file and verify its passphrase. The decrypted backup must match the exact current prepared selection.
4. Type `DELETE N`, where N is the selection size. The tool rechecks each record and reports removed, changed/skipped, recreated, or failed results. The prepared selection is invalidated afterwards.

Changing selection or filters invalidates backup approval. Stop cancels between native API calls; completed changes remain applied. Individual calls are not a multi-record transaction. Site tabs, workers and browser sync may race the final check/delete or restore calls and recreate records. Close affected site tabs and pause activity before maintenance. This is not forensic erasure, cloud-account-history deletion, or a guarantee that no new record can change during execution.

### Backup encryption and restore limits

Browser backups use a separate versioned JSON envelope: WebCrypto AES-256-GCM, fresh 16-byte salt and 12-byte nonce, PBKDF2-HMAC-SHA256 with 600,000 iterations, a 256-bit key, and fixed authenticated additional data. Ciphertext includes the selected cookie values or history visits plus all record metadata. Passwords are prompted locally, are not arguments or stored credentials, and input fields are cleared after use. Clearing references is not secure memory zeroization. The format is **not `.otvault`**, is not independently audited, and provides no password recovery. A weak password remains vulnerable to offline guesses. Envelope limits are 8 MiB plaintext and 12 MiB encoded file; sizes are not padded.

Restore requires reopening/decrypting the backup, inspecting its metadata, typing `RESTORE N`, and using the same extension profile identity/browser engine. This prevents accidental cross-profile restoration but is not a cryptographic user identity: someone with the passphrase can decrypt the backup independently. Removing extension storage can prevent in-app restore. Keep the profile/extension installation until restoration is no longer needed.

Cookie restore skips existing matching identities and expired cookies; it does not intentionally overwrite current cookies. Browser validation, expiry changes, partition/container availability and races may prevent an exact restoration. Reinstating a cookie does not reinstate an expired or revoked server-side session. History restore adds missing URLs at the current time through the portable API, **not original visit times/counts/transitions**; existing URLs are skipped. Original visit metadata remains in the encrypted backup for inspection. There is no claim of lossless history rollback.

Treat the backup as sensitive even though encrypted. Do not commit it, plaintext exports, cookie values, or passwords to Git. The extension cannot protect an unlocked browser from malware, extensions with equivalent permissions, a debugger, or the local administrator. Test with disposable records before normal use.

## Lowercase Renamer: preview, apply, recovery

Open the tool inside the shared OmniTool workspace. Enter a specific working-folder path. Preview is read-only; the root/home directory is refused. Directory names and file contents are not changed. The recursive walk skips symlinks, Windows reparse points/junctions, special files, nonportable path components, and known development/runtime directories. The operation is bounded to 20,000 inspected entries and 500 renames. The preview lists skipped entries and blocking case/Unicode-normalization collisions; collisions must be resolved before anything is renamed.

Type `RENAME N` to apply the reviewed snapshot. A changed snapshot is rejected rather than silently applying a new plan. The engine acquires an OS-held operation lock, writes a journal before every move, stages files to random sibling names, then commits lowercase names. Native no-overwrite primitives are used: Linux `renameat2(RENAME_NOREPLACE)`, Windows non-replacing `os.rename`, and macOS `renamex_np(RENAME_EXCL)`. Unsupported OS/filesystems fail closed; there is no overwrite fallback.

Ordinary failures attempt rollback. Abrupt interruption leaves a write-ahead journal that can locate staged/committed files. The operation list offers **Undo / recover** with explicit `UNDO` confirmation. All file identities and occupied original names are checked before recovery changes any name. Changed files, occupied destinations, ambiguous locations or malformed journals stop recovery; inspect and resolve them rather than forcing an overwrite. An unfinished journal blocks new apply operations. Completed operation IDs remain recoverable through the CLI even if outside the UI's latest-100 list.

Journals are outside the repository by default: `%LOCALAPPDATA%\OmniTool\renames` on Windows; `$XDG_STATE_HOME/OmniTool/renames` or `~/.local/state/OmniTool/renames` on Linux; `~/Library/Application Support/OmniTool/renames` on macOS. They contain plaintext filenames and file identities, not copies of file contents. Preserve them until undo is no longer needed, and protect them with OS access controls. The app is single-user: all authenticated local sessions can view/recover that OS user's journals; preview tokens are session-specific, ten-minute, and single-use.

Stop other programs writing into the target directory. This is not a sandbox against a malicious concurrent same-user process, a content backup, or a guarantee against every filesystem/power-loss failure. Independent data backups remain necessary. Windows/macOS behavior and network filesystems need acceptance testing; only Linux filesystem behavior was exercised here.

CLI (from the repository root):

```sh
python scripts/lowercase_renamer.py preview --root /path/to/working-folder
python scripts/lowercase_renamer.py apply --root /path/to/working-folder --fingerprint PREVIEW_FINGERPRINT --confirm RENAME
python scripts/lowercase_renamer.py undo --operation OPERATION_ID --confirm UNDO
```

The full app factory is `app.create_app`; its wrapper registers the maintenance routes on the existing core factory. Start with the normal entrypoint rather than invoking the core-only factory directly.

## Validation actually performed

Linux, Python 3.13.5 and Node 22.16.0; disposable fixtures only.

- `python -m pytest -q tests/test_rename.py tests/test_browser_package.py tests/test_maintenance_web.py`: **30 passed, 1 module skipped**. Flask HTTP integration is skipped because Flask is unavailable in the execution environment. No package installation was available.
- `node --test tests/browser_maintenance.test.js`: **14 tests passed**. Native APIs are mocked; WebCrypto encryption/decryption is real Node WebCrypto.
- Python compilation and JavaScript syntax checks passed.
- Both the native extension smoke test and the mocked-API browser UI smoke test were attempted but **blocked by managed Chromium policy** (`ERR_BLOCKED_BY_CLIENT` / `ERR_BLOCKED_BY_ADMINISTRATOR`). Neither is a passing result. Their test scripts are included for an unrestricted development/test environment, not as a way to bypass policy.
- No actual user browser profiles, browser databases, cookies, history, or real user file trees were modified. The original full-repository suite was not run in this sparse environment. Windows/macOS acceptance and Firefox native API tests remain outstanding.

Before normal use, install `requirements-dev.txt` and run the HTTP tests; run the native Chromium smoke in a disposable profile; inspect optional permission grants, partitioned cookies, bookmark retention, backup read-back and recovery in the actual supported browsers. Native tests grant fixture-only extra permissions in a temporary copy; production packaging never includes them.

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
node --test tests/browser_maintenance.test.js
python -m pip install playwright
python -m playwright install chromium
python tests/browser_ui_smoke.py
python tests/browser_native_smoke.py
```

## Primary references reviewed

- Chrome History API: https://developer.chrome.com/docs/extensions/reference/api/history
- Chrome Cookies API (selection and partitions): https://developer.chrome.com/docs/extensions/reference/api/cookies
- MDN cookies.getAll: https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/cookies/getAll
- MDN cookies.remove: https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/cookies/remove
- MDN cookies.set: https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/cookies/set
- MDN history.addUrl: https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/history/addUrl
- WebCrypto deriveKey: https://developer.mozilla.org/en-US/docs/Web/API/SubtleCrypto/deriveKey
- Chromium unpacked installation: https://developer.chrome.com/docs/extensions/get-started/tutorial/hello-world
- Firefox temporary installation: https://extensionworkshop.com/documentation/develop/temporary-installation-in-firefox/
- Firefox manifest/signing fields: https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/manifest.json/browser_specific_settings
- Python rename semantics: https://docs.python.org/3/library/os.html#os.rename
- Linux no-replace primitive: https://man7.org/linux/man-pages/man2/rename.2.html

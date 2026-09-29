# Windows startup and stabilization

Windows is the primary target. Use **64-bit Python 3.11 or newer**; Python 3.13 x64 is preferred and tested alongside 3.11 in CI. The launcher reuses the existing `.venv`; when creating one it first tries `py -3.13`, then a compatible `py -3`, then `python`. It never deletes or silently replaces an existing virtual environment.

## Start and diagnose

Double-click `start.bat`. Missing or incompatible core packages are installed/repaired using `requirements-core.txt`. Unlike the previous import-only check, installed package versions must satisfy the declared minimum and maximum. Optional tools are not installed automatically. An invalid Python environment or a failed install leaves an explanatory console message visible rather than closing the window immediately.

The local server binds only to `127.0.0.1`, using an exclusive Windows socket before handing it to Waitress. It does not enable Windows port sharing. The browser opens **after `/login` answers successfully**, not before the server starts. Enter the access code printed in the terminal; the code is not put into a URL, file, or clipboard. Authentication, CSRF/origin checks, and vault passphrases are unchanged.

Run these commands from a terminal in the project directory:

```bat
start.bat --diagnose
start.bat --no-browser
```

`--diagnose` is read-only: it reports Python architecture, core/optional package requirements, and whether FFmpeg is on PATH. It neither installs packages nor contacts a model or lyrics provider. An optional package being installed is not a live provider health check. A missing `.venv` is reported rather than created in diagnose mode. The report contains no environment-variable dump or access token.

For optional modules, choose the relevant requirement file deliberately:

```bat
.venv\Scripts\python.exe -m pip install -r requirements-content.txt
.venv\Scripts\python.exe -m pip install -r requirements-conversion.txt
```

When port 5000 is already occupied, OmniTool explains the conflict and does **not** kill another process, open the wrong service, or silently switch ports. Close an earlier instance or choose a port explicitly in the current terminal:

```bat
set OMNITOOL_PORT=5010
start.bat
```

PowerShell equivalent:

```powershell
$env:OMNITOOL_PORT = "5010"
.\start.bat
```

The Python entrypoint also accepts `--port 5010 --open-browser`. Direct `python app.py` continues to work without opening a browser automatically. Ports must be between 1024 and 65535. Changing the port does not enable remote access. This remains a single-user workspace, not a multi-instance session manager.

Press Ctrl+C in the server terminal to stop it. The canonical entrypoint invokes the full workspace shutdown callback before interpreter exit, canceling managed worker tasks. Forced termination/power loss still cannot guarantee temporary-file cleanup or cancellation inside an external model server. Existing file/vault recovery limitations still apply.

`OMNITOOL_NO_PAUSE=1` suppresses failure prompts for automated tests or scripted launch. It does not bypass any security or prerequisite check. `start.bat` changes environment variables only inside its own `setlocal` scope; no global Windows configuration is changed.

## Repairs in this batch

### Duplicate Finder

An unchanged file could be rejected on Windows because path-stat timestamps were compared directly with descriptor-stat timestamps. Both representations are now checked independently before and after hashing, and file identity is cross-checked. Files changed/replaced/grown during the operation are still rejected. Links, junctions, and other reparse points are not traversed. Library callers receive the same argument validation as the CLI. The tool remains read-only; matching hard-linked paths do not imply recoverable disk space.

### Python job output

The generic Python launcher explicitly sets UTF-8 for redirected stdout/stderr rather than decoding legacy Windows bytes as UTF-8. Incremental decoding preserves characters split across individual pipe reads. Tests cover accents, CJK characters, invalid bytes, and a parent process configured for ASCII. No workspace access token is passed to child tools. Other inherited tool credentials and the trusted-code model remain unchanged.

### Regression reporting

Large parameter payloads now receive compact hashed test IDs. Their full data is still tested, but it is not copied into verbose logs or `PYTEST_CURRENT_TEST`. This resolves the oversized diagnostic path that obscured the previous failures. No existing failing test is disabled.

## Verification

```bat
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -u tests\run_ci.py
node tests\csv.test.js
node --test tests\csv_workbench.test.js tests\browser_maintenance.test.js
```

The full workflow runs on Windows and Linux with Python 3.11 and 3.13, and now runs on merges to main as well as pull requests. Browser checks run on **Windows as well as Linux**. Superseded runs of this workflow on the same ref are canceled to avoid accumulated stale checks.

Native Windows launcher tests use an isolated copy of the checkout under a path with spaces, parentheses, accents, and CJK characters and a disposable virtual environment. They exercise `cmd.exe`, read-only diagnostics, the actual Waitress login endpoint, denial of unauthenticated API access, an occupied port, and rejection of a second socket attempting to reuse the workspace port. The test virtual environment inherits CI's already installed packages; these tests do not verify a first-time installation from PyPI or test an actual Explorer double-click. Test process cleanup targets only the subprocess tree created by the fixture.

Browser smoke exercises the real CSV worker and rendered templates; Prompt/Lyrics API responses are mocked. No real browser profile, music library, model, public provider, or private source is used. Tests are not a security audit or a guarantee that arbitrary legacy download tools and external providers work. Read the latest PR checks for actual results rather than assuming a workflow definition means a passing run.


Socket binding follows Microsoft Winsock guidance for `SO_EXCLUSIVEADDRUSE` and Waitress's documented pre-bound `sockets` interface. A recently closed connection can temporarily keep the port unavailable; select a different explicit port rather than terminating unrelated processes.

- https://learn.microsoft.com/en-us/windows/win32/winsock/using-so-reuseaddr-and-so-exclusiveaddruse
- https://docs.pylonsproject.org/projects/waitress/en/stable/arguments.html

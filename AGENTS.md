# Project priorities

OmniTool is primarily a **Windows desktop/local-host project**. Windows behavior is a release requirement, not an optional portability follow-up. Keep `start.bat` working from Explorer, paths containing spaces and non-ASCII characters, Windows process cancellation, and no-overwrite file operations covered by tests. Linux tests are supplemental; do not claim Windows success from Linux alone.

Fix failing or stalled checks and reproducible runtime bugs before expanding tools or restyling the interface. Run the full Windows Python matrix and Windows browser checks before calling a change verified. Keep the existing authentication, CSRF/origin restrictions, encrypted-bundle opacity, preview/confirmation and original-file protections intact. Never use real browser profiles, music libraries or private source as test fixtures.

Use bounded, synthetic fixtures. Large bytes/strings must not become test IDs: they leak into logs and PYTEST_CURRENT_TEST and can exceed Windows environment-variable limits. Do not disable failing tests to obtain green checks; skip only an explicitly unavailable platform capability with a reason.

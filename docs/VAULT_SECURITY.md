# Encrypted bundles: security model and limits

**Review build, not independently audited. Do not treat a successful unit test as a cryptographic or application-security audit.** The new `.otvault` container uses existing cryptographic primitives; it is not the age format or a standardized encrypted filesystem.

## Intended protection

A public repository can distribute ciphertext without distributing the plaintext source or its internal filenames. The attacker may clone the repository, read every historical ciphertext, inspect this implementation, modify a ciphertext, and make unlimited offline password guesses. Protection requires a high-entropy secret passphrase and an uncompromised machine when unlocking. Whoever obtains the passphrase and the ciphertext can decrypt; this is not tied to a GitHub identity.

Each bundle encrypts its display name, tool/folder kind, tool manifests, dependency names, parameters, internal paths, file contents, and archive metadata together. While locked, the local API exposes only an opaque ID and locked state. A folder containing many tools is one blob, so its exact child count is not disclosed in a public registry. Existing public tool names remain public; locking new code does not reclassify them.

## Construction

Version 1 header: `OMNIVAULT` plus version byte `01`, a fresh random 16-byte salt, and a fresh random 12-byte nonce. The entire header is authenticated additional data. PyCA scrypt derives a 32-byte key using fixed `N=2^17, r=8, p=1` (approximately 128 MiB working memory). PyCA AES-256-GCM encrypts and authenticates an eight-byte archive length followed by archive bytes and padding. The GCM tag is included by the library.

Ciphertext payload length is padded to a power of two, with a 64 KiB minimum. Padding reduces exact-size disclosure but does not hide coarse size or update patterns. Files inside the archive use uncompressed regular-file entries, fixed ZIP timestamps, and fixed permission metadata before encryption. Seal always generates a new salt and nonce and verifies an encryption/decryption round trip before writing. No key, passphrase, password hash, cleartext manifest, or private-name sidecar is stored in the repository.

The decryptor verifies envelope bounds and version before invoking the KDF, authenticates before parsing, then validates the complete archive before extraction. It rejects traversal, absolute/nonportable paths, symlinks, special files, duplicate/case-colliding names, file/directory conflicts, unexpected compression, and size/member limits. Fixed KDF parameters cannot be raised through an attacker-controlled header. Passphrases must be at least 16 characters when sealing, but length alone is not evidence of sufficient entropy: use a password-manager-generated secret or a long randomly generated word sequence.

## Local application boundary

The app binds only to loopback, requires a random local access code, checks host/origin and CSRF, uses no third-party scripts, marks responses no-store, and keeps secrets out of browser storage. The local sign-in code is distinct from the encryption passphrase. Unlocking is explicit, session-scoped, and fixed-duration; polling does not extend its ten-minute lease. Decrypted metadata and file bytes stay in process memory until a tool executes or a file is deliberately downloaded. No persistent decrypted cache is created by the workspace.

A private execution creates a temporary directory outside the Git checkout, writes the validated bundle there, starts Python without bytecode caching, and redirects conventional temporary-directory variables into that directory. Child stdin is disabled, argv is used without a shell, and captured output is bounded and memory-only. Private job labels are opaque. Locking/expiry deny new private reads and runs immediately, then request process cancellation and cleanup. Job output is no longer returned while the associated bundle is locked. A cleanup failure is reported rather than represented as successful erasure.

## What this does NOT guarantee

- Public observers still see encrypted-file existence, count, padded sizes, commit times, and which opaque blob changed. Public commit messages, issues, branch names, screenshots, or explanatory filenames can separately reveal sensitive names. Do not put them there. This is encryption, not steganography or plausible deniability.
- Public ciphertext permits unlimited offline guesses. UI rate limiting does not prevent offline cracking. There is no recovery backdoor and no way to revoke already copied ciphertext. Rotating a passphrase only protects newly sealed content; old Git versions retain their old protection.
- Already published plaintext remains in Git history, forks, clones, caches, screenshots, or other copies. Deleting the current file, making a repository private, or encrypting its current version does not recall those copies. History rewriting has operational costs and cannot guarantee recall. It is not performed by this change. Any exposed credential must be rotated, not merely hidden.
- The local machine must see plaintext to execute it. Malware, an administrator, a debugger, same-user processes, browser extensions, swap/hibernation, crash dumps, backups, and antivirus indexing are outside this protection. Python and JavaScript do not provide dependable secure zeroization of every copied string/byte buffer. `del` drops references; it does not certify memory wiping. Use full-disk encryption and appropriate OS access controls.
- Private tools are trusted code, not sandboxed code. They can deliberately create other plaintext files, access the network, inherit ordinary tool credentials from the environment, or spawn detached processes. The workspace cannot retroactively retract anything they disclose. Parent/process-group cancellation is best effort, particularly for detached children and Windows process-tree races; an OS-enforced worker sandbox/Windows Job Objects remain future work.
- Temporary-file deletion is ordinary cleanup, not secure erasure. Power loss, force-killing the workspace, or a process holding a file open can leave plaintext behind. Windows access control must be checked on the actual host; POSIX mode bits do not translate into an audited Windows ACL policy. Test crash/cleanup behavior with disposable data before use. Downloaded/exported files and tool output written outside the temporary folder remain plaintext after locking.
- Dependencies installed globally or in a visible local environment may reveal capabilities to an observer of that machine. This does not make the public encrypted manifest visible, but it is not full-system opacity.
- Confidentiality and GCM authentication do not establish trusted authorship against someone who already knows the passphrase, prevent malicious public launcher updates, or prevent replay of an older valid ciphertext. Review changes to the launcher before entering a passphrase. Reproducible releases/signatures and anti-rollback policies are not implemented.

## Validation and release gate

Core round trips, tampering, wrong passphrases, archive paths, session scope, expiry, and private subprocess cleanup have automated tests. Full Flask HTTP integration and real Windows behavior were not executed in the implementation environment; see TEST_REPORT.md. Before relying on this for valuable private source, run the suite with all dependencies, conduct a Windows acceptance pass, review the container and request/session boundary independently, and test backup recovery.

Primary references:
- PyCA AEAD documentation: https://cryptography.io/en/latest/hazmat/primitives/aead/
- PyCA KDF documentation: https://cryptography.io/en/latest/hazmat/primitives/key-derivation-functions/
- Flask security guidance: https://flask.palletsprojects.com/en/stable/web-security/
- GitHub guidance on removing sensitive data: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository

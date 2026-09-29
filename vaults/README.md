# Ciphertext only

Only randomly named `.otvault` bundles belong here. Generate them with the local sealing CLI described in `docs/ADDING_TOOLS.md`. Never place plaintext source, keys, passphrases, identifying sidecar files, or decrypted archives in this directory. No real private source is included by this migration.

The public can see blob existence, padded size, and Git history. Encryption does not erase earlier public commits. Read `docs/VAULT_SECURITY.md` before use.

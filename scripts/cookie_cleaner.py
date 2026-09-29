#!/usr/bin/env python3
"""Compatibility notice. Direct cookie-database mutation has been removed."""

def main() -> None:
    print('Cookie Cleaner now uses the OmniTool Browser Maintenance extension.')
    print('Open the tool in OmniTool for installation: http://127.0.0.1:5000/maintenance/browser/cookies')
    print('No cookies were read or changed. Protected hosts are now stored in the extension profile.')
    print('Existing cookies_whitelist.txt files are not imported automatically. See docs/SAFE_MAINTENANCE.md.')

if __name__ == '__main__':
    main()

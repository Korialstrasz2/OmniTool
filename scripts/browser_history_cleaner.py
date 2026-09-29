#!/usr/bin/env python3
"""Compatibility notice. Direct history-database mutation has been removed."""

def main() -> None:
    print('Browser History Cleaner now uses the OmniTool Browser Maintenance extension.')
    print('Open the tool in OmniTool for installation: http://127.0.0.1:5000/maintenance/browser/history')
    print('No browser database was read, copied, or modified. See docs/SAFE_MAINTENANCE.md.')

if __name__ == '__main__':
    main()

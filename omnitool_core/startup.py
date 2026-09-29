"""Windows-first local startup; no browser is opened before HTTP is serving."""
from __future__ import annotations

import argparse
import errno
import http.client
import os
import socket
import threading
import time
import webbrowser


class StartupError(RuntimeError):
    pass


def port_number(value: str | int) -> int:
    if isinstance(value, bool) or not str(value).isascii() or not str(value).isdecimal():
        raise ValueError('Port must be a number from 1024 to 65535')
    number = int(value)
    if not 1024 <= number <= 65535:
        raise ValueError('Port must be a number from 1024 to 65535')
    return number


def open_when_ready(port: int, stopped: threading.Event, opener=None, timeout: float = 15) -> bool:
    """Probe only our bound loopback server. No proxies, redirects, or credentials."""
    opener = opener or webbrowser.open
    deadline = time.monotonic() + timeout
    while not stopped.is_set() and time.monotonic() < deadline:
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=.5)
        try:
            connection.request('GET', '/login')
            response = connection.getresponse()
            ready = response.status == 200
        except (OSError, http.client.HTTPException):
            ready = False
        finally:
            connection.close()
        if ready and not stopped.is_set():
            try:
                if opener(f'http://127.0.0.1:{port}/'):
                    return True
            except Exception:
                pass
            print(f'Browser could not be opened. Open http://127.0.0.1:{port}/ manually.', flush=True)
            return False
        stopped.wait(.1)
    if not stopped.is_set():
        print('Browser was not opened because the login page did not become ready. Check the terminal above.', flush=True)
    return False


def bound_listener(port: int) -> socket.socket:
    """Own the loopback port exclusively on Windows, before handing it to Waitress."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if os.name == 'nt':
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', port))
        return listener
    except BaseException:
        listener.close()
        raise


def serve_local(application, *, port: int = 5000, open_browser: bool = False, server_factory=None):
    """Bind first, then start a readiness probe. Port zero is for internal tests only."""
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError('Invalid port')
    if server_factory is None:
        from waitress import create_server
        server_factory = create_server
    stopped = threading.Event()
    browser_thread = None
    server = None
    listener = None
    socket_map = {}
    try:
        try:
            listener = bound_listener(port)
            server = server_factory(application, sockets=[listener], threads=6,
                                    max_request_body_size=64 * 1024, map=socket_map)
        except OSError as exc:
            if exc.errno in {errno.EADDRINUSE, errno.EACCES} or getattr(exc, 'winerror', None) in {10048, 10013}:
                raise StartupError(f'Port {port} is occupied or unavailable. Close the other OmniTool instance, '
                                   'or set OMNITOOL_PORT to a different port. No existing process was stopped.') from None
            raise StartupError('Could not bind the local server. Check Windows networking permissions.') from exc
        actual_port = int(server.effective_port)
        print(f'OmniTool: http://127.0.0.1:{actual_port}', flush=True)
        print('Local access code (not your vault passphrase): ' + application.extensions['omnitool_access_token'], flush=True)
        print('Keep this terminal private. Ctrl+C stops the workspace and its managed jobs.', flush=True)
        if open_browser:
            browser_thread = threading.Thread(target=open_when_ready, args=(actual_port, stopped),
                                              name='omnitool-browser-ready', daemon=True)
            browser_thread.start()
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        stopped.set()
        if server is not None:
            server.close()
            server.task_dispatcher.shutdown()
        # This map belongs solely to this startup; never close unrelated sockets.
        for channel in list(socket_map.values()):
            channel.close()
        if listener is not None:
            listener.close()
        if browser_thread is not None:
            browser_thread.join(timeout=1)


def main(application, argv=None) -> int:
    parser = argparse.ArgumentParser(description='Start the local OmniTool workspace')
    parser.add_argument('--port', default=os.environ.get('OMNITOOL_PORT', '5000'))
    browser = parser.add_mutually_exclusive_group()
    browser.add_argument('--open-browser', action='store_true')
    browser.add_argument('--no-browser', action='store_true')
    try:
        args = parser.parse_args(argv)
        serve_local(application, port=port_number(args.port), open_browser=args.open_browser)
        return 0
    except (StartupError, ValueError) as exc:
        print(f'Unable to start OmniTool: {exc}', flush=True)
        return 1
    finally:
        # Stop ThreadPool workers before interpreter shutdown waits for them.
        application.extensions['omnitool_shutdown']()

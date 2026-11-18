"""Start a local Streamlit server and open the browser after it is ready."""
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser


def main():
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    logs = root / 'outputs'
    logs.mkdir(exist_ok=True)
    logfile = logs / f'launcher_{port}.log'
    print('Starting DQAOA... Please keep this window open while using the app.')
    with logfile.open('w', encoding='utf-8') as log:
        process = subprocess.Popen([
            sys.executable, '-m', 'streamlit', 'run', str(root / 'app.py'),
            '--server.address=127.0.0.1', f'--server.port={port}',
            '--server.headless=true', '--browser.gatherUsageStats=false',
        ], cwd=root, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError(f'App stopped during startup. See {logfile}')
                try:
                    with urllib.request.urlopen(url + '/_stcore/health', timeout=1) as reply:
                        ready = reply.status == 200
                except (OSError, TimeoutError):
                    ready = False
                if ready:
                    webbrowser.open(url)
                    print(f'App ready: {url}')
                    print('To stop the app, return here and press Ctrl+C.')
                    return process.wait()
                time.sleep(0.5)
            raise RuntimeError(f'App startup timed out. See {logfile}')
        except KeyboardInterrupt:
            return 0
        except Exception as error:
            print(error)
            return 1
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == '__main__':
    raise SystemExit(main())

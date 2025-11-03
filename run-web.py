#!/usr/bin/env python3
"""
run-web.py — Launch the dockerized visualizer and stop/remove everything on exit.

Usage:
    python run-web.py
    python run-web.py --no-open         # do not automatically open the browser
    python run-web.py --rm-image        # remove the built Docker image when finished
    python run-web.py --verbose         # enable verbose terminal output (default is quiet)
"""
import sys, os, subprocess, time, signal, atexit, webbrowser, shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
COMPOSE_FILE = PROJECT_DIR / "web" / "docker-compose.yml"
IMAGE_NAME = "nosql-visualizer:latest"
URL = "http://localhost:8088"

VERBOSE = "--verbose" in sys.argv

def info(msg):
    print(f"[run-web] {msg}")

def run(cmd, check=True):
    """Run a subprocess. When --quiet is set, capture output and only show it on error."""
    info("$ " + " ".join(cmd))
    if not VERBOSE:
        # Default quiet mode: capture subprocess output and only show it on error.
        # Force UTF-8 decoding and replace invalid characters so Windows cp1252 errors don't crash.
        res = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if check and res.returncode != 0:
            # Print captured output so the user can see the failure
            if res.stdout:
                sys.stdout.write(res.stdout)
            if res.stderr:
                sys.stderr.write(res.stderr)
            raise subprocess.CalledProcessError(res.returncode, cmd, output=res.stdout, stderr=res.stderr)
        return res
    else:
        # Verbose mode: stream subprocess output directly but still force the encoding to utf-8
        # and replace invalid characters to avoid decode errors on Windows.
        return subprocess.run(cmd, check=check, text=True, encoding="utf-8", errors="replace")

def compose_up():
    run(["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d", "--build"])

def wait_ready(timeout=120):
    import urllib.request
    import urllib.error
    info("Waiting for the visualizer to become ready...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(URL + "/healthz", timeout=2) as r:
                if r.status == 200:
                    info("✓ Ready.")
                    return True
        except Exception:
            pass
        time.sleep(1)
    info("WARNING: did not respond in time; continuing anyway.")
    return False

def open_browser(no_open=False):
    if no_open: return
    try:
        webbrowser.open(URL)
        info(f"Opened browser at {URL}")
    except Exception:
        info(f"Open {URL} manually")

def cleanup(rm_image=False):
    info("Cleanup...")
    try:
        run(["docker", "compose", "-f", str(COMPOSE_FILE), "down", "--remove-orphans"], check=False)
    except Exception:
        pass
    if rm_image:
        try:
            run(["docker", "rmi", "-f", IMAGE_NAME], check=False)
        except Exception:
            pass
    info("Cleanup completed.")

def main():
    rm_image = "--rm-image" in sys.argv
    no_open = "--no-open" in sys.argv

    # Start services
    compose_up()
    # Wait until the service is ready
    wait_ready()
    # Open browser (unless requested not to)
    open_browser(no_open)

    # Handle signals and block until Ctrl+C
    def _sig(_s, _f):
        sys.exit(0)
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    atexit.register(lambda: cleanup(rm_image))

    info("Running. Press Ctrl+C to stop and clean up.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()

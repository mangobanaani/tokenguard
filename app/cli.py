import os
import sys
import subprocess


def run():
    """Start the API server with uvicorn."""
    host = os.getenv("HOST", "0.0.0.0")
    port = os.getenv("PORT", "8000")
    args = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        host,
        "--port",
        str(port),
        "--reload",
    ]
    os.execvp(args[0], args)


def test():
    """Run the test suite with pytest."""
    cmd = [sys.executable, "-m", "pytest"]
    sys.exit(subprocess.call(cmd))

import os
import sys
import subprocess
from app.constants import DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT, DEFAULT_RELOAD


def run():
    """Start the API server with uvicorn."""
    host = os.getenv("HOST", DEFAULT_SERVER_HOST)
    port = int(os.getenv("PORT", str(DEFAULT_SERVER_PORT)))
    reload = os.getenv("RELOAD", str(DEFAULT_RELOAD)).lower() in ("1", "true", "yes", "on")
    
    args = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    
    if reload:
        args.append("--reload")
    
    os.execvp(args[0], args)


def test():
    """Run the test suite with pytest."""
    cmd = [sys.executable, "-m", "pytest"]
    sys.exit(subprocess.call(cmd))

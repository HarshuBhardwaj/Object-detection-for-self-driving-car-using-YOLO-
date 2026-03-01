# Simple launcher that runs imp.py with stderr suppressed
import subprocess
import sys
import os

# Run the main app with stderr redirected to NUL on Windows
if sys.platform == 'win32':
    # Redirect stderr to NUL on Windows
    with open(os.devnull, 'w') as devnull:
        result = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), 'imp.py')],
            stderr=devnull,
            stdout=sys.stdout
        )
else:
    # On Unix, redirect to /dev/null
    with open('/dev/null', 'w') as devnull:
        result = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), 'imp.py')],
            stderr=devnull,
            stdout=sys.stdout
        )

"""Build the C/OpenMP backend: python build_c.py [--no-openmp]."""
import argparse
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-openmp', action='store_true', help='Build a serial-only library')
    parser.add_argument('--sanitize', action='store_true', help='Enable undefined-behavior sanitizer')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output or root/'_sampen_c.so'
    flags = ['-O3', '-std=c11', '-fPIC', '-shared', '-Wall', '-Wextra', '-Werror']
    if not args.no_openmp:
        flags.append('-fopenmp')
    if args.sanitize:
        flags += ['-fsanitize=undefined', '-fno-sanitize-recover=undefined']
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement avoids corrupting a library mapped by another process.
    with tempfile.NamedTemporaryFile(prefix=f'.{output.name}.', dir=output.parent, delete=False) as f:
        temporary = Path(f.name)
    command = shlex.split(os.environ.get('CC', 'gcc')) + flags + [str(root/'csrc/sampen.c'), '-lm', '-o', str(temporary)]
    print(shlex.join(command), flush=True)
    try:
        subprocess.run(command, check=True)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

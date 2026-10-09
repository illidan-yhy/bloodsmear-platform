"""Interpreter checks without inline code/PowerShell command quoting."""
import argparse
import struct
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('executable', 'version', 'validate'))
    mode = parser.parse_args().mode
    if mode == 'executable':
        print(sys.executable)
    elif mode == 'version':
        print('{}.{}'.format(sys.version_info[0], sys.version_info[1]))
    else:
        if sys.version_info[:2] != (3, 11) or struct.calcsize('P') != 8:
            print('Python 3.11 x64 is required.')
            return 1
        print('Python 3.11 x64 preflight passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())

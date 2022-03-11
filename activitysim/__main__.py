import sys
import os

def main():
    # clean up message formatting
    print(sys.argv)
    if sys.argv and sys.argv[0].endswith('__main__.py'):
        sys.argv[0] = 'activitysim'
    print(sys.argv)

    # threadstopper
    # this gets run after importing sys and os but before any other libraries
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMBA_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    from .cli.main import main
    main()

if __name__ == '__main__':
    main()
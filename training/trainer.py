import sys

backend = 'tensorflow'
num_args = len(sys.argv)

if len(sys.argv) > 1:
    backend = sys.argv[1]
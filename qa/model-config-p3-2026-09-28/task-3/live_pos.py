import sys
needle = sys.argv[1]
for n, line in enumerate(sys.stdin.read().split("\n"), 1):
    i = line.find(needle)
    if i >= 0:
        print(n, i + 1)
        break

#! /usr/bin/env python3

import os
import sys

def gitList(lst):
    ADD_TAG = ["modified", "new"]
    lst = [a.strip().split(":") for a in lst]
    toAdd, toRm = [], []
    for a in lst:
        if a[0].strip() in ADD_TAG:
            toAdd.append(a[1].strip())
    
    return toAdd, toRm


def parse(lines):
    """Parse text, e.g
            modified:   src/illustrate.py
            modified:   src/pnn.py
            modified:   src/pworld.py
            modified:   test/test_pnn.py
            modified:   theory/illustrate.ipynb
    """
    toAdd, toRm = gitList(lines)
    print("git add", " ".join(toAdd))

if __name__ == "__main__":
    # text = input()
    lines = sys.stdin.read().splitlines()
    print("read .........")
    for i in lines:
        print(i)
    print("---------------")
    parse(lines)
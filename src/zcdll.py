import os
import sys
import subprocess


def shell(cmd, verbose = False):
    """execute a command and return either output or error
    message
    """
    output, err = "", ""
    # print(" ".join(cmd))
    try:
        output = subprocess.check_output(cmd, text=True)
        if verbose:
            print(output)
    except subprocess.CalledProcessError as e:
        err = e.output
        if verbose:
            print("Command failed with error:")
            print(err)
    return output.strip(), err.strip()


def getDep(exe, verbose = False):
    """Get its dependencies
    """
    tag = ["following dependencies:",
        "Image has the following delay load dependencies:",
		"Summary" ]
    cmd = ["dumpbin", "/dependents", exe]
    num = len(tag)
    
    output, _ = shell(cmd, verbose=verbose)
	
    if len(output) == 0:
        return None

    st, ed = -1, -1
    txt = ""
    for i in range(num):
        if st < 0:
            st = output.find(tag[i])
            if st>=0:
                st += len(tag[i])
                continue
				
				
        ed = output.find(tag[i], st)
        if ed >= 0:
		    # -- both st and ed exist
            txt += output[st:ed]
            st = ed + len(tag[i])
			
    dep = txt.strip().split("\n")
    lst = [d.strip() for d in dep if d.strip()]
    # print("dlls =", lst)
    return lst

GoodDll = set()
BadDll = set()
chain = []

def canLoad(dll):
    """Find out whether this dll can be loaded
    """
    if dll in GoodDll:
        return True
    if dll in BadDll:
        return False
    
    cmd = ["where", dll]
    output, _ = shell(cmd)
    if len(output) == 0:
        # -- can not find it
        BadDll.add(dll)
        return False
    # print("...", dll, "<<< ", output)
    chain.append(dll)

    lst = getDep(output, verbose=False)
    # print("Next:", output, lst)
    if lst is None or len(lst) == 0:
        GoodDll.add(dll)
        print("...".join(chain))
        chain.pop()
        return True

    broken = False
    for f in lst:
        if not canLoad(f):
            BadDll.add(dll)
            broken = True
        
    chain.pop()
    return (not broken)
        

if __name__ == "__main__":
    GoodDll.clear()
    BadDll.clear()
    file = sys.argv[1]
    # dlls = getDep(sys.argv[1])
    if canLoad(file):
        print(file, "can be loaded")
    else:
        print(file, "can not load")

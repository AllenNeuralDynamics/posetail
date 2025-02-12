import os
import re


def extract_name(fname, pattern):
    '''
    uses regex to extract name from data path
    '''
    pattern_compiled = re.compile(pattern)

    base_name = os.path.basename(fname)
    m = pattern_compiled.search(base_name)

    if m is not None:
        name = m[0]
    else: 
        name = ''

    return name


def extract_num(fname, pattern):
    '''
    uses regex to extract num from data path
    '''

    base_name = os.path.basename(fname)
    m = re.findall(pattern, base_name)

    if m is not None:
        name = m[0]
        if name.isdigit():
            name = int(name)
    else: 
        name = ''

    return name
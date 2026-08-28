#!/usr/bin/env python3
"""Minimal HE+Fix execution helpers extracted from the original experiment utilities."""
import contextlib
import io
import os
import resource
import signal
from contextlib import contextmanager
from copy import deepcopy
from typing import List
import numpy as np

PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"
_SUCCESS, _FAILED, _TIMEOUT, _UNKNOWN = 0, 1, 2, 3
_mapping = {_SUCCESS: PASS, _FAILED: FAIL, _TIMEOUT: TIMEOUT, _UNKNOWN: None}

class TimeoutException(Exception):
    pass

class WriteOnlyStringIO(io.StringIO):
    def read(self, *args, **kwargs): raise IOError
    def readline(self, *args, **kwargs): raise IOError
    def readlines(self, *args, **kwargs): raise IOError
    def readable(self, *args, **kwargs): return False

class redirect_stdin(contextlib._RedirectStream):
    _stream = "stdin"

@contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield

@contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

def is_floats(x) -> bool:
    if isinstance(x, float): return True
    if isinstance(x, list): return all(is_floats(i) for i in x)
    return False

def eval_exact_match(out, exp, atol=0, use_set=False):
    if isinstance(out, str) and isinstance(exp, str):
        if 'error' in out.lower() and 'error' in exp.lower(): return True
    if atol == 0 and is_floats(exp): atol = 1e-6
    if use_set:
        try:
            out = set(out); exp = set(exp)
        except Exception:
            pass
    if isinstance(out, str) and isinstance(exp, str) and 'Error' in out and 'Error' in exp: return True
    elif out != exp and atol != 0:
        return np.allclose(out, exp, rtol=1e-07, atol=atol)
    else:
        return out == exp

IMPORT_HELPER = {
    "python": [
        "import math", "import re", "import sys", "import copy", "import datetime",
        "import itertools", "import collections", "import heapq", "import functools",
        "import hashlib", "import numpy", "import numpy as np", "import string",
        "from typing import *", "from collections import *", "from functools import *"
    ]
}

@contextmanager
def memory_limit(max_mem):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (max_mem, hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))

def unsafe_execute(dataset: str, entry_point: str, code: str, inputs: List, expected: List, verbose: bool=False, use_set: bool=False):
    exec_globals = {}
    pass_rate, exec_out = [], []
    try:
        with time_limit(2.0), memory_limit(256 * 1024 * 1024):
            with swallow_io():
                exec(code, exec_globals)
                if len(entry_point): fn = exec_globals[entry_point]
    except Exception as e:
        if not len(inputs) or not len(entry_point):
            if verbose: return _mapping[_TIMEOUT], 'Error: ' + str(e)
            return _mapping[_TIMEOUT]
        if verbose: return [_mapping[_TIMEOUT]]*len(inputs), ['Error: ' + str(e)]*len(inputs)
        return [_mapping[_TIMEOUT]]*len(inputs)
    if not len(inputs) or not len(entry_point):
        if verbose: return _mapping[_SUCCESS], _mapping[_SUCCESS]
        return _mapping[_SUCCESS]
    for i, inp in enumerate(inputs):
        try:
            with time_limit(2.0), memory_limit(256 * 1024 * 1024):
                with swallow_io(): out = fn(*deepcopy(inp))
            exp = expected[i]
            exact_match = eval_exact_match(out, exp, use_set=use_set)
            exec_out.append(out)
        except Exception as e:
            pass_rate.append(_mapping[_TIMEOUT]); exec_out.append('Error: ' + str(e)); continue
        if not isinstance(exact_match, bool):
            try: exact_match = exact_match.all()
            except Exception:
                pass_rate.append(_mapping[_FAILED]); continue
        pass_rate.append(_mapping[_SUCCESS] if exact_match else _mapping[_FAILED])
    if verbose:
        assert len(exec_out) == len(pass_rate)
        return pass_rate, exec_out
    return pass_rate

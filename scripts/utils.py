########################################################################
##
## Copyright 2014 PMC-Sierra, Inc.
##
## Licensed under the Apache License, Version 2.0 (the "License"); you
## may not use this file except in compliance with the License. You may
## obtain a copy of the License at
## http://www.apache.org/licenses/LICENSE-2.0 Unless required by
## applicable law or agreed to in writing, software distributed under the
## License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
## CONDITIONS OF ANY KIND, either express or implied. See the License for
## the specific language governing permissions and limitations under the
## License.
##
########################################################################

########################################################################
##
##   Author: Logan Gunthorpe
##
##   Date: Oct 23, 2014
##
##   Description:
##     Misc utility functions.
##
########################################################################

import os

def isjpeg(fname):
    ext = os.path.splitext(fname)[-1]
    return ext.lower() in (".jpg", ".jpeg")

def istiff(fname):
    ext = os.path.splitext(fname)[-1]
    return ext.lower() in (".tif", ".tiff")

def run_dir(directory, call, *args, **kws):
    for root, dirs, files in os.walk(directory, followlinks=True):
        for f in files:
            fname = os.path.join(root, f)
            call(fname, *args, **kws)

def run(files, call, *args, **kws):
    for f in files:
        if os.path.isdir(f):
            kws["reldir"] = f
            run_dir(f, call, *args, **kws)
        else:
            kws["reldir"] = os.path.dirname(f)
            call(f, *args, **kws)


_si_suffixes = [(1e15,  'P'),
                (1e12,  'T'),
                (1e9,   'G'),
                (1e6,   'M'),
                (1e3,   'k'),
                (1e0,   ''),
                (1e-3,  'm'),
                (1e-6,  'u'),
                (1e-9,  'n'),
                (1e-12, 'p'),
                (1e-15, 'f'),
               ]

def si_suffix(val):
    for mult, suf in _si_suffixes:
        if val >= mult:
            return val / mult, suf
    return val, ''

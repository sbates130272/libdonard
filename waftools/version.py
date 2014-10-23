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
##      Obtain a version string from git information
##
########################################################################

import os
import subprocess as sp

template = """
#ifndef VERSION_H
#define VERSION_H

#define VERSION "%s"

#endif
"""

def options(opt):
    pass

def configure(conf):
    conf.find_program(["git"], var='GIT')


from waflib.Task import Task

class VersionHeader(Task):
    color = "PINK"

    def __init__(self, *k, **kw):
        Task.__init__(self, *k, **kw)

        if "target" in kw:
            self.set_outputs(kw["target"])

    def run(self):
        rev = self.signature()
        rev = rev.strip()

        for o in self.outputs:
            f = open(o.abspath(), "w")
            print >> f, template % (rev)
            f.close()

    def signature(self):
        try: return self.cache_sig
        except AttributeError: pass

        p = sp.Popen([self.env.GIT, "describe"], stdout=sp.PIPE,
                     stderr=open(os.devnull, "w"))
        if p.wait() != 0:
            p = sp.Popen([self.env.GIT, "rev-parse",
                          "--short", "HEAD"],
                         stdout=sp.PIPE, stderr=open(os.devnull, "w"))
        version = p.communicate()[0].strip()

        status = sp.Popen([self.env.GIT, "status", "--porcelain"],
                          stdout=sp.PIPE).communicate()[0]
        if status.strip():
            version += "M"


        self.cache_sig = version

        return self.cache_sig


def build(ctx):
    tsk = VersionHeader(target=ctx.path.find_or_declare("version.h"),
                        env=ctx.env)
    ctx.add_to_group(tsk)

#!/usr/bin/env python
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
##      OpenCL for WAF
##
########################################################################


"OpenCL"

import os
from waflib import Task
from waflib.TaskGen import extension

class opencl(Task.Task):
	run_str = ("${LD} -r -b binary -o ${TGT} ${SRC}")
	color   = 'GREEN'
	shell   = False

@extension('.cl')
def c_hook(self, node):
	return self.create_compiled_task('opencl', node)

def configure(conf):
    conf.find_program('ld', var="LD")
    conf.check(header_name="CL/opencl.h", lib="OpenCL", uselib_store="OPENCL",
               mandatory=False)

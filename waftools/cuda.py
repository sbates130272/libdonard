#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2010

"cuda"

import os
from waflib import Task
from waflib.TaskGen import extension
from waflib.Tools import ccroot, c_preproc
from waflib.Configure import conf

class cuda(Task.Task):
	run_str = ("${NVCC} ${CUDAFLAGS} ${NVCC_COMPILER:CXXFLAGS} "
               "${FRAMEWORKPATH_ST:FRAMEWORKPATH} "
               "${CPPPATH_ST:INCPATHS} ${DEFINES_ST:DEFINES} "
               "${NVCC_SRC_F}${SRC} ${NVCC_TGT_F}${TGT}")
	color   = 'GREEN'
	ext_in  = ['.h']
	vars    = ['CCDEPS']
	scan    = c_preproc.scan
	shell   = False

@extension('.cu', '.cuda')
def c_hook(self, node):
	return self.create_compiled_task('cuda', node)

def configure(conf):
    v = conf.env
    v['NVCC_COMPILER'] = "-Xcompiler=%s"
    v['NVCC_SRC_F'] = []
    v['NVCC_TGT_F']= ['-c','-o']

    if conf.find_program('nvcc', var='NVCC', mandatory=False):
        conf.find_cuda_libs()

@conf
def find_cuda_libs(self):
	"""
	find the cuda include and library folders

	use ctx.program(source='main.c', target='app', use='CUDA CUDART')
	"""

	if not self.env.NVCC:
		self.fatal('check for nvcc first')

	d = self.root.find_node(self.env.NVCC).parent.parent

	node = d.find_node('include')
	_includes = node and node.abspath() or ''
	self.env.INCLUDES += [_includes]

	_libpath = []
	for x in ('lib64', 'lib'):
		try:
			_libpath.append(d.find_node(x).abspath())
		except:
			pass

	mandatory = False
	self.check(header_name='cuda.h', lib='cuda', libpath=_libpath, includes=_includes,
               uselib_store='CUDA', mandatory=mandatory)
	self.check(header_name='cuda_runtime.h', lib='cudart', libpath=_libpath, includes=_includes,
               uselib_store='CUDART', mandatory=mandatory)
	self.check(header_name='cufft.h', lib='cufft', libpath=_libpath, includes=_includes,
	           uselib_store='CUFFT', mandatory=mandatory)

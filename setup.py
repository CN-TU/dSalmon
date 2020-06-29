#!/usr/bin/env python3

# Copyright (c) 2020 CN Group, TU Wien
# Released under the GNU Lesser General Public License version 3,
# see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

import glob
from distutils.core import setup, Extension

import os

try:
	import numpy
except:
	raise ImportError('Numpy is required for building this package.', name='numpy')

numpy_path = os.path.dirname(numpy.__file__)
numpy_include = numpy_path + '/core/include'

CPP_SOURCES = [
	'cpp/slidingWindow.cpp',
	'swig/MTree_wrapper.cpp',
	'swig/outlier_wrapper.cpp',
	'swig/dSalmon_wrap.cxx'
]

dSalmon_cpp = Extension(
	'dSalmon.swig._dSalmon',
	CPP_SOURCES,
	include_dirs=['cpp', numpy_include, 'contrib/boost']
)

setup(
	name='dSalmon',
	version='0.1',
	author='Alexander Hartl',
	author_email='alexander.hartl@tuwien.ac.at',
	url='none',
	packages=['dSalmon', 'dSalmon.swig'],
	package_dir={'dSalmon': 'python', 'dSalmon.swig': 'swig'},
	ext_modules = [ dSalmon_cpp ],
	install_requires=['numpy']
)

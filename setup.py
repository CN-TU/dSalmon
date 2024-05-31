#!/usr/bin/env python3

# Copyright (c) 2020 CN Group, TU Wien
# Released under the GNU Lesser General Public License version 3,
# see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

import glob
import os
import pathlib
import platform

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

parent_path = pathlib.Path(__file__).parent

try:
    import numpy
except:
    raise ImportError('Numpy is required for building this package.', name='numpy')

try:
    import cpuinfo
except:
    raise ImportError('py-cpuinfo is required for building this package.', name='py-cpuinfo')

def get_simd_args():
    info = cpuinfo.get_cpu_info()
    flags = info.get('flags', [])
    args = []
    if 'avx2' in flags:
        args.append('/arch:AVX2' if platform.system() == 'Windows' else '-mavx2')
    if 'sse3' in flags:
        args.append('/arch:SSE3' if platform.system() == 'Windows' else '-msse3')
    return args
    
def get_strip_args():
    # Strip .so file to an acceptable size
    if platform.system() == 'Windows':
        return [ ]
    else:
        return [ '-g0' ]
    
numpy_path = os.path.dirname(numpy.__file__)
numpy_include = numpy_path + '/core/include'

CPP_SOURCES = [
    'swig/MTree_wrapper.cpp',
    'swig/outlier_wrapper.cpp',
    'swig/statisticstree_wrapper.cpp',
    'swig/dSalmon_wrap.cxx'
]

dSalmon_cpp = Extension(
    'dSalmon.swig._dSalmon',
    CPP_SOURCES,
    include_dirs=['cpp', numpy_include, 'contrib/boost'],
    extra_compile_args=get_simd_args() + get_strip_args()
)

setup(
    name='dSalmon',
    version='0.1',
    license='LGPL-3.0',
    description='dSalmon is a framework for analyzing data streams',
    author='Alexander Hartl',
    author_email='alexander.hartl@tuwien.ac.at',
    url='https://github.com/CN-TU/dSalmon',
    project_urls={
        'Source': 'https://github.com/CN-TU/dSalmon',
        'Documentation': 'https://dSalmon.readthedocs.io',
        'Tracker': 'https://github.com/CN-TU/dSalmon/issues'
    },
    long_description=(parent_path / 'README.rst').read_text(encoding='utf-8'),
    long_description_content_type='text/x-rst',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering'
    ],
    packages=['dSalmon', 'dSalmon.swig'],
    package_dir={'dSalmon': 'python', 'dSalmon.swig': 'swig'},
    ext_modules = [ dSalmon_cpp ],
    install_requires=['numpy'],
    python_requires='>=3.5'
)

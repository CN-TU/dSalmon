// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

%module dSalmon

%{
#define SWIG_FILE_WITH_INIT
#define SWIG_PYTHON_STRICT_BYTE_CHAR

#include "distance_wrappers.h"
#include "MTree_wrapper.h"
#include "outlier_wrapper.h"

%}

%include "numpy.i"
%include "std_string.i"
%include "std_complex.i"

%include "array_types.h"

// typemaps for NumpyArray1 and NumpyArray2 types
%define %dSalmon_typemaps(DATA_TYPE, DATA_TYPECODE, DIM_TYPE)

// this is necessary to stop SWIG from wrapping NumpyArray types in a SwigValueWrapper,
// which would be used without being initialized.


%feature("novaluewrapper") NumpyArray1<DATA_TYPE>;
%template() NumpyArray1<DATA_TYPE>;
%feature("novaluewrapper") NumpyArray2<DATA_TYPE>;
%template() NumpyArray2<DATA_TYPE>;


%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY,
           fragment="NumPy_Macros")
  (const NumpyArray1<DATA_TYPE>)
{
  $1 = is_array($input) || PySequence_Check($input);
}
%typemap(in,
         fragment="NumPy_Fragments")
  (const NumpyArray1<DATA_TYPE>)
  (PyArrayObject* array=NULL, int is_new_object=0)
{
  npy_intp size[1] = { -1 };
  array = obj_to_array_contiguous_allow_conversion($input,
                                                   DATA_TYPECODE,
                                                   &is_new_object);
  if (!array || !require_dimensions(array, 1) ||
      !require_size(array, size, 1)) SWIG_fail;
  $1.data = (DATA_TYPE*) array_data(array);
  $1.dim1 = (DIM_TYPE) array_size(array,0);
}
%typemap(freearg)
  (const NumpyArray1<DATA_TYPE>)
{
  if (is_new_object$argnum && array$argnum)
    { Py_DECREF(array$argnum); }
}


%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY,
           fragment="NumPy_Macros")
  (const NumpyArray2<DATA_TYPE>)
{
  $1 = is_array($input) || PySequence_Check($input);
}
%typemap(in,
         fragment="NumPy_Fragments")
  (const NumpyArray2<DATA_TYPE>)
  (PyArrayObject* array=NULL, int is_new_object=0)
{
  npy_intp size[2] = { -1, -1 };
  array = obj_to_array_contiguous_allow_conversion($input, DATA_TYPECODE,
                                                   &is_new_object);
  if (!array || !require_dimensions(array, 2) ||
      !require_size(array, size, 2)) SWIG_fail;
  $1.data = (DATA_TYPE*) array_data(array);
  $1.dim1 = (DIM_TYPE) array_size(array,0);
  $1.dim2 = (DIM_TYPE) array_size(array,1);
}
%typemap(freearg)
  (const NumpyArray2<DATA_TYPE>)
{
  if (is_new_object$argnum && array$argnum)
    { Py_DECREF(array$argnum); }
}





%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY,
           fragment="NumPy_Macros")
  (NumpyArray1<DATA_TYPE>)
{
  $1 = is_array($input) && PyArray_EquivTypenums(array_type($input),
                                                 DATA_TYPECODE);
}
%typemap(in,
         fragment="NumPy_Fragments")
  (NumpyArray1<DATA_TYPE>)
  (PyArrayObject* array=NULL, int i=1)
{
  array = obj_to_array_no_conversion($input, DATA_TYPECODE);
  if (!array || !require_dimensions(array,1) || !require_contiguous(array)
      || !require_native(array)) SWIG_fail;
  $1.data = (DATA_TYPE*) array_data(array);
  $1.dim1 = (DIM_TYPE) array_size(array,0);
}

%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY,
           fragment="NumPy_Macros")
  (NumpyArray2<DATA_TYPE>)
{
  $1 = is_array($input) && PyArray_EquivTypenums(array_type($input),
                                                 DATA_TYPECODE);
}
%typemap(in,
         fragment="NumPy_Fragments")
  (NumpyArray2<DATA_TYPE>)
  (PyArrayObject* array=NULL)
{
  array = obj_to_array_no_conversion($input, DATA_TYPECODE);
  if (!array || !require_dimensions(array,2) || !require_contiguous(array)
      || !require_native(array)) SWIG_fail;
  $1.data = (DATA_TYPE*) array_data(array);
  $1.dim1 = (DIM_TYPE) array_size(array,0);
  $1.dim2 = (DIM_TYPE) array_size(array,1);
}

%enddef



%dSalmon_typemaps(signed char         , NPY_BYTE     , int)
%dSalmon_typemaps(unsigned char       , NPY_UBYTE    , int)
%dSalmon_typemaps(short               , NPY_SHORT    , int)
%dSalmon_typemaps(unsigned short      , NPY_USHORT   , int)
%dSalmon_typemaps(int                 , NPY_INT      , int)
%dSalmon_typemaps(unsigned int        , NPY_UINT     , int)
%dSalmon_typemaps(long                , NPY_LONG     , int)
%dSalmon_typemaps(unsigned long       , NPY_ULONG    , int)
%dSalmon_typemaps(long long           , NPY_LONGLONG , int)
%dSalmon_typemaps(unsigned long long  , NPY_ULONGLONG, int)
%dSalmon_typemaps(float               , NPY_FLOAT    , int)
%dSalmon_typemaps(double              , NPY_DOUBLE   , int)
%dSalmon_typemaps(int8_t              , NPY_INT8     , int)
%dSalmon_typemaps(int16_t             , NPY_INT16    , int)
%dSalmon_typemaps(int32_t             , NPY_INT32    , int)
%dSalmon_typemaps(int64_t             , NPY_INT64    , int)
%dSalmon_typemaps(uint8_t             , NPY_UINT8    , int)
%dSalmon_typemaps(uint16_t            , NPY_UINT16   , int)
%dSalmon_typemaps(uint32_t            , NPY_UINT32   , int)
%dSalmon_typemaps(uint64_t            , NPY_UINT64   , int)
%dSalmon_typemaps(std::complex<float> , NPY_CFLOAT   , int)
%dSalmon_typemaps(std::complex<double>, NPY_CDOUBLE  , int)

%init %{
import_array();
%}

// we instantiate float and double versions of all algorithms
%define DEFINE_FLOATINSTANTIATIONS(name)
%template(name ## 32) name ## _wrapper<float>;
%template(name ## 64) name ## _wrapper<double>;
%enddef


%include "distance_wrappers.h"
%include "MTree_wrapper.h"
%include "outlier_wrapper.h"

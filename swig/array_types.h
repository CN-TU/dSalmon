
#ifndef DSALMON_ARRAY_TYPES_H
#define DSALMON_ARRAY_TYPES_H


template<typename DataType>
struct NumpyArray1 {
	DataType* data;
	int dim1;
};

template<typename DataType>
struct NumpyArray2 {
	DataType* data;
	int dim1;
	int dim2;
};

// This is a SWIG macro which is used for instantiating
// float and double specializations of wrapper templates.
// Since it is only used by SWIG, define it to nothing
// for the C++ compiler.
#ifndef SWIG
#define DEFINE_FLOATINSTANTIATIONS(name)
#endif

#endif

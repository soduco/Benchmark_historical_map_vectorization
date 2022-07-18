#include <Python.h>
#include <assert.h>
#include "numpy/arrayobject.h"
#include "MBD_distance_2d.h"
#include <iostream>
using namespace std;

// example to use numpy object: http://blog.debao.me/2013/04/my-first-c-extension-to-numpy/
// write a c extension ot Numpy: http://folk.uio.no/hpl/scripting/doc/python/NumPy/Numeric/numpy-13.html


static PyObject *
geodesic_saddle_wrapper(PyObject *self, PyObject *args)
{
    PyObject *I=NULL, *Seed=NULL;
    PyArrayObject *arr_I=NULL, *arr_Seed=NULL;
    
    if (!PyArg_ParseTuple(args, "OO", &I, &Seed)) return NULL;
    
    arr_I = (PyArrayObject*)PyArray_FROM_OTF(I, NPY_UINT8, NPY_IN_ARRAY);
    if (arr_I == NULL) return NULL;
    
    arr_Seed = (PyArrayObject*)PyArray_FROM_OTF(Seed, NPY_INT, NPY_IN_ARRAY);
    if (arr_Seed == NULL) return NULL;
    
    
    int nd = PyArray_NDIM(arr_I);   //number of dimensions
    npy_intp * shape = PyArray_DIMS(arr_I);  // npy_intp array of length nd showing length in each dim.
    npy_intp * shape_seed = PyArray_DIMS(arr_Seed);

    npy_intp output_shape[2];
    output_shape[0] = shape[0];
    output_shape[1] = shape[1];

    PyArrayObject * saddle = (PyArrayObject*)  PyArray_SimpleNew(2, output_shape, NPY_UINT8);
    geodesic_saddle((const unsigned char *)arr_I->data, (const int *)arr_Seed->data, 
           (unsigned char *) saddle->data, shape[0], shape[1]);
    
    Py_DECREF(arr_I);
    Py_DECREF(arr_Seed);
    //Py_INCREF(distance);
    return PyArray_Return(saddle);
}


static PyMethodDef Methods[] = {
    {"geodesic_saddle",  geodesic_saddle_wrapper, METH_VARARGS, "computing 2d shortest path all"},
};

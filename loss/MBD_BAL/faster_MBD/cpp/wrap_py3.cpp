#include "MBD_distance.cpp"


static struct PyModuleDef cMBDDis =
{
    PyModuleDef_HEAD_INIT,
    "MBD", /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    Methods
};


PyMODINIT_FUNC PyInit_MBD(void) {
    import_array();
    return PyModule_Create(&cMBDDis);
}

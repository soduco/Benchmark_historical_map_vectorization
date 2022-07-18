#include "pathloss_distance.cpp"


static struct PyModuleDef cpathlossDis =
{
    PyModuleDef_HEAD_INIT,
    "pathloss", /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    Methods
};


PyMODINIT_FUNC PyInit_pathloss(void) {
    import_array();
    return PyModule_Create(&cpathlossDis);
}

#include "MBD_distance.cpp"

PyMODINIT_FUNC
initMBD(void) {
    (void) Py_InitModule("MBD", Methods);
    import_array();
}

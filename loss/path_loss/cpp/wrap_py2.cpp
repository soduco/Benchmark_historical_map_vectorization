#include "pathloss_distance.cpp"

PyMODINIT_FUNC
initMBD(void) {
    (void) Py_InitModule("pathloss", Methods);
    import_array();
}

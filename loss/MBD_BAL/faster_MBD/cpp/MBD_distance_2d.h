#include <iostream>
#include <vector>
using namespace std;

struct Point2D
{
    // float distance;
    int w;
    int h;
};


void geodesic_saddle(const unsigned char * img, const int * seeds, unsigned char * saddle, 
    int height, int width);

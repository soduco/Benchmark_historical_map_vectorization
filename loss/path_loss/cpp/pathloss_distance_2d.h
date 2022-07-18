#include <iostream>
#include <vector>
using namespace std;

struct Point2D
{
    // float distance;
    int w;
    int h;
};


void pathloss_waterflow(const unsigned char * img, const unsigned char * seeds, unsigned char * distance, 
    int height, int width);

void geodesic_shortest(const unsigned char * img, const unsigned char * seeds, const unsigned char * destination, unsigned char * shortestpath, 
                              int height, int width);

void geodesic_shortest_all(const unsigned char * img, const unsigned char * seeds, const unsigned char * destination, unsigned char * shortestpath, 
                              int height, int width);
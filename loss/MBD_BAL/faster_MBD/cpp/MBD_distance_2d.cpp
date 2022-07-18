#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include "util.h"
#include "MBD_distance_2d.h"
#include <queue>
#include <stdio.h>
#include <stdlib.h> 
#include <math.h> 
#include <limits.h>
using namespace std;



void geodesic_saddle(const unsigned char * img, const int * seeds, unsigned char * saddle, 
                              int height, int width)
{
    int * label = new int[height * width];
    int * distance = new int[height * width];

    int max_seed = 0;
    for(int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            int seed_type = get_pixel<int>(seeds, height, width, h, w);
            if (max_seed < seed_type)
                max_seed = seed_type;
        }
    }
    
    max_seed = max_seed + 1;

    int * count_saddle = new int[max_seed * max_seed];
    for(int h = 0; h < max_seed; h++)
    {
        for (int w = 0; w < max_seed; w++)
        {
            set_pixel<int>(count_saddle, max_seed, max_seed, h, w, 0); 
        }
    }
    
    vector<queue<Point2D> > Q(1000);

    // point state: 0--acceptd, 1--temporary, 2--far away
    // get initial accepted set and far away set

    int init_dis;

    for(int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            Point2D p;
            p.h = h;
            p.w = w;
            int seed_type = get_pixel<int>(seeds, height, width, h, w);
            unsigned char img_value = get_pixel<unsigned char>(img, height, width, h, w);

            if(seed_type > 0){
                init_dis = 0;
                Q[init_dis].push(p);
                set_pixel<int>(label, height, width, h, w, int(seed_type));
                set_pixel<int>(distance, height, width, h, w, init_dis);
                set_pixel<unsigned char>(saddle, height, width, h, w, 0); 
                   
            }
            else{
                init_dis = 1000;
                set_pixel<int>(distance, height, width, h, w, init_dis);
                set_pixel<int>(label, height, width, h, w, 0);
                set_pixel<unsigned char>(saddle, height, width, h, w, 0);                                        
            }
        }
    }

    int dh[4] = { 1 ,-1 , 0, 0};
    int dw[4] = { 0 , 0 , 1,-1};

    // Proceed the propagation from the marker to all pixels in the image
    for (int lvl = 0; lvl < 1000; lvl++)
    {
        while (!Q[lvl].empty())
        {
            Point2D p = Q[lvl].front();
            Q[lvl].pop();

            for (int n1 = 0 ; n1 < 4 ; n1++)
            {
                int tmp_h  = p.h + dh[n1];
                int tmp_w  = p.w + dw[n1];

                if (tmp_h >= 0 and tmp_h < height and tmp_w >= 0 and tmp_w < width)
                {
                    Point2D r;
                    r.h = tmp_h;
                    r.w = tmp_w;

                    unsigned char image_value_r = get_pixel<unsigned char>(img, height, width, r.h, r.w);
                    unsigned char image_value_p = get_pixel<unsigned char>(img, height, width, p.h, p.w);

                    int temp_r = get_pixel<int>(distance, height, width, r.h, r.w);
                    int temp_p = get_pixel<int>(distance, height, width, p.h, p.w);

                    int label_r = get_pixel<int>(label, height, width, r.h, r.w);
                    int label_p = get_pixel<int>(label, height, width, p.h, p.w);

                    if (label_r != 0 and label_r != label_p)
                    {
                        // int count = get_pixel<int>(count_saddle, max_seed, max_seed, min(label_r, label_p) ,max(label_r, label_p) );
                        // if (count < 500)
                        // {
                        //     count = count + 1;
                            // set_pixel<int>(count_saddle, max_seed, max_seed, min(label_r, label_p) ,max(label_r, label_p), count);
                            set_pixel<unsigned char>(saddle, height, width, r.h, r.w, 255);
                        // }                        
                    }
                    int tmp_dis = temp_p + abs(image_value_p - image_value_r);

                    if (tmp_dis < temp_r)
                    {
                        set_pixel<int>(distance, height, width, r.h, r.w, tmp_dis);
                        Q[tmp_dis].push(r);


                        set_pixel<int>(label, height, width, r.h, r.w, label_p);
                    }
                }
            }
        }
    }


    delete label;
    delete distance;

}

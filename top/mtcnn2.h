#ifndef MTCNN_H
#define MTCNN_H

#include <stdint.h>
#include "cnntype.h"

#define MAX_FACES_PER_FRAME    ( 256)

class Pnet
{
public:
    Pnet();
    ~Pnet();

    // void run(Mat &image, float scale);
    void run(int M, int N, unsigned char * image, float scale);

    float nms_threshold;
    mydataFmt Pthreshold;
    bool firstFlag;

public:
    Pnet(int M, int N);
    void generateBbox(int M, int N, cnn_type_t scale, const cnn_type_t * psco, const cnn_type_t * ploc);

public:
    int M0, 
        N0, 
        C0;

    int M01_3x3, 
        N01_3x3, 
        C01_3x3;

    int M01_max2x2, 
        N01_max2x2, 
        C01_max2x2;

    int M02_3x3, 
        N02_3x3, 
        C02_3x3;

    int M03_3x3, 
        N03_3x3, 
        C03_3x3;

    int M41_1x1, 
        N41_1x1, 
        C41_1x1;

    int M42_1x1, 
        N42_1x1, 
        C42_1x1;

    cnn_type_t * X0;

    cnn_type_t * Y01_3x3;
    cnn_type_t * Y01_max2x2;

    cnn_type_t * Y02_3x3;
    cnn_type_t * Y03_3x3;

    cnn_type_t * Y41_1x1;
    cnn_type_t * Y42_1x1;

    int       _nbox;
    int16_t * _x1, 
            * _x2, 
            * _y1, 
            * _y2;

    int32_t    * _area;
    cnn_type_t * _score;

    cnn_type_t * _loc;
};

class Rnet
{
public:
    Rnet();
    ~Rnet();
    float Rthreshold;
    // void run(Mat &image);

    void run(unsigned char *);
public:
    int M0, 
        N0, 
        C0;

    int M01_3x3, 
        N01_3x3, 
        C01_3x3;

    int M01_max3x3, 
        N01_max3x3, 
        C01_max3x3;

    int M02_3x3, 
        N02_3x3, 
        C02_3x3;

    int M02_max3x3, 
        N02_max3x3, 
        C02_max3x3;

    int M03_2x2, 
        N03_2x2, 
        C03_2x2;

    int M04_fc4, 
        N04_fc4, 
        C04_fc4;

    int M51_scr, 
        N51_scr, 
        C51_scr;

    int M52_loc, 
        N52_loc, 
        C52_loc;

    cnn_type_t * X0;

    cnn_type_t * Y01_3x3;
    cnn_type_t * Y01_max3x3;

    cnn_type_t * Y02_3x3;
    cnn_type_t * Y02_max3x3;

    cnn_type_t * Y03_2x2;
    cnn_type_t * Y04_fc4;

    cnn_type_t * Y51_scr;
    cnn_type_t * Y52_loc;
};

class Onet
{
public:
    Onet();
    ~Onet();
    // void run(Mat &image);
    float Othreshold;

    void run(unsigned char * image);
public:
    int M0, 
        N0, 
        C0;

    int M01_3x3, 
        N01_3x3, 
        C01_3x3;

    int M01_max3x3, 
        N01_max3x3, 
        C01_max3x3;

    int M02_3x3, 
        N02_3x3, 
        C02_3x3;

    int M02_max3x3, 
        N02_max3x3, 
        C02_max3x3;

    int M03_3x3, 
        N03_3x3, 
        C03_3x3;

    int M03_max2x2, 
        N03_max2x2, 
        C03_max2x2;

    int M04_2x2, 
        N04_2x2, 
        C04_2x2;

    int M05_fc5, 
        N05_fc5, 
        C05_fc5;

    int M61_scr, 
        N61_scr, 
        C61_scr;

    int M62_loc, 
        N62_loc, 
        C62_loc;

    int M63_key, 
        N63_key, 
        C63_key;

    cnn_type_t * X0;

    cnn_type_t * Y01_3x3;
    cnn_type_t * Y01_max3x3;

    cnn_type_t * Y02_3x3;
    cnn_type_t * Y02_max3x3;

    cnn_type_t * Y03_3x3;
    cnn_type_t * Y03_max2x2;

    cnn_type_t * Y04_2x2;
    cnn_type_t * Y05_fc5;

    cnn_type_t * Y61_scr;
    cnn_type_t * Y62_loc;
    cnn_type_t * Y63_key;    
};

class mtcnn
{
public:
    mtcnn(int row, int col);
    ~mtcnn();
    void findFace(uint8_t * pimage, bbox_chain_t * pbbox);
    
public:
    // Mat reImage;
    
    float nms_threshold[3];
    
    int    _nscale;
    float  _scales[64];
    void * _interpcb[64];

    uint8_t * _images[64];

    Rnet refineNet;
    Onet outNet;

public:
    Pnet *simpleFace_;

    int16_t * _x1_fst;
    int16_t * _y1_fst;
    int16_t * _x2_fst;
    int16_t * _y2_fst;
    int32_t * _aa_fst;

    cnn_type_t * _sc_fst;
    cnn_type_t * _loc_fst;

    int16_t * _x1_2nd;
    int16_t * _y1_2nd;
    int16_t * _x2_2nd;
    int16_t * _y2_2nd;
    int32_t * _aa_2nd;

    cnn_type_t * _sc_2nd;
    cnn_type_t * _loc_2nd;

    int16_t * _x1_3rd;
    int16_t * _y1_3rd;
    int16_t * _x2_3rd;
    int16_t * _y2_3rd;
    int32_t * _aa_3rd;

    cnn_type_t * _sc_3rd;
    cnn_type_t * _loc_3rd;
    cnn_type_t * _key_3rd;

    int32_t   _npick;
    int32_t * _ipick;

    void       * _pbufa;
};

#endif
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "cnntype.h"
#include "mtcnn2.h"

#include "../coefs/pnet_w01_3x3.h"
#include "../coefs/pnet_w02_3x3.h"
#include "../coefs/pnet_w03_3x3.h"
#include "../coefs/pnet_w41_1x1.h"
#include "../coefs/pnet_w42_1x1.h"

#include "../coefs/pnet_p01_3x3.h"
#include "../coefs/pnet_p02_3x3.h"
#include "../coefs/pnet_p03_3x3.h"

#include "../coefs/pnet_b01_3x3.h"
#include "../coefs/pnet_b02_3x3.h"
#include "../coefs/pnet_b03_3x3.h"
#include "../coefs/pnet_b41_1x1.h"
#include "../coefs/pnet_b42_1x1.h"

#include "../coefs/rnet_w01_3x3.h"
#include "../coefs/rnet_w02_3x3.h"
#include "../coefs/rnet_w03_2x2.h"
#include "../coefs/rnet_w04_fc4.h"
#include "../coefs/rnet_w51_scr.h"
#include "../coefs/rnet_w52_loc.h"

#include "../coefs/rnet_p01_3x3.h"
#include "../coefs/rnet_p02_3x3.h"
#include "../coefs/rnet_p03_2x2.h"
#include "../coefs/rnet_p04_fc4.h"

#include "../coefs/rnet_b01_3x3.h"
#include "../coefs/rnet_b02_3x3.h"
#include "../coefs/rnet_b03_2x2.h"
#include "../coefs/rnet_b04_fc4.h"
#include "../coefs/rnet_b51_scr.h"
#include "../coefs/rnet_b52_loc.h"

#include "../coefs/onet_w01_3x3.h"
#include "../coefs/onet_w02_3x3.h"
#include "../coefs/onet_w03_3x3.h"
#include "../coefs/onet_w04_2x2.h"
#include "../coefs/onet_w05_fc5.h"
#include "../coefs/onet_w61_scr.h"
#include "../coefs/onet_w62_loc.h"
#include "../coefs/onet_w63_key.h"

#include "../coefs/onet_p01_3x3.h"
#include "../coefs/onet_p02_3x3.h"
#include "../coefs/onet_p03_3x3.h"
#include "../coefs/onet_p04_2x2.h"
#include "../coefs/onet_p05_fc5.h"

#include "../coefs/onet_b01_3x3.h"
#include "../coefs/onet_b02_3x3.h"
#include "../coefs/onet_b03_3x3.h"
#include "../coefs/onet_b04_2x2.h"
#include "../coefs/onet_b05_fc5.h"
#include "../coefs/onet_b61_scr.h"
#include "../coefs/onet_b62_loc.h"
#include "../coefs/onet_b63_key.h"

extern "C" void mulpool2(
         int          _n,
   const cnn_type_t * _xloc,
         cnn_type_t   _wloc,
         cnn_type_t * _yloc
);

extern "C" void addpoolbch(
         int          _n,
         int          _bchm,
   const cnn_type_t * _xloc,
   const cnn_type_t * _wloc,
         cnn_type_t * _yloc
);

extern "C" void conv3x3(
         int       M,
         int       N,
         int       C1,
         int       C2,
   const float * __px,
   const float * __pw,
         float * __py
);

extern "C" void conv2x2(
         int       M,
         int       N,
         int       C1,
         int       C2,
   const float * __px,
   const float * __pw,
         float * __py
);

extern "C" void conv1x1(
         int       M,
         int       N,
         int       C1,
         int       C2,
   const float * __px,
   const float * __pw,
         float * __py
);

extern "C" void prelupool(
         int          _n,
   const cnn_type_t * _xloc,
   const cnn_type_t * _bloc,
   const cnn_type_t * _ploc,
         cnn_type_t * _yloc
);

extern "C" void prelupool2(
         int          _n,
         int          _bchm,
   const cnn_type_t * _xloc,
   const cnn_type_t * _bloc,
   const cnn_type_t * _ploc,
         cnn_type_t * _yloc
);

extern "C" void maxpool2x2(
         int  M,
         int  N,
         int  C,
   const cnn_type_t * __px,
         cnn_type_t * __py
);

extern "C" void maxpool3x3(
         int  M,
         int  N,
         int  C,
         int  S,
   const cnn_type_t * __px,
         cnn_type_t * __py
);

extern "C" void softmaxpool(
         int  M,
         int  N,
         int  C,
   const cnn_type_t * _px,
         cnn_type_t * _py
);

extern "C" void fullnet(
         int          _M,
         int          _N,
   const cnn_type_t * _xloc,
   const cnn_type_t * _wloc,
         cnn_type_t * _yloc   
);

extern "C" void rgb2cnnpool(
         int             _n,
   const unsigned char * _xloc,
         cnn_type_t    * _yloc
);

extern "C" void * bilinearInterpolationRGBSetup(
   int  heightSource,
   int  widthSource,
   int  height,
   int  width
);

extern "C" void bilinearInterpolationRGBExecute(
   void          * pcb_,
   uint8_t       * pdst,
   const uint8_t * psrc
);

extern "C" void bilinearInterpolationRGBDestroy(
   void * pcb_
);

extern "C" void bilinearInterpolationRGB24(
   uint8_t       * pdst,
   const uint8_t * psrc,
   int             heightSource,
   int             widthSource,
   int             strideSource,
   int             depthDest
);

extern "C" void bilinearInterpolationRGB48(
   uint8_t       * pdst,
   const uint8_t * psrc,
   int             heightSource,
   int             widthSource,
   int             strideSource,
   int             depthDest
);

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
Pnet::Pnet()
{
    Pthreshold    = 0.6;
    nms_threshold = 0.5;
    firstFlag     = true;
}

Pnet::~Pnet()
{
    _aligned_free(X0        );

    _aligned_free(Y01_3x3   );
    _aligned_free(Y01_max2x2);

    _aligned_free(Y02_3x3   );
    _aligned_free(Y03_3x3   );

    _aligned_free(Y41_1x1   );
    _aligned_free(Y42_1x1   );

    _aligned_free(_x1       );
    _aligned_free(_loc      );
}

void Pnet::run(int rows, int cols, unsigned char * psrc, float scale)
{
    if (firstFlag)
    {
        firstFlag = false;

        M0      = rows; // image.rows;
        N0      = cols; // image.cols;
        C0      = 3;

        M01_3x3 = M0 - 3 + 1;
        N01_3x3 = N0 - 3 + 1;
        C01_3x3 = 10;

        M01_max2x2 = M01_3x3 / 2;
        N01_max2x2 = N01_3x3 / 2;
        C01_max2x2 = C01_3x3;

        M02_3x3 = M01_max2x2 - 3 + 1;
        N02_3x3 = N01_max2x2 - 3 + 1;
        C02_3x3 = 16;

        M03_3x3 = M02_3x3 - 3 + 1;
        N03_3x3 = N02_3x3 - 3 + 1;
        C03_3x3 = 32;

        M41_1x1 = M03_3x3;
        N41_1x1 = N03_3x3;
        C41_1x1 = 2;

        M42_1x1 = M03_3x3;
        N42_1x1 = N03_3x3;
        C42_1x1 = 4;

        // X0 = rgb->pdata;

        X0         = (cnn_type_t *)_aligned_malloc((M0         * N0         * C0         + 16) * sizeof(cnn_type_t), 16);
        
        Y01_3x3    = (cnn_type_t *)_aligned_malloc((M01_3x3    * N01_3x3    * C01_3x3    + 16) * sizeof(cnn_type_t), 16);
        Y01_max2x2 = (cnn_type_t *)_aligned_malloc((M01_max2x2 * N01_max2x2 * C01_max2x2 + 16) * sizeof(cnn_type_t), 16);

        Y02_3x3    = (cnn_type_t *)_aligned_malloc((M02_3x3    * N02_3x3    * C02_3x3    + 16) * sizeof(cnn_type_t), 16);
        Y03_3x3    = (cnn_type_t *)_aligned_malloc((M03_3x3    * N03_3x3    * C03_3x3    + 16) * sizeof(cnn_type_t), 16);

        Y41_1x1    = (cnn_type_t *)_aligned_malloc((M41_1x1    * N41_1x1    * C41_1x1    + 16) * sizeof(cnn_type_t), 16);
        Y42_1x1    = (cnn_type_t *)_aligned_malloc((M42_1x1    * N42_1x1    * C42_1x1    + 16) * sizeof(cnn_type_t), 16);

        _loc       = (cnn_type_t *)_aligned_malloc((8192 * 5) * sizeof(cnn_type_t), 16);
        _score     = (cnn_type_t *)(_loc + 4 * 8192);

        _x1        = (int16_t    *)_aligned_malloc(8192 * 4 * sizeof(int16_t) + 8192 * sizeof(int32_t), 16);
        _y1        = (int16_t    *)(_x1 + 8192);
        _x2        = (int16_t    *)(_y1 + 8192);
        _y2        = (int16_t    *)(_x2 + 8192);
        _area      = (int32_t    *)(_y2 + 8192);
    }
  
    {
        // cnn_type_t *p = X0;
        rgb2cnnpool(M0 * N0 * 3, psrc, X0);
    }

    /* STAGE-1: CONV3x3 - PRELU - MAXPOOL2X2 */
    conv3x3(
        M0, 
        N0, 
        C0, 
        C01_3x3,
        X0, 
        pnet_w01_3x3,
        Y01_3x3
        );

    prelupool2(
        C01_3x3 * M01_3x3 * N01_3x3,
        C01_3x3 * 512,
        Y01_3x3,
        pnet_b01_3x3,
        pnet_p01_3x3,
        Y01_3x3
        );

    maxpool2x2(
        M01_3x3, 
        N01_3x3, 
        C01_3x3,

        Y01_3x3,
        Y01_max2x2 
        );

    /* RTL MATCHING - STEP 1 */
    mulpool2(
        C01_max2x2 * M01_max2x2 * N01_max2x2,
        Y01_max2x2, 
        1.0/8.0, 
        Y01_max2x2
        );

    /* STAGE-2: CONV3x3 - PRELU */
    conv3x3(
        M01_max2x2, 
        N01_max2x2, 
        C01_max2x2, 
        C02_3x3,
        Y01_max2x2, 
        pnet_w02_3x3,
        Y02_3x3
        );

    prelupool2(
        C02_3x3 * M02_3x3 * N02_3x3,
        C02_3x3 * 512,
        Y02_3x3,
        pnet_b02_3x3,
        pnet_p02_3x3,
        Y02_3x3
        );

    /* STAGE-3: CONV3x3 - PRELU */
    conv3x3(
        M02_3x3, 
        N02_3x3, 
        C02_3x3, 
        C03_3x3,
        Y02_3x3, 
        pnet_w03_3x3,
        Y03_3x3
        );

    prelupool2(
        C03_3x3 * M03_3x3 * N03_3x3,
        C03_3x3 * 512,
        Y03_3x3,
        pnet_b03_3x3,
        pnet_p03_3x3,
        Y03_3x3
        );

    /* RTL MATCHING - STEP 3 */
    mulpool2(
        C03_3x3 * M03_3x3 * N03_3x3,
        Y03_3x3, 
        2.0/1.0, 
        Y03_3x3
        );

    /* STAGE-4.1: CONV1x1 - PRELU - SOFTMAX */
    conv1x1(
        M03_3x3, 
        N03_3x3, 
        C03_3x3, 
        C41_1x1,
        Y03_3x3, 
        pnet_w41_1x1,
        Y41_1x1
        );

    addpoolbch(
        C41_1x1 * M41_1x1 * N41_1x1,
        C41_1x1 * 512,
        Y41_1x1,
        pnet_b41_1x1,
        Y41_1x1
        );

    /* RTL MATCHING - STEP 4.1 */
    mulpool2(
        C41_1x1 * M41_1x1 * N41_1x1,
        Y41_1x1, 
        32.0/1.0, 
        Y41_1x1
        );

    {
        cnn_type_t * pd = Y41_1x1;
        for (int i = 0; i < M41_1x1 * N41_1x1; i++)
        {
            cnn_type_t s0 = exp(pd[0]);
            cnn_type_t s1 = exp(pd[1]);

            pd[0] = (s0) / (s0 + s1);
            pd[1] = (s1) / (s0 + s1);

            pd += 2;
        }
    }

    /* STAGE-4.2: CONV1x1 - PRELU */
    conv1x1(
        M03_3x3, 
        N03_3x3, 
        C03_3x3, 
        C42_1x1,
        Y03_3x3, 
        pnet_w42_1x1,
        Y42_1x1
        );

    addpoolbch(
        C42_1x1 * M42_1x1 * N42_1x1,
        C42_1x1 * 512,
        Y42_1x1,
        pnet_b42_1x1,
        Y42_1x1
        );

    /* RTL MATCHING - STEP 4.2 */
    mulpool2(
        C42_1x1 * M42_1x1 * N42_1x1,
        Y42_1x1, 
        8.0/1.0, 
        Y42_1x1
        );

    /* final step - generate box */
    generateBbox(
        M41_1x1, 
        N41_1x1,
        scale,
        Y41_1x1,
        Y42_1x1
        );
}

void Pnet::generateBbox(int M, int N, cnn_type_t scale, const cnn_type_t * psco, const cnn_type_t * ploc)
{
    //for pooling 
    int stride   =  2;
    int cellsize = 12;
    int count    =  0;

    for(int row = 0; row < M; row++)
    {
        for(int col = 0; col < N; col++)
        {
            if (psco[1] > Pthreshold)
            {
                _score[count] = psco[1];

                _x1   [count] = round((stride * row + 1) / scale);
                _y1   [count] = round((stride * col + 1) / scale);
                _x2   [count] = round((stride * row + 1 + cellsize) / scale);
                _y2   [count] = round((stride * col + 1 + cellsize) / scale);

                _area [count] = (_x2[count] - _x1[count]) * (_y2[count] - _y1[count]);
                
                _loc  [count * 4 + 0] = ploc[0];
                _loc  [count * 4 + 1] = ploc[1];
                _loc  [count * 4 + 2] = ploc[2];
                _loc  [count * 4 + 3] = ploc[3];

                count++;
            }

            psco += 2;
            ploc += 4;
        }
    }
    _nbox = count;
}

Rnet::Rnet()
{
    Rthreshold = 0.7;

    {
        M0 = 24;
        N0 = 24;
        C0 = 3;

        M01_3x3 = M0 - 3 + 1;
        N01_3x3 = N0 - 3 + 1;
        C01_3x3 = 28;

        M01_max3x3 = (M01_3x3 - 3 + 1) / 2 + 1;
        N01_max3x3 = (N01_3x3 - 3 + 1) / 2 + 1;
        C01_max3x3 = (C01_3x3    ) ;

        M02_3x3 = M01_max3x3 - 3 + 1;
        N02_3x3 = N01_max3x3 - 3 + 1;
        C02_3x3 = 48;

        M02_max3x3 = (M02_3x3 - 3 + 1) / 2 + 1;
        N02_max3x3 = (N02_3x3 - 3 + 1) / 2 + 1;
        C02_max3x3 = (C02_3x3    ) ;

        M03_2x2 = M02_max3x3 - 2 + 1;
        N03_2x2 = N02_max3x3 - 2 + 1;
        C03_2x2 = 64;

        M04_fc4 = 1;
        N04_fc4 = 1;
        C04_fc4 = 128;

        M51_scr = 1;
        N51_scr = 1;
        C51_scr = 2;

        M52_loc = 1;
        N52_loc = 1;
        C52_loc = 4;

        X0         = (cnn_type_t *)_aligned_malloc((M0         * N0         * C0         + 16) * sizeof(cnn_type_t), 16);
        
        Y01_3x3    = (cnn_type_t *)_aligned_malloc((M01_3x3    * N01_3x3    * C01_3x3    + 16) * sizeof(cnn_type_t), 16);
        Y01_max3x3 = (cnn_type_t *)_aligned_malloc((M01_max3x3 * N01_max3x3 * C01_max3x3 + 16) * sizeof(cnn_type_t), 16);

        Y02_3x3    = (cnn_type_t *)_aligned_malloc((M02_3x3    * N02_3x3    * C02_3x3    + 16) * sizeof(cnn_type_t), 16);
        Y02_max3x3 = (cnn_type_t *)_aligned_malloc((M02_max3x3 * N02_max3x3 * C02_max3x3 + 16) * sizeof(cnn_type_t), 16);

        Y03_2x2    = (cnn_type_t *)_aligned_malloc((M03_2x2    * N03_2x2    * C03_2x2    + 16) * sizeof(cnn_type_t), 16);
        Y04_fc4    = (cnn_type_t *)_aligned_malloc((M04_fc4    * N04_fc4    * C04_fc4    + 64) * sizeof(cnn_type_t), 16);

        Y51_scr    = Y04_fc4 + M04_fc4 * N04_fc4 * C04_fc4;
        Y52_loc    = Y51_scr + 8;
    }
}

Rnet::~Rnet()
{
    _aligned_free(X0        );

    _aligned_free(Y01_3x3   );
    _aligned_free(Y01_max3x3);

    _aligned_free(Y02_3x3   );
    _aligned_free(Y02_max3x3);

    _aligned_free(Y03_2x2   );
    _aligned_free(Y04_fc4   );
}

void Rnet::run(unsigned char * psrc)
{
    /* Mat -> input */
    {
        rgb2cnnpool(24 * 24 * 3, psrc, X0);
    }

    /* STAGE-1: CONV3x3 - PRELU - MAXPOOL2X2 */
    conv3x3(
        M0, 
        N0, 
        C0, 
        C01_3x3,
        X0, 
        rnet_w01_3x3,
        Y01_3x3
        );

    prelupool(
        C01_3x3 * M01_3x3 * N01_3x3,
        Y01_3x3,
        rnet_b01_3x3,
        rnet_p01_3x3,
        Y01_3x3
        );

    maxpool3x3(
        M01_3x3, 
        N01_3x3, 
        C01_3x3,
        2,

        Y01_3x3,
        Y01_max3x3
        );

    /* STAGE-2: CONV3x3 - PRELU - MAXPOOL2X2 */
    conv3x3(
        M01_max3x3, 
        N01_max3x3, 
        C01_max3x3, 
        C02_3x3,
        Y01_max3x3, 
        rnet_w02_3x3,
        Y02_3x3
        );

    prelupool(
        C02_3x3 * M02_3x3 * N02_3x3,
        Y02_3x3,
        rnet_b02_3x3,
        rnet_p02_3x3,
        Y02_3x3
        );

    maxpool3x3(
        M02_3x3, 
        N02_3x3, 
        C02_3x3,
        2,

        Y02_3x3,
        Y02_max3x3
        );

    /* STAGE-3: CONV2x2 - PRELU */
    conv2x2(
        M02_max3x3, 
        N02_max3x3, 
        C02_max3x3, 
        C03_2x2,
        Y02_max3x3, 
        rnet_w03_2x2,
        Y03_2x2
        );

    prelupool(
        C03_2x2 * M03_2x2 * N03_2x2,
        Y03_2x2,
        rnet_b03_2x2,
        rnet_p03_2x2,
        Y03_2x2
        );

    /* STAGE-4: FULLC */
    fullnet(
        M04_fc4 * N04_fc4 * C04_fc4, 
        M03_2x2 * N03_2x2 * C03_2x2,

        Y03_2x2, 
        rnet_w04_fc4, 
        Y04_fc4
    );

    prelupool(
        M04_fc4 * N04_fc4 * C04_fc4,

        Y04_fc4,
        rnet_b04_fc4,
        rnet_p04_fc4,
        Y04_fc4
        );

    /* STAGE-5-1: FULLC */
    fullnet(
        C51_scr,
        C04_fc4,

        Y04_fc4, 
        rnet_w51_scr, 
        Y51_scr
    );

    /* hand optimized for bias */
    Y51_scr[0] += rnet_b51_scr[0];
    Y51_scr[1] += rnet_b51_scr[1];

    /* softmax - score */
    cnn_type_t s0 = exp(Y51_scr[0]);
    cnn_type_t s1 = exp(Y51_scr[1]);

    Y51_scr[0] = s0 / (s0 + s1);
    Y51_scr[1] = s1 / (s0 + s1);

    /* STAGE-5-1: FULLC */
    fullnet(
        C52_loc,
        C04_fc4,

        Y04_fc4, 
        rnet_w52_loc, 
        Y52_loc
    );

    /* hand optimized for bias */
    Y52_loc[0] += rnet_b52_loc[0];
    Y52_loc[1] += rnet_b52_loc[1];
    Y52_loc[2] += rnet_b52_loc[2];
    Y52_loc[3] += rnet_b52_loc[3];
}

Onet::Onet()
{
    Othreshold = 0.8;

    {
        M0 = 48;
        N0 = 48;
        C0 = 3;

        M01_3x3 = M0 - 3 + 1;
        N01_3x3 = N0 - 3 + 1;
        C01_3x3 = 32;

        M01_max3x3 = (M01_3x3 - 3 + 1) / 2 + 1;
        N01_max3x3 = (N01_3x3 - 3 + 1) / 2 + 1;
        C01_max3x3 = (C01_3x3    ) ;

        M02_3x3 = M01_max3x3 - 3 + 1;
        N02_3x3 = N01_max3x3 - 3 + 1;
        C02_3x3 = 64;

        M02_max3x3 = (M02_3x3 - 3 + 1) / 2 + 1;
        N02_max3x3 = (N02_3x3 - 3 + 1) / 2 + 1;
        C02_max3x3 = (C02_3x3    ) ;

        M03_3x3 = M02_max3x3 - 3 + 1;
        N03_3x3 = N02_max3x3 - 3 + 1;
        C03_3x3 = 64;
        
        M03_max2x2 = (M03_3x3) / 2;
        N03_max2x2 = (N03_3x3) / 2;
        C03_max2x2 = (C03_3x3) ;

        M04_2x2 = M03_max2x2 - 2 + 1;
        N04_2x2 = N03_max2x2 - 2 + 1;
        C04_2x2 = 128;

        M05_fc5 = 1;
        N05_fc5 = 1;
        C05_fc5 = 256;

        M61_scr = 1;
        N61_scr = 1;
        C61_scr = 2;

        M62_loc = 1;
        N62_loc = 1;
        C62_loc = 4;

        M63_key = 1;
        N63_key = 1;
        C63_key = 10;

        X0         = (cnn_type_t *)_aligned_malloc((M0         * N0         * C0         + 16) * sizeof(cnn_type_t), 16);
        
        Y01_3x3    = (cnn_type_t *)_aligned_malloc((M01_3x3    * N01_3x3    * C01_3x3    + 16) * sizeof(cnn_type_t), 16);
        Y01_max3x3 = (cnn_type_t *)_aligned_malloc((M01_max3x3 * N01_max3x3 * C01_max3x3 + 16) * sizeof(cnn_type_t), 16);

        Y02_3x3    = (cnn_type_t *)_aligned_malloc((M02_3x3    * N02_3x3    * C02_3x3    + 16) * sizeof(cnn_type_t), 16);
        Y02_max3x3 = (cnn_type_t *)_aligned_malloc((M02_max3x3 * N02_max3x3 * C02_max3x3 + 16) * sizeof(cnn_type_t), 16);

        Y03_3x3    = (cnn_type_t *)_aligned_malloc((M03_3x3    * N03_3x3    * C03_3x3    + 16) * sizeof(cnn_type_t), 16);
        Y03_max2x2 = (cnn_type_t *)_aligned_malloc((M03_max2x2 * N03_max2x2 * C03_max2x2 + 16) * sizeof(cnn_type_t), 16);

        Y04_2x2    = (cnn_type_t *)_aligned_malloc((M04_2x2    * N04_2x2    * C04_2x2    + 16) * sizeof(cnn_type_t), 16);
        Y05_fc5    = (cnn_type_t *)_aligned_malloc((M05_fc5    * N05_fc5    * C05_fc5    + 64) * sizeof(cnn_type_t), 16);

        Y61_scr    = Y05_fc5 + M05_fc5 * N05_fc5 * C05_fc5 + 16;
        Y62_loc    = Y61_scr + 8;
        Y63_key    = Y62_loc + 8;
    }  
}

Onet::~Onet()
{
    _aligned_free(X0        );

    _aligned_free(Y01_3x3   );
    _aligned_free(Y01_max3x3);

    _aligned_free(Y02_3x3   );
    _aligned_free(Y02_max3x3);

    _aligned_free(Y03_3x3   );
    _aligned_free(Y03_max2x2);

    _aligned_free(Y04_2x2   );
    _aligned_free(Y05_fc5   );
}

void Onet::run(unsigned char * psrc)
{
    /* Mat -> input */
    {
        rgb2cnnpool(48 * 48 * 3, psrc, X0);
    }

    /* STAGE-1: CONV3x3 - PRELU - MAXPOOL2X2 */
    conv3x3(
        M0, 
        N0, 
        C0, 
        C01_3x3,
        X0, 
        onet_w01_3x3,
        Y01_3x3
        );

    prelupool2(
        C01_3x3 * M01_3x3 * N01_3x3,
        C01_3x3 * 512,
        Y01_3x3,
        onet_b01_3x3,
        onet_p01_3x3,
        Y01_3x3
        );

    maxpool3x3(
        M01_3x3, 
        N01_3x3, 
        C01_3x3,
        2,

        Y01_3x3,
        Y01_max3x3
        );

    /* STAGE-2: CONV3x3 - PRELU - MAXPOOL2X2 */
    conv3x3(
        M01_max3x3, 
        N01_max3x3, 
        C01_max3x3, 
        C02_3x3,
        Y01_max3x3, 
        onet_w02_3x3,
        Y02_3x3
        );

    prelupool(
        C02_3x3 * M02_3x3 * N02_3x3,
        Y02_3x3,
        onet_b02_3x3,
        onet_p02_3x3,
        Y02_3x3
        );

    maxpool3x3(
        M02_3x3, 
        N02_3x3, 
        C02_3x3,
        2,

        Y02_3x3,
        Y02_max3x3
        );

    /* STAGE-3: CONV3x3 - PRELU - MAXPOOL2X2 */
    conv3x3(
        M02_max3x3, 
        N02_max3x3, 
        C02_max3x3, 
        C03_3x3,
        Y02_max3x3, 
        onet_w03_3x3,
        Y03_3x3
        );

    prelupool(
        C03_3x3 * M03_3x3 * N03_3x3,
        Y03_3x3,
        onet_b03_3x3,
        onet_p03_3x3,
        Y03_3x3
        );

    maxpool2x2(
        M03_3x3, 
        N03_3x3, 
        C03_3x3,
        Y03_3x3,
        Y03_max2x2
        );

    /* STAGE-4: CONV2x2 - PRELU */
    conv2x2(
        M03_max2x2, 
        N03_max2x2, 
        C03_max2x2, 
        C04_2x2,
        Y03_max2x2, 
        onet_w04_2x2,
        Y04_2x2
        );

    prelupool(
        C04_2x2 * M04_2x2 * N04_2x2,
        Y04_2x2,
        onet_b04_2x2,
        onet_p04_2x2,
        Y04_2x2
        );

    /* STAGE-5: FULLC */
    fullnet(
        M05_fc5 * N05_fc5 * C05_fc5, 
        M04_2x2 * N04_2x2 * C04_2x2,

        Y04_2x2, 
        onet_w05_fc5, 
        Y05_fc5
    );

    prelupool(
        M05_fc5 * N05_fc5 * C05_fc5,

        Y05_fc5,
        onet_b05_fc5,
        onet_p05_fc5,
        Y05_fc5
        );

    /* STAGE-6-2: FULLC */
    fullnet(
        C61_scr,
        C05_fc5,

        Y05_fc5, 
        onet_w61_scr, 
        Y61_scr
    );

    /* hand optimized for bias */
    Y61_scr[0] += onet_b61_scr[0];
    Y61_scr[1] += onet_b61_scr[1];

    /* softmax - score */
    cnn_type_t s0 = exp(Y61_scr[0]);
    cnn_type_t s1 = exp(Y61_scr[1]);

    Y61_scr[0] = s0 / (s0 + s1);
    Y61_scr[1] = s1 / (s0 + s1);

    /* STAGE-6-2: FULLC */
    fullnet(
        C62_loc,
        C05_fc5,

        Y05_fc5, 
        onet_w62_loc, 
        Y62_loc
    );

    /* hand optimized for bias */
    Y62_loc[0] += onet_b62_loc[0];
    Y62_loc[1] += onet_b62_loc[1];
    Y62_loc[2] += onet_b62_loc[2];
    Y62_loc[3] += onet_b62_loc[3];

    /* STAGE-6-3: FULLC */
    fullnet(
        C63_key,
        C05_fc5,

        Y05_fc5, 
        onet_w63_key, 
        Y63_key
    );

    /* hand optimized for bias */
    Y63_key[0] += onet_b63_key[0];
    Y63_key[1] += onet_b63_key[1];
    Y63_key[2] += onet_b63_key[2];
    Y63_key[3] += onet_b63_key[3];
    Y63_key[4] += onet_b63_key[4];
    Y63_key[5] += onet_b63_key[5];
    Y63_key[6] += onet_b63_key[6];
    Y63_key[7] += onet_b63_key[7];
    Y63_key[8] += onet_b63_key[8];
    Y63_key[9] += onet_b63_key[9];
}

mtcnn::mtcnn(int row, int col)
{
    nms_threshold[0] = 0.7;
    nms_threshold[1] = 0.7;
    nms_threshold[2] = 0.7;

    float minl = row > col ? row : col;
    int MIN_DET_SIZE = 12;
    int minsize      = 36;
    float m          = (float)MIN_DET_SIZE / minsize;
    float factor     = 0.6853;

    _nscale = 0;
    while(minl > MIN_DET_SIZE)
    {
        _scales[_nscale++] = m;

        minl *= factor;
        m    *= factor;
    }

    #if (1)
    float minside = row < col ? row : col;
    for (int i = 0; i < _nscale; i++)
    {
        if (_scales[i] < (MIN_DET_SIZE / minside))
        {
            _nscale = i;
            break;
        }
    }
    #endif

    simpleFace_ = new Pnet[_nscale];

    _ipick   = (int32_t    *)_aligned_malloc((8192 * 4) * sizeof(score_t   ), 16);
    _pbufa   = (void       *)(_ipick + 8192);

    _loc_fst = (cnn_type_t *)_aligned_malloc(
        8192 * 2 * sizeof(cnn_type_t) + 
        8192 * 4 * sizeof(int16_t   ) + 
        8192 * 1 * sizeof(int32_t   ) , 16
        );
    _sc_fst  = (cnn_type_t *)(_loc_fst + 8192);
    _x1_fst  = (int16_t    *)(_sc_fst  + 8192);
    _y1_fst  = (int16_t    *)(_x1_fst  + 8192);
    _x2_fst  = (int16_t    *)(_y1_fst  + 8192);
    _y2_fst  = (int16_t    *)(_x2_fst  + 8192);
    _aa_fst  = (int32_t    *)(_y2_fst  + 8192);

    _loc_2nd = (cnn_type_t *)_aligned_malloc(
        4096 * 2 * sizeof(cnn_type_t) + 
        4096 * 4 * sizeof(int16_t   ) + 
        4096 * 1 * sizeof(int32_t   ) , 16
        );
    _sc_2nd  = (cnn_type_t *)(_loc_2nd + 4096);
    _x1_2nd  = (int16_t    *)(_sc_2nd  + 4096);
    _y1_2nd  = (int16_t    *)(_x1_2nd  + 4096);
    _x2_2nd  = (int16_t    *)(_y1_2nd  + 4096);
    _y2_2nd  = (int16_t    *)(_x2_2nd  + 4096);
    _aa_2nd  = (int32_t    *)(_y2_2nd  + 4096);

    _loc_3rd = (cnn_type_t *)_aligned_malloc(
        2048 * 12 * sizeof(cnn_type_t) + 
        2048 *  4 * sizeof(int16_t   ) + 
        2048 *  1 * sizeof(int32_t   ) , 16
        );
    _key_3rd = (cnn_type_t *)(_loc_3rd + 2048);
    _sc_3rd  = (cnn_type_t *)(_key_3rd + 20480);
    _x1_3rd  = (int16_t    *)(_sc_3rd  + 2048);
    _y1_3rd  = (int16_t    *)(_x1_3rd  + 2048);
    _x2_3rd  = (int16_t    *)(_y1_3rd  + 2048);
    _y2_3rd  = (int16_t    *)(_x2_3rd  + 2048);
    _aa_3rd  = (int32_t    *)(_y2_3rd  + 2048);

    /* resize pre */
    for (size_t i = 0; i < _nscale; i++) 
    {
        int changedH = (int)ceil(row * _scales[i]);
        int changedW = (int)ceil(col * _scales[i]);

        /* aligned at 2 */
        changedH = (changedH + 1) / 2 * 2;
        changedW = (changedW + 1) / 2 * 2;

        _interpcb[i] = bilinearInterpolationRGBSetup(
            row, col,
            changedH, changedW
            );

        _images[i] = (uint8_t *)_aligned_malloc(sizeof(uint8_t) * changedH * changedW * 4 + 64, 16);
    }
}

mtcnn::~mtcnn()
{
    delete []simpleFace_;

    _aligned_free(_ipick  );
    _aligned_free(_loc_fst);
    _aligned_free(_loc_2nd);
    _aligned_free(_loc_3rd);

    for (size_t i = 0; i < _nscale; i++) 
    {
        bilinearInterpolationRGBDestroy(_interpcb[i]);
        _aligned_free(_images[i]);
    }
}

static int compareScore(const void * _pa, const void * _pb)
{
    const score_t * pa = (const score_t *)(_pa);
    const score_t * pb = (const score_t *)(_pb);

    cnn_type_t d = pa->score - pb->score;

    return ((d < 0) ? (-1) : ((d > 0) ? (+1) : (0)));
}

int _nms2(          
          int       n, 

          float     fa,

    const int16_t * x1,
    const int16_t * y1,
    const int16_t * x2,
    const int16_t * y2,
    const int32_t * aa,

    const cnn_type_t * sc,

          int32_t * ipick,

          void    * pbufa
)
{
    int npick;
    int last;
    int i, j;

    score_t * pscore = (score_t *)(pbufa);
    int16_t * supre  = (int16_t *)(pscore + (n+15)/16*16);

    for (int i = 0; i < n; i++)
    {
        pscore[i].score = sc[i];
        pscore[i].index = i;

        supre[i] = 0;
    }
    qsort(pscore, n, sizeof(score_t), compareScore);

    npick = 0;
    last  = n;

    while (last > 0)
    {
        i = pscore[last-1].index;
        
        ipick[npick] = i;
        npick++;

        supre[last-1] = 1;

        for (int pos = 0; pos < last; pos++)
        {
            j = pscore[pos].index;

            int maxX = (x1[i] > x1[j]) ? x1[i] : x1[j];
            int maxY = (y1[i] > y1[j]) ? y1[i] : y1[j];
            int minX = (x2[i] < x2[j]) ? x2[i] : x2[j];
            int minY = (y2[i] < y2[j]) ? y2[i] : y2[j];

            //maxX1 and maxY1 reuse 
            int w = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
            int h = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
            
            float a = (float)(w * h);
            float b = ((float)(aa[i] + aa[j] - a)) * fa;

            supre[pos] = (a > b);
        }

        /* remove supressed items */
        int remain = 0;
        for (int pos = 0; pos < last; pos++)
        {
            if (!supre[pos])
            {
                pscore[remain].index = pscore[pos].index;
                remain++;
            }

            supre[pos] = 0;
        }

        last = remain;
    }

    return (npick);
}

int _nms3(          
          int       n, 

          float     fa,

    const int16_t * x1,
    const int16_t * y1,
    const int16_t * x2,
    const int16_t * y2,
    const int32_t * aa,

    const cnn_type_t * sc,

          int32_t * ipick,

          void    * pbufa
)
{
    int npick;
    int last;
    int i, j;

    score_t * pscore = (score_t *)(pbufa);
    int16_t * supre  = (int16_t *)(pscore + (n+15)/16*16);

    for (int i = 0; i < n; i++)
    {
        pscore[i].score = sc[i];
        pscore[i].index = i;

        supre[i] = 0;
    }
    qsort(pscore, n, sizeof(score_t), compareScore);

    npick = 0;
    last  = n;

    while (last > 0)
    {
        i = pscore[last-1].index;
        
        ipick[npick] = i;
        npick++;

        supre[last-1] = 1;

        for (int pos = 0; pos < last; pos++)
        {
            j = pscore[pos].index;

            int maxX = (x1[i] > x1[j]) ? x1[i] : x1[j];
            int maxY = (y1[i] > y1[j]) ? y1[i] : y1[j];
            int minX = (x2[i] < x2[j]) ? x2[i] : x2[j];
            int minY = (y2[i] < y2[j]) ? y2[i] : y2[j];

            //maxX1 and maxY1 reuse 
            int w = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
            int h = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
            
            float a = (float)(w * h);
            float b = ((float)((aa[i] > aa[j]) ? (aa[j]) : (aa[i]))) * fa;

            supre[pos] = (a > b);
        }

        /* remove supressed items */
        int remain = 0;
        for (int pos = 0; pos < last; pos++)
        {
            if (!supre[pos])
            {
                pscore[remain].index = pscore[pos].index;
                remain++;
            }

            supre[pos] = 0;
        }

        last = remain;
    }

    return (npick);
}

void refineAndSquareBbox(
          int       height,
          int       width,

          int       npick,
    const int32_t * ipick,

          int16_t * _x1,
          int16_t * _y1,
          int16_t * _x2,
          int16_t * _y2,
          int32_t * _aa,

    const cnn_type_t * _loc
) 
{
    if(npick <= 0){return;};

    float bbw = 0, bbh = 0, maxSide = 0;
    float h   = 0, w   = 0;
    float x1  = 0, y1  = 0, x2 = 0, y2 = 0;

    for (int i = 0; i < npick; i++) 
    {
        int j = (ipick == NULL) ? (i) : (ipick[i]);

        {
            bbh = _x2[j] - _x1[j] + 1;
            bbw = _y2[j] - _y1[j] + 1;
            x1  = _x1[j] + _loc[j * 4 + 1] * bbh;
            y1  = _y1[j] + _loc[j * 4 + 0] * bbw;
            x2  = _x2[j] + _loc[j * 4 + 3] * bbh;
            y2  = _y2[j] + _loc[j * 4 + 2] * bbw;

            h = x2 - x1 + 1;
            w = y2 - y1 + 1;

            maxSide = (h > w) ? h : w;
            x1 = x1 + h * 0.5 - maxSide * 0.5;
            y1 = y1 + w * 0.5 - maxSide * 0.5;
            _x2[j] = round(x1 + maxSide - 1);
            _y2[j] = round(y1 + maxSide - 1);
            _x1[j] = round(x1);
            _y1[j] = round(y1);

            //boundary check
            if (_x1[j] < 0){_x1[j] = 0;};
            if (_y1[j] < 0){_y1[j] = 0;};
            if (_x2[j] > height){_x2[j] = height - 1;};
            if (_y2[j] > width ){_y2[j] = width  - 1;};

            _aa[j] = (_x2[j] - _x1[j]) * (_y2[j] - _y1[j]);
        }
    }
}

void mtcnn::findFace(uint8_t * pimage, bbox_chain_t * pbbox)
{
    int count   = 0;
    pbbox->nbox = 0;

    // uint8_t * pimage = image.data;

    for (size_t i = 0; i < _nscale; i++) 
    {
        int changedH = (int)ceil(MTCNN_IMGH * _scales[i]);
        int changedW = (int)ceil(MTCNN_IMGW * _scales[i]);

        /* aligned at 2 */
        changedH = (changedH + 1) / 2 * 2;
        changedW = (changedW + 1) / 2 * 2;

        bilinearInterpolationRGBExecute(
            _interpcb[i], _images[i], pimage
            );

        simpleFace_[i].run(changedH, changedW, (_images[i]), _scales[i]);

        _npick = _nms2(
           (simpleFace_[i]._nbox), 
           (simpleFace_[i].nms_threshold),
           (simpleFace_[i]._x1), 
           (simpleFace_[i]._y1), 
           (simpleFace_[i]._x2), 
           (simpleFace_[i]._y2), 
           (simpleFace_[i]._area),
           (simpleFace_[i]._score), 
           _ipick,
           _pbufa
           ); 

        for (int j = 0; j < _npick; j++)
        {
            int pos = _ipick[j];

            _x1_fst[count] = simpleFace_[i]._x1[pos];
            _y1_fst[count] = simpleFace_[i]._y1[pos];
            _x2_fst[count] = simpleFace_[i]._x2[pos];
            _y2_fst[count] = simpleFace_[i]._y2[pos];

            _aa_fst[count] = simpleFace_[i]._area[pos];
            _sc_fst[count] = simpleFace_[i]._score[pos];

            memcpy(&_loc_fst[count * 4], &simpleFace_[i]._loc[pos * 4], 4 * sizeof(cnn_type_t));

            count++;
        }
    }

    //the first stage's nms
    if (count < 1)
    {
        return;
    }

    {
        _npick = _nms2(
           count, 
           nms_threshold[0],
           _x1_fst, 
           _y1_fst, 
           _x2_fst, 
           _y2_fst, 
           _aa_fst,
           _sc_fst, 
           _ipick,
           _pbufa
           ); 

        refineAndSquareBbox(
            MTCNN_IMGH, 
            MTCNN_IMGW,
            _npick,
            _ipick,
           (_x1_fst), 
           (_y1_fst), 
           (_x2_fst), 
           (_y2_fst), 
           (_aa_fst),
           (_loc_fst)
           );
    }

    // printf("Rnet: %d out of %d\n", _npick, count);

    //second stage
    count = 0;
    for (int j = 0; j < _npick; j++)
    {
        int pos = _ipick[j];

        {
            uint8_t _rnetImgData[24 * 24 * 4];

            bilinearInterpolationRGB24(
                _rnetImgData,
                pimage + _x1_fst[pos] * MTCNN_IMGW * 3 + _y1_fst[pos] * 3,
                _x2_fst[pos] - _x1_fst[pos],
                _y2_fst[pos] - _y1_fst[pos],
                MTCNN_IMGW * 3, 3
            );

            refineNet.run(_rnetImgData);

            if (*(refineNet.Y51_scr+1) > refineNet.Rthreshold)
            {
                _x1_2nd[count] = _x1_fst[pos];
                _y1_2nd[count] = _y1_fst[pos];
                _x2_2nd[count] = _x2_fst[pos];
                _y2_2nd[count] = _y2_fst[pos];

                _sc_2nd[count] = refineNet.Y51_scr[1];
                memcpy(&_loc_2nd[count * 4], refineNet.Y52_loc, 4 * sizeof(cnn_type_t));

                _aa_2nd[count] = (_x2_2nd[count] - _x1_2nd[count])*(_y2_2nd[count] - _y1_2nd[count]);
                count++;
            }
        }
    }

    if(count<1) return;

    {
        _npick = _nms2(
           count, 
           nms_threshold[1],
           _x1_2nd, 
           _y1_2nd, 
           _x2_2nd, 
           _y2_2nd, 
           _aa_2nd,
           _sc_2nd, 
           _ipick,
           _pbufa
           ); 

        refineAndSquareBbox(
            MTCNN_IMGH, 
            MTCNN_IMGW,
            _npick,
            _ipick,
            _x1_2nd, 
            _y1_2nd, 
            _x2_2nd, 
            _y2_2nd, 
            _aa_2nd,
            _loc_2nd
           );
    }

    // printf("Onet: %d out of %d\n", _npick, count);

    //third stage 
    count = 0;
    for (int j = 0; j < _npick; j++)
    {
        int pos = _ipick[j];

        {
            uint8_t _onetImgData[48 * 48 * 4];

            bilinearInterpolationRGB48(
                _onetImgData,
                pimage + _x1_2nd[pos] * MTCNN_IMGW * 3 + _y1_2nd[pos] * 3,
                _x2_2nd[pos] - _x1_2nd[pos],
                _y2_2nd[pos] - _y1_2nd[pos],
                MTCNN_IMGW * 3, 3
            );
            
            outNet.run(_onetImgData);

            if(*(outNet.Y61_scr+1) > outNet.Othreshold)
            {
                _x1_3rd[count] = _x1_2nd[pos];
                _y1_3rd[count] = _y1_2nd[pos];
                _x2_3rd[count] = _x2_2nd[pos];
                _y2_3rd[count] = _y2_2nd[pos];
                memcpy(&_loc_3rd[count * 4], outNet.Y62_loc, 4 * sizeof(cnn_type_t));
                for(int num=0; num < 5; num++)
                {
                    (_key_3rd)[count * 10 + num + 0] = _y1_3rd[count] + (_y2_3rd[count] - _y1_3rd[count]) * (outNet.Y63_key[num + 0]);
                }
                for(int num=0; num < 5; num++)
                {
                    (_key_3rd)[count * 10 + num + 5] = _x1_3rd[count] + (_x2_3rd[count] - _x1_3rd[count]) * (outNet.Y63_key[num + 5]);
                }
                _aa_3rd[count] = (_x2_3rd[count] - _x1_3rd[count])*(_y2_3rd[count] - _y1_3rd[count]);
                _sc_3rd[count] = *(outNet.Y61_scr+1);
                count++;
            }
        }
    }

    if (count<1)
    {
        return;
    }

    {
        refineAndSquareBbox(
            MTCNN_IMGH, MTCNN_IMGW,
            count, NULL,

            _x1_3rd, 
            _y1_3rd, 
            _x2_3rd, 
            _y2_3rd, 
            _aa_3rd,
            _loc_3rd
           );

        _npick = _nms3(
           count, 
           nms_threshold[2],
           _x1_3rd, 
           _y1_3rd, 
           _x2_3rd, 
           _y2_3rd, 
           _aa_3rd,
           _sc_3rd, 
           _ipick,
           _pbufa
           );
    }
    
    ////////////////////////////////////////////////
    /* collect detected bbox                      */ 
    ////////////////////////////////////////////////
    pbbox->nbox = _npick;
    for (int j = 0; j < _npick; j++)
    {
        int pos = _ipick[j];

        pbbox->bbox[j].l = _y1_3rd[pos];
        pbbox->bbox[j].t = _x1_3rd[pos];

        pbbox->bbox[j].r = _y2_3rd[pos];
        pbbox->bbox[j].b = _x2_3rd[pos];
    }
    ///////////////////////////////////////////////
}

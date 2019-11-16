#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <assert.h>

#include "cnntype.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define mulpool2_08(xloc, wloc, yloc)  \
{  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = (xloc[p]) * wloc;};  \
   \
   xloc += CNN_BCHSIZ;  \
   yloc += CNN_BCHSIZ;  \
}

#define mulpool2_16(xloc, wloc, yloc) \
{  \
   mulpool2_08(xloc, wloc, yloc);  \
   mulpool2_08(xloc, wloc, yloc);  \
}

#define mulpool2_32(xloc, wloc, yloc) \
{  \
   mulpool2_16(xloc, wloc, yloc);  \
   mulpool2_16(xloc, wloc, yloc);  \
}

#define mulpool2_64(xloc, wloc, yloc) \
{  \
   mulpool2_32(xloc, wloc, yloc);  \
   mulpool2_32(xloc, wloc, yloc);  \
}

#define mulpool2_128(xloc, wloc, yloc) \
{  \
   mulpool2_64(xloc, wloc, yloc);  \
   mulpool2_64(xloc, wloc, yloc);  \
}

#define mulpool2_256(xloc, wloc, yloc) \
{  \
   mulpool2_128(xloc, wloc, yloc);  \
   mulpool2_128(xloc, wloc, yloc);  \
}

#define mulpool2_512(xloc, wloc, yloc) \
{  \
   mulpool2_256(xloc, wloc, yloc);  \
   mulpool2_256(xloc, wloc, yloc);  \
}

static void _mulpool2_00(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   ;
}

static void _mulpool2_1536(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   mulpool2_512(xloc, wloc, yloc);
   mulpool2_512(xloc, wloc, yloc);
   mulpool2_512(xloc, wloc, yloc);
}

static void _mulpool2_1024(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   mulpool2_512(xloc, wloc, yloc);
   mulpool2_512(xloc, wloc, yloc);
}

static void _mulpool2_512(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   mulpool2_512(xloc, wloc, yloc);
}

static void _mulpool2_384(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   mulpool2_128(xloc, wloc, yloc);
   mulpool2_128(xloc, wloc, yloc);
   mulpool2_128(xloc, wloc, yloc);
}

static void _mulpool2_256(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   mulpool2_128(xloc, wloc, yloc);
   mulpool2_128(xloc, wloc, yloc);
}

static void _mulpool2_128(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   mulpool2_128(xloc, wloc, yloc);
}

static void _mulpool2_96(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   mulpool2_64(xloc, wloc, yloc);
   mulpool2_32(xloc, wloc, yloc);
}

static void _mulpool2_64(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   mulpool2_64(xloc, wloc, yloc);
}

static void _mulpool2_32(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   mulpool2_32(xloc, wloc, yloc);
}

static void _mulpool2_24(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   mulpool2_16(xloc, wloc, yloc);
   mulpool2_08(xloc, wloc, yloc);
}

static void _mulpool2_16(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   mulpool2_16(xloc, wloc, yloc);
}

static void _mulpool2_08(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   mulpool2_08(xloc, wloc, yloc);
}

static void _mulpool2_07(
   const cnn_type_t * restrict xloc,
         cnn_type_t            wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 7; i++){yloc[i] = (xloc[i] * wloc);};
}

static void _mulpool2_06(
   const cnn_type_t * restrict xloc,
         cnn_type_t            wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 6; i++){yloc[i] = (xloc[i] * wloc);};
}

static void _mulpool2_05(
   const cnn_type_t * restrict xloc,
         cnn_type_t            wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 5; i++){yloc[i] = (xloc[i] * wloc);};
}

static void _mulpool2_04(
   const cnn_type_t * restrict xloc,
         cnn_type_t            wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 4; i++){yloc[i] = (xloc[i] * wloc);};
}

static void _mulpool2_03(
   const cnn_type_t * restrict xloc,
         cnn_type_t            wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 3; i++){yloc[i] = (xloc[i] * wloc);};
}

static void _mulpool2_02(
   const cnn_type_t * restrict xloc,
         cnn_type_t            wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 2; i++){yloc[i] = (xloc[i] * wloc);};
}

static void _mulpool2_01(
   const cnn_type_t * restrict xloc,
         cnn_type_t            wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 1; i++){yloc[i] = (xloc[i] * wloc);};
}


typedef void (*mulpool2_func_t)(const cnn_type_t *, cnn_type_t, cnn_type_t *);
static mulpool2_func_t _mulpool2_tab_512[] = {
   _mulpool2_00,
   _mulpool2_512,
   _mulpool2_1024,
   _mulpool2_1536,
};

static mulpool2_func_t _mulpool2_tab_128[] = {
   _mulpool2_00,
   _mulpool2_128,
   _mulpool2_256,
   _mulpool2_384,
};

static mulpool2_func_t _mulpool2_tab_32[] = {
   _mulpool2_00,
   _mulpool2_32,
   _mulpool2_64,
   _mulpool2_96,
};

static mulpool2_func_t _mulpool2_tab_08[] = {
   _mulpool2_00,
   _mulpool2_08,
   _mulpool2_16,
   _mulpool2_24,
};

static mulpool2_func_t _mulpool2_tab_00[] = {
   _mulpool2_00,
   _mulpool2_01,
   _mulpool2_02,
   _mulpool2_03,
   _mulpool2_04,
   _mulpool2_05,
   _mulpool2_06,
   _mulpool2_07,
};

void mulpool2(
         int                   _n,
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_n >= (1 << 11))
   {
      int nb = (_n >> 11);
      
      for (int i = 0; i < nb; i++){
         (_mulpool2_tab_512[2])(xloc, wloc, yloc);
         xloc += (1 << 10);
         yloc += (1 << 10);

         (_mulpool2_tab_512[2])(xloc, wloc, yloc);
         xloc += (1 << 10);
         yloc += (1 << 10);
      }
      _n -= (nb << 11);
   }

   if (_n >= (1 << 9))
   {
      int nb = (_n >> 9);
      (_mulpool2_tab_512[nb & 3])(xloc, wloc, yloc);

      xloc += (nb << 9);
      yloc += (nb << 9);
      _n   -= (nb << 9);
   }

   if (_n >= (1 << 7))
   {
      int nb = (_n >> 7);
      (_mulpool2_tab_128[nb])(xloc, wloc, yloc);

      xloc += (nb << 7);
      yloc += (nb << 7);
      _n   -= (nb << 7);
   }

   if (_n >= (1 << 5))
   {
      int nb = (_n >> 5);
      (_mulpool2_tab_32[nb])(xloc, wloc, yloc);

      xloc += (nb << 5);
      yloc += (nb << 5);
      _n   -= (nb << 5);
   }

   if (_n >= (1 << 3))
   {
      int nb = (_n >> 3);
      (_mulpool2_tab_08[nb])(xloc, wloc, yloc);

      xloc += (nb << 3);
      yloc += (nb << 3);
      _n   -= (nb << 3);
   }

   /* 0 - 7 */
   {
      (_mulpool2_tab_00[_n])(xloc, wloc, yloc);
   }
}

#define macpool_08(xloc, wloc, yloc)  \
{  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] += (xloc[p]) * (wloc[p]);};  \
   \
   xloc += CNN_BCHSIZ;  \
   wloc += CNN_BCHSIZ;  \
}

#define macpool_16(xloc, wloc, yloc) \
{  \
   macpool_08(xloc, wloc, yloc);  \
   macpool_08(xloc, wloc, yloc);  \
}

#define macpool_32(xloc, wloc, yloc) \
{  \
   macpool_16(xloc, wloc, yloc);  \
   macpool_16(xloc, wloc, yloc);  \
}

#define macpool_64(xloc, wloc, yloc) \
{  \
   macpool_32(xloc, wloc, yloc);  \
   macpool_32(xloc, wloc, yloc);  \
}

#define macpool_128(xloc, wloc, yloc) \
{  \
   macpool_64(xloc, wloc, yloc);  \
   macpool_64(xloc, wloc, yloc);  \
}

#define macpool_256(xloc, wloc, yloc) \
{  \
   macpool_128(xloc, wloc, yloc);  \
   macpool_128(xloc, wloc, yloc);  \
}

#define macpool_512(xloc, wloc, yloc) \
{  \
   macpool_256(xloc, wloc, yloc);  \
   macpool_256(xloc, wloc, yloc);  \
}

static void _macpool_00(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   ;
}

static void _macpool_1536(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_512(xloc, wloc, yloc);
   macpool_512(xloc, wloc, yloc);
   macpool_512(xloc, wloc, yloc);
}

static void _macpool_1024(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_512(xloc, wloc, yloc);
   macpool_512(xloc, wloc, yloc);
}

static void _macpool_512(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_512(xloc, wloc, yloc);
}

static void _macpool_384(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_128(xloc, wloc, yloc);
   macpool_128(xloc, wloc, yloc);
   macpool_128(xloc, wloc, yloc);
}

static void _macpool_256(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_128(xloc, wloc, yloc);
   macpool_128(xloc, wloc, yloc);
}

static void _macpool_128(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_128(xloc, wloc, yloc);
}

static void _macpool_96(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_64(xloc, wloc, yloc);
   macpool_32(xloc, wloc, yloc);
}

static void _macpool_64(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_64(xloc, wloc, yloc);
}

static void _macpool_32(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_32(xloc, wloc, yloc);
}

static void _macpool_24(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_16(xloc, wloc, yloc);
   macpool_08(xloc, wloc, yloc);
}

static void _macpool_16(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_16(xloc, wloc, yloc);
}

static void _macpool_08(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_08(xloc, wloc, yloc);
}

typedef void (*macpool_func_t)(const cnn_type_t *, const cnn_type_t *, acc_type_t *);
static macpool_func_t _macpool_tab_512[] = {
   _macpool_00,
   _macpool_512,
   _macpool_1024,
   _macpool_1536,
};

static macpool_func_t _macpool_tab_128[] = {
   _macpool_00,
   _macpool_128,
   _macpool_256,
   _macpool_384,
};

static macpool_func_t _macpool_tab_32[] = {
   _macpool_00,
   _macpool_32,
   _macpool_64,
   _macpool_96,
};

static macpool_func_t _macpool_tab_08[] = {
   _macpool_00,
   _macpool_08,
   _macpool_16,
   _macpool_24,
};

cnn_type_t macpool(
         int                   _n,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);

   acc_type_t _yloc[CNN_BCHSIZ + 64]; // = {0};
   
   uint64_t     mask = 31;
   acc_type_t * yloc = (acc_type_t *)(((uint64_t)(_yloc) + mask) & (~mask));

   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = 0;};

   if (_n >= (1 << 11))
   {
      int nb = (_n >> 11);
      
      for (int i = 0; i < nb; i++){
         (_macpool_tab_512[2])(xloc, wloc, yloc);
         xloc += (1 << 10);
         wloc += (1 << 10);

         (_macpool_tab_512[2])(xloc, wloc, yloc);
         xloc += (1 << 10);
         wloc += (1 << 10);
      }
      _n -= (nb << 11);
   }

   if (_n >= (1 << 9))
   {
      int nb = (_n >> 9);
      (_macpool_tab_512[nb & 3])(xloc, wloc, yloc);

      xloc += (nb << 9);
      wloc += (nb << 9);
      _n   -= (nb << 9);
   }

   if (_n >= (1 << 7))
   {
      int nb = (_n >> 7);
      (_macpool_tab_128[nb])(xloc, wloc, yloc);

      xloc += (nb << 7);
      wloc += (nb << 7);
      _n   -= (nb << 7);
   }

   if (_n >= (1 << 5))
   {
      int nb = (_n >> 5);
      (_macpool_tab_32[nb])(xloc, wloc, yloc);

      xloc += (nb << 5);
      wloc += (nb << 5);
      _n   -= (nb << 5);
   }

   if (_n >= (1 << 3))
   {
      int nb = (_n >> 3);
      (_macpool_tab_08[nb])(xloc, wloc, yloc);

      xloc += (nb << 3);
      wloc += (nb << 3);
      _n   -= (nb << 3);
   }

   {
      for (int p = 0; p < (CNN_BCHSIZ/ 2); p++){yloc[p] += yloc[p + (CNN_BCHSIZ/ 2)];};
      for (int p = 0; p < (CNN_BCHSIZ/ 4); p++){yloc[p] += yloc[p + (CNN_BCHSIZ/ 4)];};
      for (int p = 0; p < (CNN_BCHSIZ/ 8); p++){yloc[p] += yloc[p + (CNN_BCHSIZ/ 8)];};

#if (CNN_BCHSIZ >= 16)
      for (int p = 0; p < (CNN_BCHSIZ/16); p++){yloc[p] += yloc[p + (CNN_BCHSIZ/16)];};
#endif
   }

   return (yloc[0]); // ((yloc[0] >> 15) + ((yloc[0] >> 14) & 1));
}

void conv3x3(
         int  M,
         int  N,
         int  C1,
         int  C2,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
         cnn_type_t * restrict __py
)
{
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);
   cnn_type_t * _pw = __builtin_assume_aligned(__pw, 16);

   int localbuf_depth = (CNN_BCHSIZ * ((C1+CNN_BCHSIZ-1)/CNN_BCHSIZ));

   cnn_type_t * localxbuf = (cnn_type_t *)_aligned_malloc(3 * 3 * localbuf_depth *  N * sizeof(cnn_type_t), 16);
   cnn_type_t * localwbuf = _pw; // (cnn_type_t *)aligned_alloc(16, 3 * 3 * localbuf_depth * C2 * sizeof(cnn_type_t));

   /* initialize localbuf */
   memset(localxbuf, 0, sizeof(cnn_type_t) * 3 * 3 * localbuf_depth * N );
   // memset(localwbuf, 0, sizeof(cnn_type_t) * 3 * 3 * localbuf_depth * C2);


   #if (0)
   /* reorganize w buffer (transposed) */
   for (int c = 0; c < C2; c++)
   {
      cnn_type_t * ploc;
      cnn_type_t * srcd; 

      /* point - (0, 0) */
      ploc = localwbuf + (3 * 3 * localbuf_depth) * c;
      srcd = _pw       + (3 * 3 * C1            ) * c;
      for (int p = 0; p < C1; p++){ploc[localbuf_depth * 0 + p] = srcd[p];}

      /* point - (1, 0) */
      srcd = srcd + C1;
      for (int p = 0; p < C1; p++){ploc[localbuf_depth * 3 + p] = srcd[p];};

      /* point - (2, 0) */
      srcd = srcd + C1;
      for (int p = 0; p < C1; p++){ploc[localbuf_depth * 6 + p] = srcd[p];};

      /* point - (0, 1) */
      srcd = srcd + C1;
      for (int p = 0; p < C1; p++){ploc[localbuf_depth * 1 + p] = srcd[p];};

      /* point - (1, 1) */
      srcd = srcd + C1;
      for (int p = 0; p < C1; p++){ploc[localbuf_depth * 4 + p] = srcd[p];};

      /* point - (2, 1) */
      srcd = srcd + C1;
      for (int p = 0; p < C1; p++){ploc[localbuf_depth * 7 + p] = srcd[p];};

      /* point - (0, 2) */
      srcd = srcd + C1;
      for (int p = 0; p < C1; p++){ploc[localbuf_depth * 2 + p] = srcd[p];};

      /* point - (1, 2) */
      srcd = srcd + C1;
      for (int p = 0; p < C1; p++){ploc[localbuf_depth * 5 + p] = srcd[p];};

      /* point - (2, 2) */
      srcd = srcd + C1;
      for (int p = 0; p < C1; p++){ploc[localbuf_depth * 8 + p] = srcd[p];};
   }
   #endif

   for (int m = 0; m < M - 3 + 1; m++)
   {
      cnn_type_t * px = _px + m * (N-0) * C1;
      cnn_type_t * py = _py + m * (N-2) * C2;

      {
         int n = 0;

         cnn_type_t * xloc = localxbuf + n * 3 * localbuf_depth;
         /* read (3x3*C1) source plane into local memory */
         {
            cnn_type_t * srcd;

            /* point - (0, 0) */
            srcd = px;
            for (int p = 0; p < C1; p++){xloc[localbuf_depth * 0 + p] = srcd[p];};

            /* point - (1, 0) */
            srcd = srcd + C1;
            for (int p = 0; p < C1; p++){xloc[localbuf_depth * 3 + p] = srcd[p];};

            /* point - (2, 0) */
            srcd = srcd + C1;
            for (int p = 0; p < C1; p++){xloc[localbuf_depth * 6 + p] = srcd[p];};

            /* point - (0, 1) */
            srcd = px + N * C1;
            for (int p = 0; p < C1; p++){xloc[localbuf_depth * 1 + p] = srcd[p];};

            /* point - (1, 1) */
            srcd = srcd + C1;
            for (int p = 0; p < C1; p++){xloc[localbuf_depth * 4 + p] = srcd[p];};

            /* point - (2, 1) */
            srcd = srcd + C1;
            for (int p = 0; p < C1; p++){xloc[localbuf_depth * 7 + p] = srcd[p];};

            /* point - (0, 2) */
            srcd = px + 2 * N * C1;
            for (int p = 0; p < C1; p++){xloc[localbuf_depth * 2 + p] = srcd[p];};

            /* point - (1, 2) */
            srcd = srcd + C1;
            for (int p = 0; p < C1; p++){xloc[localbuf_depth * 5 + p] = srcd[p];};

            /* point - (2, 2) */
            srcd = srcd + C1;
            for (int p = 0; p < C1; p++){xloc[localbuf_depth * 8 + p] = srcd[p];};
         }

         {
            cnn_type_t * wloc = localwbuf;

            for (int c = 0; c < C2; c++)
            {
               *py++ = macpool(3 * 3 * localbuf_depth, xloc, wloc); wloc += (3 * 3 * localbuf_depth);
            }
         }

         px += C1;
      }

      for (int n = 1; n < N - 3 + 1; n++)
      {
         cnn_type_t * xloc = localxbuf + n * 3 * localbuf_depth;

         /* read (3x3*C1) source plane into local memory */
         {
            cnn_type_t * srcd;

            /* point - (2, 0) */
            srcd = px + (0 * N + 2) * C1;
            for (int p = 0; p < C1; p++){xloc[localbuf_depth * 6 + p] = srcd[p];};

            /* point - (2, 1) */
            srcd = px + (1 * N + 2) * C1;
            for (int p = 0; p < C1; p++){xloc[localbuf_depth * 7 + p] = srcd[p];};

            /* point - (2, 2) */
            srcd = px + (2 * N + 2) * C1;
            for (int p = 0; p < C1; p++){xloc[localbuf_depth * 8 + p] = srcd[p];};
         }

         {
            cnn_type_t * wloc = localwbuf;
            for (int c = 0; c < C2; c++)
            {
               *py++ = macpool(3 * 3 * localbuf_depth, xloc, wloc); wloc += (3 * 3 * localbuf_depth);
            }
         }

         px += C1;
      }
   }

   _aligned_free(localxbuf);
   // _aligned_free(localwbuf);
}

void conv2x2(
         int  M,
         int  N,
         int  C1,
         int  C2,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
         cnn_type_t * restrict __py
)
{
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);
   cnn_type_t * _pw = __builtin_assume_aligned(__pw, 16);

   int localbuf_depth = (CNN_BCHSIZ * ((C1+CNN_BCHSIZ-1)/CNN_BCHSIZ));

   cnn_type_t * localxbuf = (cnn_type_t *)_aligned_malloc(2 * 2 * localbuf_depth *  N * sizeof(cnn_type_t), 16);
   cnn_type_t * localwbuf = _pw; // (cnn_type_t *)aligned_alloc(16, 2 * 2 * localbuf_depth * C2 * sizeof(cnn_type_t));

   /* initialize localbuf */
   memset(localxbuf, 0, sizeof(cnn_type_t) * 2 * 2 * localbuf_depth * N );
   
   #if (0)
   memset(localwbuf, 0, sizeof(cnn_type_t) * 2 * 2 * localbuf_depth * C2);

   /* reorganize w buffer (transposed) */
   for (int c = 0; c < C2; c++)
   {
      cnn_type_t * ploc;
      cnn_type_t * srcd; 

      /* point - (0, 0) */
      ploc = localwbuf + (2 * 2 * localbuf_depth) * c;
      srcd = _pw       + (2 * 2 * C1            ) * c;
      for (int p = 0; p < C1; p++){ploc[localbuf_depth * 0 + p] = srcd[p];}

      /* point - (1, 0) */
      srcd = srcd + C1;
      for (int p = 0; p < C1; p++){ploc[localbuf_depth * 2 + p] = srcd[p];};

      /* point - (0, 1) */
      srcd = srcd + C1;
      for (int p = 0; p < C1; p++){ploc[localbuf_depth * 1 + p] = srcd[p];};

      /* point - (1, 1) */
      srcd = srcd + C1;
      for (int p = 0; p < C1; p++){ploc[localbuf_depth * 3 + p] = srcd[p];};
   }
   #endif

   for (int m = 0; m < M - 2 + 1; m++)
   {
      cnn_type_t * px = _px + m * (N-0) * C1;
      cnn_type_t * py = _py + m * (N-1) * C2;

      {
         int n = 0;

         cnn_type_t * xloc = localxbuf + n * 2 * localbuf_depth;

         /* read (2x2*C1) source plane into local memory (transposed, column major) */
         {
            cnn_type_t * srcd;

            /* point - (0, 0) */
            srcd = px;
            for (int p = 0; p < C1; p++){xloc[localbuf_depth * 0 + p] = srcd[p];};

            /* point - (1, 0) */
            srcd = srcd + C1;
            for (int p = 0; p < C1; p++){xloc[localbuf_depth * 2 + p] = srcd[p];};

            /* point - (0, 1) */
            srcd = px + N * C1;
            for (int p = 0; p < C1; p++){xloc[localbuf_depth * 1 + p] = srcd[p];};

            /* point - (1, 1) */
            srcd = srcd + C1;
            for (int p = 0; p < C1; p++){xloc[localbuf_depth * 3 + p] = srcd[p];};
         }

         {
            cnn_type_t * wloc = localwbuf;

            for (int c = 0; c < C2; c++)
            {
               *py++ = macpool(2 * 2 * localbuf_depth, xloc, wloc); wloc += (2 * 2 * localbuf_depth);
            }
         }

         px += C1;
      }

      for (int n = 1; n < N - 2 + 1; n++)
      {
         cnn_type_t * xloc = localxbuf + n * 2 * localbuf_depth;

         /* read (2x2*C1) source plane into local memory (transposed, column major) */
         {
            cnn_type_t * srcd;

            /* point - (1, 0) */
            srcd = px + (0 * N + 1) * C1;
            for (int p = 0; p < C1; p++){xloc[localbuf_depth * 2 + p] = srcd[p];};

            /* point - (1, 1) */
            srcd = px + (1 * N + 1) * C1;
            for (int p = 0; p < C1; p++){xloc[localbuf_depth * 3 + p] = srcd[p];};
         }

         {
            cnn_type_t * wloc = localwbuf;

            for (int c = 0; c < C2; c++)
            {
               *py++ = macpool(2 * 2 * localbuf_depth, xloc, wloc); wloc += (2 * 2 * localbuf_depth);
            }
         }

         px += C1;
      }
   }

   _aligned_free(localxbuf);
   /* _aligned_free(localwbuf); */
}

void conv1x1(
         int  M,
         int  N,
         int  C1,
         int  C2,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
         cnn_type_t * restrict __py
)
{
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);
   cnn_type_t * _pw = __builtin_assume_aligned(__pw, 16);

   int localbuf_depth = (CNN_BCHSIZ * ((C1+CNN_BCHSIZ-1)/CNN_BCHSIZ));

   cnn_type_t * localxbuf = (cnn_type_t *)_aligned_malloc(1 * 1 * localbuf_depth *  N * sizeof(cnn_type_t), 16);
   cnn_type_t * localwbuf = _pw; // (cnn_type_t *)aligned_alloc(16, 1 * 1 * localbuf_depth * C2 * sizeof(cnn_type_t));

   /* initialize localbuf */
   memset(localxbuf, 0, sizeof(cnn_type_t) * 1 * 1 * localbuf_depth * N );

   #if (0)
   memset(localwbuf, 0, sizeof(cnn_type_t) * 1 * 1 * localbuf_depth * C2);

   /* reorganize w buffer (transposed) */
   for (int c = 0; c < C2; c++)
   {
      cnn_type_t * ploc;
      cnn_type_t * srcd; 

      /* point - (0, 0) */
      ploc = localwbuf + (1 * 1 * localbuf_depth) * c;
      srcd = _pw       + (1 * 1 * C1            ) * c;
      for (int p = 0; p < C1; p++){ploc[localbuf_depth * 0 + p] = srcd[p];};
   }
   #endif

   for (int m = 0; m < M - 1 + 1; m++)
   {
      cnn_type_t * px = _px + m * (N-0) * C1;
      cnn_type_t * py = _py + m * (N-0) * C2;

      for (int n = 0; n < (N-1+1); n++)
      {
         cnn_type_t * xloc = localxbuf;

         /* read (1x1*C1) source plane into local memory */
         {
            cnn_type_t * srcd;

            /* point - (0, 0) */
            srcd = px;
            for (int p = 0; p < C1; p++){xloc[localbuf_depth * 0 + p] = srcd[p];};
         }

         {
            cnn_type_t * wloc = localwbuf;
            for (int c = 0; c < C2; c++)
            {
               *py++ = macpool(1 * 1 * localbuf_depth, xloc, wloc); wloc += (1 * 1 * localbuf_depth);
            }
         }

         px += C1;
      }
   }

   _aligned_free(localxbuf);
   /* _aligned_free(localwbuf); */
}

////////////////////////////////////////////////////////////////////////////////////////////
#define addpool_08(xloc, wloc, yloc)  \
{  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = (xloc[p]) + (wloc[p]);};  \
   \
   xloc += CNN_BCHSIZ;  \
   wloc += CNN_BCHSIZ;  \
   yloc += CNN_BCHSIZ;  \
}

#define addpool_16(xloc, wloc, yloc) \
{  \
   addpool_08(xloc, wloc, yloc);  \
   addpool_08(xloc, wloc, yloc);  \
}

#define addpool_32(xloc, wloc, yloc) \
{  \
   addpool_16(xloc, wloc, yloc);  \
   addpool_16(xloc, wloc, yloc);  \
}

#define addpool_64(xloc, wloc, yloc) \
{  \
   addpool_32(xloc, wloc, yloc);  \
   addpool_32(xloc, wloc, yloc);  \
}

#define addpool_128(xloc, wloc, yloc) \
{  \
   addpool_64(xloc, wloc, yloc);  \
   addpool_64(xloc, wloc, yloc);  \
}

#define addpool_256(xloc, wloc, yloc) \
{  \
   addpool_128(xloc, wloc, yloc);  \
   addpool_128(xloc, wloc, yloc);  \
}

#define addpool_512(xloc, wloc, yloc) \
{  \
   addpool_256(xloc, wloc, yloc);  \
   addpool_256(xloc, wloc, yloc);  \
}

static void _addpool_00(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   ;
}

static void _addpool_1536(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_512(xloc, wloc, yloc);
   addpool_512(xloc, wloc, yloc);
   addpool_512(xloc, wloc, yloc);
}

static void _addpool_1024(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_512(xloc, wloc, yloc);
   addpool_512(xloc, wloc, yloc);
}

static void _addpool_512(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_512(xloc, wloc, yloc);
}

static void _addpool_384(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_128(xloc, wloc, yloc);
   addpool_128(xloc, wloc, yloc);
   addpool_128(xloc, wloc, yloc);
}

static void _addpool_256(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_128(xloc, wloc, yloc);
   addpool_128(xloc, wloc, yloc);
}

static void _addpool_128(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_128(xloc, wloc, yloc);
}

static void _addpool_96(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_64(xloc, wloc, yloc);
   addpool_32(xloc, wloc, yloc);
}

static void _addpool_64(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_64(xloc, wloc, yloc);
}

static void _addpool_32(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_32(xloc, wloc, yloc);
}

static void _addpool_24(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_16(xloc, wloc, yloc);
   addpool_08(xloc, wloc, yloc);
}

static void _addpool_16(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_16(xloc, wloc, yloc);
}

static void _addpool_08(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_08(xloc, wloc, yloc);
}

typedef void (*addpool_func_t)(const cnn_type_t *, const cnn_type_t *, cnn_type_t *);
static addpool_func_t _addpool_tab_512[] = {
   _addpool_00,
   _addpool_512,
   _addpool_1024,
   _addpool_1536,
};

static addpool_func_t _addpool_tab_128[] = {
   _addpool_00,
   _addpool_128,
   _addpool_256,
   _addpool_384,
};

static addpool_func_t _addpool_tab_32[] = {
   _addpool_00,
   _addpool_32,
   _addpool_64,
   _addpool_96,
};

static addpool_func_t _addpool_tab_08[] = {
   _addpool_00,
   _addpool_08,
   _addpool_16,
   _addpool_24,
};

void addpool(
         int                   _n,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_n >= (1 << 11))
   {
      int nb = (_n >> 11);
      
      for (int i = 0; i < nb; i++){
         (_addpool_tab_512[2])(xloc, wloc, yloc);
         xloc += (1 << 10);
         wloc += (1 << 10);
         yloc += (1 << 10);

         (_addpool_tab_512[2])(xloc, wloc, yloc);
         xloc += (1 << 10);
         wloc += (1 << 10);
         yloc += (1 << 10);
      }
      _n -= (nb << 11);
   }

   if (_n >= (1 << 9))
   {
      int nb = (_n >> 9);
      (_addpool_tab_512[nb & 3])(xloc, wloc, yloc);

      xloc += (nb << 9);
      wloc += (nb << 9);
      yloc += (nb << 9);
      _n   -= (nb << 9);
   }

   if (_n >= (1 << 7))
   {
      int nb = (_n >> 7);
      (_addpool_tab_128[nb])(xloc, wloc, yloc);

      xloc += (nb << 7);
      wloc += (nb << 7);
      yloc += (nb << 7);
      _n   -= (nb << 7);
   }

   if (_n >= (1 << 5))
   {
      int nb = (_n >> 5);
      (_addpool_tab_32[nb])(xloc, wloc, yloc);

      xloc += (nb << 5);
      wloc += (nb << 5);
      yloc += (nb << 5);
      _n   -= (nb << 5);
   }

   if (_n >= (1 << 3))
   {
      int nb = (_n >> 3);
      (_addpool_tab_08[nb])(xloc, wloc, yloc);

      xloc += (nb << 3);
      wloc += (nb << 3);
      yloc += (nb << 3);
      _n   -= (nb << 3);
   }

   for (int i = 0; i < _n; i++)
   {
      yloc[i] = xloc[i] + wloc[i];
   }
}

void addpoolbch(
         int                   _n,
         int                   _bchm,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
){
   while (_n > _bchm)
   {
      addpool(
         _bchm, 
         _xloc, 
         _wloc, 
         _yloc
         );

      _xloc += _bchm;
      _yloc += _bchm;
      _n    -= _bchm;
   }

   /* rest-part */
   {
      addpool(
         _n, 
         _xloc, 
         _wloc,
         _yloc
         );
   }
}

/////////////////////////////////////////////////////////////////////////////////////
cnn_type_t macpool(
         int                   _n,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc
);

/* N%4 = 0 */
void fullnet(
         int                   M,
         int                   N,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc   
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = _yloc;

   for (int m = 0; m < M; m++)
   {
      *yloc++ = macpool(N, xloc, wloc);

      wloc += N;
   }
}

/////////////////////////////////////////////////////////////////////////////////////
#define prelupool_08(xloc, bloc, ploc, yloc)   \
{  \
   cnn_type_t uloc[CNN_BCHSIZ];          \
   acc_type_t wloc[CNN_BCHSIZ];          \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      uloc[p] = xloc[p] + bloc[p];       \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      wloc[p] = uloc[p] * ploc[p];       \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      yloc[p] = (uloc[p] < 0) ? (wloc[p]) : (uloc[p]);  \
   }                                     \
                                         \
   xloc += CNN_BCHSIZ;                   \
   bloc += CNN_BCHSIZ;                   \
   ploc += CNN_BCHSIZ;                   \
   yloc += CNN_BCHSIZ;                   \
}

#define prelupool_16(xloc, bloc, ploc, yloc) \
{  \
   prelupool_08(xloc, bloc, ploc, yloc);  \
   prelupool_08(xloc, bloc, ploc, yloc);  \
}

#define prelupool_32(xloc, bloc, ploc, yloc) \
{  \
   prelupool_16(xloc, bloc, ploc, yloc);  \
   prelupool_16(xloc, bloc, ploc, yloc);  \
}

#define prelupool_64(xloc, bloc, ploc, yloc) \
{  \
   prelupool_32(xloc, bloc, ploc, yloc);  \
   prelupool_32(xloc, bloc, ploc, yloc);  \
}

#define prelupool_128(xloc, bloc, ploc, yloc) \
{  \
   prelupool_64(xloc, bloc, ploc, yloc);  \
   prelupool_64(xloc, bloc, ploc, yloc);  \
}

#define prelupool_256(xloc, bloc, ploc, yloc) \
{  \
   prelupool_128(xloc, bloc, ploc, yloc);  \
   prelupool_128(xloc, bloc, ploc, yloc);  \
}

#define prelupool_512(xloc, bloc, ploc, yloc) \
{  \
   prelupool_256(xloc, bloc, ploc, yloc);  \
   prelupool_256(xloc, bloc, ploc, yloc);  \
}

static void _prelupool_00(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
)
{
   ;
}

static void _prelupool_1536(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * ploc = __builtin_assume_aligned(_ploc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   prelupool_512(xloc, bloc, ploc, yloc);
   prelupool_512(xloc, bloc, ploc, yloc);
   prelupool_512(xloc, bloc, ploc, yloc);
}

static void _prelupool_1024(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * ploc = __builtin_assume_aligned(_ploc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   prelupool_512(xloc, bloc, ploc, yloc);
   prelupool_512(xloc, bloc, ploc, yloc);
}

static void _prelupool_512(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * ploc = __builtin_assume_aligned(_ploc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   prelupool_512(xloc, bloc, ploc, yloc);
}

static void _prelupool_384(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * ploc = __builtin_assume_aligned(_ploc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   prelupool_128(xloc, bloc, ploc, yloc);
   prelupool_128(xloc, bloc, ploc, yloc);
   prelupool_128(xloc, bloc, ploc, yloc);
}

static void _prelupool_256(  
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * ploc = __builtin_assume_aligned(_ploc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   prelupool_128(xloc, bloc, ploc, yloc);
   prelupool_128(xloc, bloc, ploc, yloc);
}

static void _prelupool_128(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * ploc = __builtin_assume_aligned(_ploc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   prelupool_128(xloc, bloc, ploc, yloc);
}

static void _prelupool_96(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * ploc = __builtin_assume_aligned(_ploc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   prelupool_64(xloc, bloc, ploc, yloc);
   prelupool_32(xloc, bloc, ploc, yloc);
}

static void _prelupool_64(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * ploc = __builtin_assume_aligned(_ploc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   prelupool_64(xloc, bloc, ploc, yloc);
}

static void _prelupool_32(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * ploc = __builtin_assume_aligned(_ploc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   prelupool_32(xloc, bloc, ploc, yloc);
}

static void _prelupool_24(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * ploc = __builtin_assume_aligned(_ploc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   prelupool_16(xloc, bloc, ploc, yloc);
   prelupool_08(xloc, bloc, ploc, yloc);
}

static void _prelupool_16(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * ploc = __builtin_assume_aligned(_ploc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   prelupool_16(xloc, bloc, ploc, yloc);
}

static void _prelupool_08(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * ploc = __builtin_assume_aligned(_ploc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   prelupool_08(xloc, bloc, ploc, yloc);
}

typedef void (*prelupool_func_t)(const cnn_type_t *, const cnn_type_t *, const cnn_type_t *, cnn_type_t *);
static prelupool_func_t _prelupool_tab_512[] = {
   _prelupool_00,
   _prelupool_512,
   _prelupool_1024,
   _prelupool_1536,
};

static prelupool_func_t _prelupool_tab_128[] = {
   _prelupool_00,
   _prelupool_128,
   _prelupool_256,
   _prelupool_384,
};

static prelupool_func_t _prelupool_tab_32[] = {
   _prelupool_00,
   _prelupool_32,
   _prelupool_64,
   _prelupool_96,
};

static prelupool_func_t _prelupool_tab_08[] = {
   _prelupool_00,
   _prelupool_08,
   _prelupool_16,
   _prelupool_24,
};

void prelupool(
         int                   _n,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * ploc = __builtin_assume_aligned(_ploc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_n >= (1 << 11))
   {
      int nb = (_n >> 11);
      
      for (int i = 0; i < nb; i++)
      {
         (_prelupool_tab_512[2])(xloc, bloc, ploc, yloc);
         xloc += (1 << 10);
         bloc += (1 << 10);
         ploc += (1 << 10);
         yloc += (1 << 10);

         (_prelupool_tab_512[2])(xloc, bloc, ploc, yloc);
         xloc += (1 << 10);
         bloc += (1 << 10);
         ploc += (1 << 10);
         yloc += (1 << 10);
      }
      _n -= (nb << 11);
   }

   if (_n >= (1 << 9))
   {
      int nb = (_n >> 9);

      (_prelupool_tab_512[nb])(xloc, bloc, ploc, yloc);
      xloc += (nb << 9);
      bloc += (nb << 9);
      ploc += (nb << 9);
      yloc += (nb << 9);
      _n   -= (nb << 9);
   }

   if (_n >= (1 << 7))
   {
      int nb = (_n >> 7);

      (_prelupool_tab_128[nb])(xloc, bloc, ploc, yloc);
      xloc += (nb << 7);
      bloc += (nb << 7);
      ploc += (nb << 7);
      yloc += (nb << 7);
      _n   -= (nb << 7);
   }

   if (_n >= (1 << 5))
   {
      int nb = (_n >> 5);

      (_prelupool_tab_32[nb])(xloc, bloc, ploc, yloc);
      xloc += (nb << 5);
      bloc += (nb << 5);
      ploc += (nb << 5);
      yloc += (nb << 5);
      _n   -= (nb << 5);
   }

   if (_n >= (1 << 3))
   {
      int nb = (_n >> 3);
      (_prelupool_tab_08[nb])(xloc, bloc, ploc, yloc);

      xloc += (nb << 3);
      bloc += (nb << 3);
      ploc += (nb << 3);
      yloc += (nb << 3);
      _n   -= (nb << 3);
   }

   for (int i = 0; i < _n; i++)
   {
      cnn_type_t uloc[1];
      acc_type_t wloc[1];

      for (int p = 0; p < 1; p++)
      {
         uloc[p] = xloc[p] + bloc[p];
      }

      for (int p = 0; p < 1; p++)
      {
         wloc[p] = uloc[p] * ploc[p];
      }

      for (int p = 0; p < 1; p++)
      {
         yloc[p] = (uloc[p] < 0) ? (wloc[p]) : (uloc[p]);
      }
   }
}

void prelupool2(
         int                   _n,
         int                   _bchm,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
){
   while (_n > _bchm)
   {
      prelupool(
         _bchm, 
         _xloc, 
         _bloc, 
         _ploc, 
         _yloc
         );

      _xloc += _bchm;
      _yloc += _bchm;
      _n    -= _bchm;
   }

   /* reset part */
   {
      prelupool(
         _n, 
         _xloc, 
         _bloc, 
         _ploc, 
         _yloc
         );
   }
}

/////////////////////////////////////////////////////////////////////////////////////
#define relupool_08(xloc, bloc, yloc)   \
{  \
   cnn_type_t uloc[CNN_BCHSIZ];          \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      uloc[p] = xloc[p] + bloc[p];       \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      yloc[p] = (uloc[p] > 0) ? (uloc[p]) : (0.0);  \
   }                                     \
                                         \
   xloc += CNN_BCHSIZ;                   \
   bloc += CNN_BCHSIZ;                   \
   yloc += CNN_BCHSIZ;                   \
}

#define relupool_16(xloc, bloc, yloc) \
{  \
   relupool_08(xloc, bloc, yloc);  \
   relupool_08(xloc, bloc, yloc);  \
}

#define relupool_32(xloc, bloc, yloc) \
{  \
   relupool_16(xloc, bloc, yloc);  \
   relupool_16(xloc, bloc, yloc);  \
}

#define relupool_64(xloc, bloc, yloc) \
{  \
   relupool_32(xloc, bloc, yloc);  \
   relupool_32(xloc, bloc, yloc);  \
}

#define relupool_128(xloc, bloc, yloc) \
{  \
   relupool_64(xloc, bloc, yloc);  \
   relupool_64(xloc, bloc, yloc);  \
}

#define relupool_256(xloc, bloc, yloc) \
{  \
   relupool_128(xloc, bloc, yloc);  \
   relupool_128(xloc, bloc, yloc);  \
}

#define relupool_512(xloc, bloc, yloc) \
{  \
   relupool_256(xloc, bloc, yloc);  \
   relupool_256(xloc, bloc, yloc);  \
}

static void _relupool_00(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   ;
}

static void _relupool_1536(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   relupool_512(xloc, bloc, yloc);
   relupool_512(xloc, bloc, yloc);
   relupool_512(xloc, bloc, yloc);
}

static void _relupool_1024(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   relupool_512(xloc, bloc, yloc);
   relupool_512(xloc, bloc, yloc);
}

static void _relupool_512(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   relupool_512(xloc, bloc, yloc);
}

static void _relupool_384(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   relupool_128(xloc, bloc, yloc);
   relupool_128(xloc, bloc, yloc);
   relupool_128(xloc, bloc, yloc);
}

static void _relupool_256(  
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   relupool_128(xloc, bloc, yloc);
   relupool_128(xloc, bloc, yloc);
}

static void _relupool_128(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   relupool_128(xloc, bloc, yloc);
}

static void _relupool_96(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   relupool_64(xloc, bloc, yloc);
   relupool_32(xloc, bloc, yloc);
}

static void _relupool_64(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   relupool_64(xloc, bloc, yloc);
}

static void _relupool_32(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   relupool_32(xloc, bloc, yloc);
}

static void _relupool_24(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   relupool_16(xloc, bloc, yloc);
   relupool_08(xloc, bloc, yloc);
}

static void _relupool_16(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   relupool_16(xloc, bloc, yloc);
}

static void _relupool_08(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   relupool_08(xloc, bloc, yloc);
}

typedef void (*relupool_func_t)(const cnn_type_t *, const cnn_type_t *, cnn_type_t *);
static relupool_func_t _relupool_tab_512[] = {
   _relupool_00,
   _relupool_512,
   _relupool_1024,
   _relupool_1536,
};

static relupool_func_t _relupool_tab_128[] = {
   _relupool_00,
   _relupool_128,
   _relupool_256,
   _relupool_384,
};

static relupool_func_t _relupool_tab_32[] = {
   _relupool_00,
   _relupool_32,
   _relupool_64,
   _relupool_96,
};

static relupool_func_t _relupool_tab_08[] = {
   _relupool_00,
   _relupool_08,
   _relupool_16,
   _relupool_24,
};

void relupool(
         int                   _n,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_n >= (1 << 11))
   {
      int nb = (_n >> 11);
      
      for (int i = 0; i < nb; i++)
      {
         (_relupool_tab_512[2])(xloc, bloc, yloc);
         xloc += (1 << 10);
         bloc += (1 << 10);
         yloc += (1 << 10);

         (_relupool_tab_512[2])(xloc, bloc, yloc);
         xloc += (1 << 10);
         bloc += (1 << 10);
         yloc += (1 << 10);
      }
      _n   -= (nb << 11);
   }

   if (_n >= (1 << 9))
   {
      int nb = (_n >> 9);
      (_relupool_tab_512[nb & 3])(xloc, bloc, yloc);

      xloc += (nb << 9);
      bloc += (nb << 9);
      yloc += (nb << 9);
      _n   -= (nb << 9);
   }

   if (_n >= (1 << 7))
   {
      int nb = (_n >> 7);
      (_relupool_tab_128[nb])(xloc, bloc, yloc);

      xloc += (nb << 7);
      bloc += (nb << 7);
      yloc += (nb << 7);
      _n   -= (nb << 7);
   }

   if (_n >= (1 << 5))
   {
      int nb = (_n >> 5);
      (_relupool_tab_32[nb])(xloc, bloc, yloc);

      xloc += (nb << 5);
      bloc += (nb << 5);
      yloc += (nb << 5);
      _n   -= (nb << 5);
   }

   if (_n >= (1 << 3))
   {
      int nb = (_n >> 3);
      (_relupool_tab_08[nb])(xloc, bloc, yloc);

      xloc += (nb << 3);
      bloc += (nb << 3);
      yloc += (nb << 3);
      _n   -= (nb << 3);
   }

   for (int i = 0; i < _n; i++)
   {
      cnn_type_t uloc[1];

      for (int p = 0; p < 1; p++){uloc[p] = xloc[p] + bloc[p];                };
      for (int p = 0; p < 1; p++){yloc[p] = (uloc[p] > 0) ? (uloc[p]) : (0.0);};
   }
}

/////////////////////////////////////////////////////////////////////////////////////
#define rgb2cnnpool_08(xloc, yloc)       \
{  \
   cnn_type_t uloc[CNN_BCHSIZ];          \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      uloc[p] = xloc[p] - 127.5;         \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      yloc[p] = uloc[p] * 0.0078125;     \
   }                                     \
                                         \
   xloc += CNN_BCHSIZ;                   \
   yloc += CNN_BCHSIZ;                   \
}

#define rgb2cnnpool_16(xloc, yloc) \
{  \
   rgb2cnnpool_08(xloc, yloc);  \
   rgb2cnnpool_08(xloc, yloc);  \
}

#define rgb2cnnpool_32(xloc, yloc) \
{  \
   rgb2cnnpool_16(xloc, yloc);  \
   rgb2cnnpool_16(xloc, yloc);  \
}

#define rgb2cnnpool_64(xloc, yloc) \
{  \
   rgb2cnnpool_32(xloc, yloc);  \
   rgb2cnnpool_32(xloc, yloc);  \
}

#define rgb2cnnpool_128(xloc, yloc) \
{  \
   rgb2cnnpool_64(xloc, yloc);  \
   rgb2cnnpool_64(xloc, yloc);  \
}

#define rgb2cnnpool_256(xloc, yloc) \
{  \
   rgb2cnnpool_128(xloc, yloc);  \
   rgb2cnnpool_128(xloc, yloc);  \
}

#define rgb2cnnpool_512(xloc, yloc) \
{  \
   rgb2cnnpool_256(xloc, yloc);  \
   rgb2cnnpool_256(xloc, yloc);  \
}

static void _rgb2cnnpool_00(
   const unsigned char * restrict _xloc,
         cnn_type_t    * restrict _yloc
)
{
   ;
}

static void _rgb2cnnpool_1536(
   const unsigned char * restrict _xloc,
         cnn_type_t    * restrict _yloc
)
{
   uint8_t    * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   rgb2cnnpool_512(xloc, yloc);
   rgb2cnnpool_512(xloc, yloc);
   rgb2cnnpool_512(xloc, yloc);
}

static void _rgb2cnnpool_1024(
   const unsigned char * restrict _xloc,
         cnn_type_t    * restrict _yloc
)
{
   uint8_t    * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   rgb2cnnpool_512(xloc, yloc);
   rgb2cnnpool_512(xloc, yloc);
}

static void _rgb2cnnpool_512(
   const unsigned char * restrict _xloc,
         cnn_type_t    * restrict _yloc
)
{
   uint8_t    * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   rgb2cnnpool_512(xloc, yloc);
}

static void _rgb2cnnpool_384(
   const unsigned char * restrict _xloc,
         cnn_type_t    * restrict _yloc
)
{
   uint8_t    * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   rgb2cnnpool_128(xloc, yloc);
   rgb2cnnpool_128(xloc, yloc);
   rgb2cnnpool_128(xloc, yloc);
}

static void _rgb2cnnpool_256(
   const unsigned char * restrict _xloc,
         cnn_type_t    * restrict _yloc
)
{
   uint8_t    * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   rgb2cnnpool_128(xloc, yloc);
   rgb2cnnpool_128(xloc, yloc);
}

static void _rgb2cnnpool_128(
   const unsigned char * restrict _xloc,
         cnn_type_t    * restrict _yloc
)
{
   uint8_t    * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   rgb2cnnpool_128(xloc, yloc);
}

static void _rgb2cnnpool_96(
   const unsigned char * restrict _xloc,
         cnn_type_t    * restrict _yloc
)
{
   uint8_t    * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   rgb2cnnpool_64(xloc, yloc);
   rgb2cnnpool_32(xloc, yloc);
}

static void _rgb2cnnpool_64(
   const unsigned char * restrict _xloc,
         cnn_type_t    * restrict _yloc
)
{
   uint8_t    * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   rgb2cnnpool_64(xloc, yloc);
}

static void _rgb2cnnpool_32(
   const unsigned char * restrict _xloc,
         cnn_type_t    * restrict _yloc
)
{
   uint8_t    * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   rgb2cnnpool_32(xloc, yloc);
}

static void _rgb2cnnpool_24(
   const unsigned char * restrict _xloc,
         cnn_type_t    * restrict _yloc
)
{
   uint8_t    * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   rgb2cnnpool_16(xloc, yloc);
   rgb2cnnpool_08(xloc, yloc);
}

static void _rgb2cnnpool_16(
   const unsigned char * restrict _xloc,
         cnn_type_t    * restrict _yloc
)
{
   uint8_t    * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   rgb2cnnpool_16(xloc, yloc);
}

static void _rgb2cnnpool_08(
   const unsigned char * restrict _xloc,
         cnn_type_t    * restrict _yloc
)
{
   uint8_t    * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   rgb2cnnpool_08(xloc, yloc);
}

typedef void (*rgb2cnnpool_func_t)(const unsigned char *, cnn_type_t *);
static rgb2cnnpool_func_t _rgb2cnnpool_tab_512[] = {
   _rgb2cnnpool_00,
   _rgb2cnnpool_512,
   _rgb2cnnpool_1024,
   _rgb2cnnpool_1536,
};

static rgb2cnnpool_func_t _rgb2cnnpool_tab_128[] = {
   _rgb2cnnpool_00,
   _rgb2cnnpool_128,
   _rgb2cnnpool_256,
   _rgb2cnnpool_384,
};

static rgb2cnnpool_func_t _rgb2cnnpool_tab_32[] = {
   _rgb2cnnpool_00,
   _rgb2cnnpool_32,
   _rgb2cnnpool_64,
   _rgb2cnnpool_96,
};

static rgb2cnnpool_func_t _rgb2cnnpool_tab_08[] = {
   _rgb2cnnpool_00,
   _rgb2cnnpool_08,
   _rgb2cnnpool_16,
   _rgb2cnnpool_24,
};

void rgb2cnnpool(
         int                      _n,
   const unsigned char * restrict _xloc,
         cnn_type_t    * restrict _yloc
)
{
   uint8_t    * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_n >= (1 << 11))
   {
      int nb = (_n >> 11);
      
      for (int i = 0; i < nb; i++)
      {
         (_rgb2cnnpool_tab_512[2])(xloc, yloc);
         xloc += (1 << 10);
         yloc += (1 << 10);

         (_rgb2cnnpool_tab_512[2])(xloc, yloc);
         xloc += (1 << 10);
         yloc += (1 << 10);
      }
      _n -= (nb << 11);
   }

   if (_n >= (1 << 9))
   {
      int nb = (_n >> 9);

      (_rgb2cnnpool_tab_512[nb])(xloc, yloc);
      xloc += (nb << 9);
      yloc += (nb << 9);
      _n   -= (nb << 9);
   }

   if (_n >= (1 << 7))
   {
      int nb = (_n >> 7);

      (_rgb2cnnpool_tab_128[nb])(xloc, yloc);
      xloc += (nb << 7);
      yloc += (nb << 7);
      _n   -= (nb << 7);
   }

   if (_n >= (1 << 5))
   {
      int nb = (_n >> 5);

      (_rgb2cnnpool_tab_32[nb])(xloc, yloc);
      xloc += (nb << 5);
      yloc += (nb << 5);
      _n   -= (nb << 5);
   }

   if (_n >= (1 << 3))
   {
      int nb = (_n >> 3);
      (_rgb2cnnpool_tab_08[nb])(xloc, yloc);

      xloc += (nb << 3);
      yloc += (nb << 3);
      _n   -= (nb << 3);
   }

   for (int i = 0; i < _n; i++)
   {
      cnn_type_t uloc[1];

      for (int p = 0; p < 1; p++)
      {
         uloc[p] = xloc[p] - 127.5;
      }

      for (int p = 0; p < 1; p++)
      {
         yloc[p] = uloc[p] * 0.0078125;
      }
   }
}

/////////////////////////////////////////////////////////////////////////////////////
#define maxpool2x2_08(_Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = xloc[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Lx +       p]) ? (wloc[p]) : (xloc[_Lx +       p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln +       p]) ? (wloc[p]) : (xloc[_Ln +       p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Lx + p]) ? (wloc[p]) : (xloc[_Ln + _Lx + p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = wloc[p];};  \
   xloc += CNN_BCHSIZ; \
   yloc += CNN_BCHSIZ; \
}

#define maxpool2x2_16(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool2x2_08(_Lx, _Ln, xloc, yloc) \
   maxpool2x2_08(_Lx, _Ln, xloc, yloc) \
}

#define maxpool2x2_32(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool2x2_16(_Lx, _Ln, xloc, yloc) \
   maxpool2x2_16(_Lx, _Ln, xloc, yloc) \
}

#define maxpool2x2_64(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool2x2_32(_Lx, _Ln, xloc, yloc) \
   maxpool2x2_32(_Lx, _Ln, xloc, yloc) \
}

#define maxpool2x2_128(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool2x2_64(_Lx, _Ln, xloc, yloc) \
   maxpool2x2_64(_Lx, _Ln, xloc, yloc) \
}

void _maxpool2x2_00(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
}

void _maxpool2x2_96(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x2_64(_Lx, _Ln, xloc, yloc);
   maxpool2x2_32(_Lx, _Ln, xloc, yloc);   
}

void _maxpool2x2_64(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x2_64(_Lx, _Ln, xloc, yloc);
}

void _maxpool2x2_32(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x2_32(_Lx, _Ln, xloc, yloc);
}

void _maxpool2x2_24(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x2_16(_Lx, _Ln, xloc, yloc);
   maxpool2x2_08(_Lx, _Ln, xloc, yloc);   
}

void _maxpool2x2_16(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x2_16(_Lx, _Ln, xloc, yloc);
}

void _maxpool2x2_08(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x2_08(_Lx, _Ln, xloc, yloc);
}

typedef void (*maxpool2x2_func_t)(int, int, const cnn_type_t *, cnn_type_t *);
static maxpool2x2_func_t maxpool2x2_fntab_32[] = {
   _maxpool2x2_00,
   _maxpool2x2_32,
   _maxpool2x2_64,
   _maxpool2x2_96,
};

static maxpool2x2_func_t maxpool2x2_fntab_08[] = {
   _maxpool2x2_00,
   _maxpool2x2_08,
   _maxpool2x2_16,
   _maxpool2x2_24,
};

static void _maxpool2x2_nn(
         int                   _nn,
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_nn >= 128)
   {
      int nb = (_nn >> 7);
      for (int i = 0; i < nb; i++)
      {
         (maxpool2x2_fntab_32[1])(_Lx, _Ln, xloc, yloc);
         xloc += 64;
         yloc += 64;

         (maxpool2x2_fntab_32[1])(_Lx, _Ln, xloc, yloc);
         xloc += 64;
         yloc += 64;
      }
      _nn -= (nb << 7);
   }

   if (_nn >= 32)
   {
      int nb = _nn >> 5;
      (maxpool2x2_fntab_32[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 5);
      yloc += (nb << 5);
      _nn  -= (nb << 5);
   }

   if (_nn >= 8)
   {
      int nb = _nn >> 3;
      (maxpool2x2_fntab_08[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 3);
      yloc += (nb << 3);
      _nn  -= (nb << 3);
   }

   {
      cnn_type_t wloc[8];
      for (int p = 0; p < _nn; p++){wloc[p] = xloc[p];};
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Lx +       p]) ? (wloc[p]) : (xloc[_Lx +       p]);};
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln +       p]) ? (wloc[p]) : (xloc[_Ln +       p]);};
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Lx + p]) ? (wloc[p]) : (xloc[_Ln + _Lx + p]);};
      for (int p = 0; p < _nn; p++){yloc[p] = wloc[p];};
   }
}

/////////////////////////////////////////////////////////////////////////////////////
#define maxpool2x1_08(_Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = xloc[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln + p]) ? (wloc[p]) : (xloc[_Ln + p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = wloc[p];};  \
   xloc += CNN_BCHSIZ; \
   yloc += CNN_BCHSIZ; \
}

#define maxpool2x1_16(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool2x1_08(_Lx, _Ln, xloc, yloc) \
   maxpool2x1_08(_Lx, _Ln, xloc, yloc) \
}

#define maxpool2x1_32(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool2x1_16(_Lx, _Ln, xloc, yloc) \
   maxpool2x1_16(_Lx, _Ln, xloc, yloc) \
}

#define maxpool2x1_64(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool2x1_32(_Lx, _Ln, xloc, yloc) \
   maxpool2x1_32(_Lx, _Ln, xloc, yloc) \
}

#define maxpool2x1_128(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool2x1_64(_Lx, _Ln, xloc, yloc) \
   maxpool2x1_64(_Lx, _Ln, xloc, yloc) \
}

void _maxpool2x1_00(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
}

void _maxpool2x1_96(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x1_64(_Lx, _Ln, xloc, yloc);
   maxpool2x1_32(_Lx, _Ln, xloc, yloc);   
}

void _maxpool2x1_64(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x1_64(_Lx, _Ln, xloc, yloc);
}

void _maxpool2x1_32(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x1_32(_Lx, _Ln, xloc, yloc);
}

void _maxpool2x1_24(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x1_16(_Lx, _Ln, xloc, yloc);
   maxpool2x1_08(_Lx, _Ln, xloc, yloc);   
}

void _maxpool2x1_16(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x1_16(_Lx, _Ln, xloc, yloc);
}

void _maxpool2x1_08(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x1_08(_Lx, _Ln, xloc, yloc);
}

typedef void (*maxpool2x1_func_t)(int, int, const cnn_type_t *, cnn_type_t *);
static maxpool2x1_func_t maxpool2x1_fntab_32[] = {
   _maxpool2x1_00,
   _maxpool2x1_32,
   _maxpool2x1_64,
   _maxpool2x1_96,
};

static maxpool2x1_func_t maxpool2x1_fntab_08[] = {
   _maxpool2x1_00,
   _maxpool2x1_08,
   _maxpool2x1_16,
   _maxpool2x1_24,
};

static void _maxpool2x1_nn(
         int                   _nn,
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_nn >= 128)
   {
      int nb = (_nn >> 7);
      for (int i = 0; i < nb; i++)
      {
         (maxpool2x1_fntab_32[1])(_Lx, _Ln, xloc, yloc);
         xloc += 64;
         yloc += 64;

         (maxpool2x1_fntab_32[1])(_Lx, _Ln, xloc, yloc);
         xloc += 64;
         yloc += 64;
      }
      _nn -= (nb << 7);
   }

   if (_nn >= 32)
   {
      int nb = _nn >> 5;
      (maxpool2x1_fntab_32[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 5);
      yloc += (nb << 5);
      _nn  -= (nb << 5);
   }

   if (_nn >= 8)
   {
      int nb = _nn >> 3;
      (maxpool2x1_fntab_08[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 3);
      yloc += (nb << 3);
      _nn  -= (nb << 3);
   }

   {
      cnn_type_t wloc[8];
      for (int p = 0; p < _nn; p++){wloc[p] = xloc[p];};
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln + p]) ? (wloc[p]) : (xloc[_Ln + p]);};
      for (int p = 0; p < _nn; p++){yloc[p] = wloc[p];};
   }
}

/////////////////////////////////////////////////////////////////////////////////////
#define maxpool1x2_08(_Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = xloc[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Lx + p]) ? (wloc[p]) : (xloc[_Lx + p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = wloc[p];};  \
   xloc += CNN_BCHSIZ; \
   yloc += CNN_BCHSIZ; \
}

#define maxpool1x2_16(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool1x2_08(_Lx, _Ln, xloc, yloc) \
   maxpool1x2_08(_Lx, _Ln, xloc, yloc) \
}

#define maxpool1x2_32(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool1x2_16(_Lx, _Ln, xloc, yloc) \
   maxpool1x2_16(_Lx, _Ln, xloc, yloc) \
}

#define maxpool1x2_64(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool1x2_32(_Lx, _Ln, xloc, yloc) \
   maxpool1x2_32(_Lx, _Ln, xloc, yloc) \
}

#define maxpool1x2_128(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool1x2_64(_Lx, _Ln, xloc, yloc) \
   maxpool1x2_64(_Lx, _Ln, xloc, yloc) \
}

void _maxpool1x2_00(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
}

void _maxpool1x2_96(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x2_64(_Lx, _Ln, xloc, yloc);
   maxpool1x2_32(_Lx, _Ln, xloc, yloc);   
}

void _maxpool1x2_64(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x2_64(_Lx, _Ln, xloc, yloc);
}

void _maxpool1x2_32(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x2_32(_Lx, _Ln, xloc, yloc);
}

void _maxpool1x2_24(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x2_16(_Lx, _Ln, xloc, yloc);
   maxpool1x2_08(_Lx, _Ln, xloc, yloc);   
}

void _maxpool1x2_16(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x2_16(_Lx, _Ln, xloc, yloc);
}

void _maxpool1x2_08(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x2_08(_Lx, _Ln, xloc, yloc);
}

typedef void (*maxpool1x2_func_t)(int, int, const cnn_type_t *, cnn_type_t *);
static maxpool1x2_func_t maxpool1x2_fntab_32[] = {
   _maxpool1x2_00,
   _maxpool1x2_32,
   _maxpool1x2_64,
   _maxpool1x2_96,
};

static maxpool1x2_func_t maxpool1x2_fntab_08[] = {
   _maxpool1x2_00,
   _maxpool1x2_08,
   _maxpool1x2_16,
   _maxpool1x2_24,
};

static void _maxpool1x2_nn(
         int                   _nn,
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_nn >= 128)
   {
      int nb = (_nn >> 7);
      for (int i = 0; i < nb; i++)
      {
         (maxpool1x2_fntab_32[1])(_Lx, _Ln, xloc, yloc);
         xloc += 64;
         yloc += 64;

         (maxpool1x2_fntab_32[1])(_Lx, _Ln, xloc, yloc);
         xloc += 64;
         yloc += 64;
      }
      _nn -= (nb << 7);
   }

   if (_nn >= 32)
   {
      int nb = _nn >> 5;
      (maxpool1x2_fntab_32[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 5);
      yloc += (nb << 5);
      _nn  -= (nb << 5);
   }

   if (_nn >= 8)
   {
      int nb = _nn >> 3;
      (maxpool1x2_fntab_08[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 3);
      yloc += (nb << 3);
      _nn  -= (nb << 3);
   }

   {
      cnn_type_t wloc[8];
      for (int p = 0; p < _nn; p++){wloc[p] = xloc[p];};
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Lx + p]) ? (wloc[p]) : (xloc[_Lx + p]);};
      for (int p = 0; p < _nn; p++){yloc[p] = wloc[p];};
   }
}

static void _maxpool1x1_nn(
         int                   _nn,
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   for (int p = 0; p < _nn; p++){yloc[p] = xloc[p];};
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* C should be even number !!!                                                                            */
////////////////////////////////////////////////////////////////////////////////////////////////////////////
void maxpool2x2(
         int  M,
         int  N,
         int  C,
   const cnn_type_t * restrict _px,
         cnn_type_t * restrict _py
)
{
   cnn_type_t * px = __builtin_assume_aligned(_px, 16);
   cnn_type_t * py = __builtin_assume_aligned(_py, 16);

   int _Lx = C;
   int _Ln = C * N;

   cnn_type_t * xloc = px;
   cnn_type_t * yloc = py;
   for (int m = 0; m < M; m+=2)
   {
      for (int n = 0; n < N; n+=2)
      {
         xloc = px + ( ((m  ) * (N  ) * C) + (n  ) * C);
         yloc = py + ( ((m/2) * (N/2) * C) + (n/2) * C);

         _maxpool2x2_nn(C, _Lx, _Ln, xloc, yloc);
      }
   }
}

/////////////////////////////////////////////////////////////////////////////////////
#define maxpool3x3_08(_Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = xloc[p];};                                                              \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Lx +       p]) ? (wloc[p]) : (xloc[_Lx +       p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Lx + _Lx + p]) ? (wloc[p]) : (xloc[_Lx + _Lx + p]);};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln +             p]) ? (wloc[p]) : (xloc[_Ln +             p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Lx +       p]) ? (wloc[p]) : (xloc[_Ln + _Lx +       p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Lx + _Lx + p]) ? (wloc[p]) : (xloc[_Ln + _Lx + _Lx + p]);};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Ln +             p]) ? (wloc[p]) : (xloc[_Ln + _Ln +             p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Ln + _Lx +       p]) ? (wloc[p]) : (xloc[_Ln + _Ln + _Lx +       p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Ln + _Lx + _Lx + p]) ? (wloc[p]) : (xloc[_Ln + _Ln + _Lx + _Lx + p]);};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = wloc[p];};  \
   xloc += CNN_BCHSIZ; \
   yloc += CNN_BCHSIZ; \
}

#define maxpool3x3_16(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool3x3_08(_Lx, _Ln, xloc, yloc) \
   maxpool3x3_08(_Lx, _Ln, xloc, yloc) \
}

#define maxpool3x3_32(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool3x3_16(_Lx, _Ln, xloc, yloc) \
   maxpool3x3_16(_Lx, _Ln, xloc, yloc) \
}

#define maxpool3x3_64(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool3x3_32(_Lx, _Ln, xloc, yloc) \
   maxpool3x3_32(_Lx, _Ln, xloc, yloc) \
}

#define maxpool3x3_128(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool3x3_64(_Lx, _Ln, xloc, yloc) \
   maxpool3x3_64(_Lx, _Ln, xloc, yloc) \
}

void _maxpool3x3_00(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
}

void _maxpool3x3_96(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x3_64(_Lx, _Ln, xloc, yloc);
   maxpool3x3_32(_Lx, _Ln, xloc, yloc);   
}

void _maxpool3x3_64(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x3_64(_Lx, _Ln, xloc, yloc);
}

void _maxpool3x3_32(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x3_32(_Lx, _Ln, xloc, yloc);
}

void _maxpool3x3_24(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x3_16(_Lx, _Ln, xloc, yloc);
   maxpool3x3_08(_Lx, _Ln, xloc, yloc);   
}

void _maxpool3x3_16(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x3_16(_Lx, _Ln, xloc, yloc);
}

void _maxpool3x3_08(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x3_08(_Lx, _Ln, xloc, yloc);
}

typedef void (*maxpool3x3_func_t)(int, int, const cnn_type_t *, cnn_type_t *);
static maxpool3x3_func_t maxpool3x3_fntab_32[] = {
   _maxpool3x3_00,
   _maxpool3x3_32,
   _maxpool3x3_64,
   _maxpool3x3_96,
};

static maxpool3x3_func_t maxpool3x3_fntab_08[] = {
   _maxpool3x3_00,
   _maxpool3x3_08,
   _maxpool3x3_16,
   _maxpool3x3_24,
};

static void _maxpool3x3_nn(
         int                   _nn,
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_nn >= 128)
   {
      int nb = (_nn >> 7);
      for (int i = 0; i < nb; i++)
      {
         (maxpool3x3_fntab_32[1])(_Lx, _Ln, xloc, yloc);
         xloc += 64;
         yloc += 64;

         (maxpool3x3_fntab_32[1])(_Lx, _Ln, xloc, yloc);
         xloc += 64;
         yloc += 64;
      }
      _nn -= (nb << 7);
   }

   if (_nn >= 32)
   {
      int nb = _nn >> 5;
      (maxpool3x3_fntab_32[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 5);
      yloc += (nb << 5);
      _nn  -= (nb << 5);
   }

   if (_nn >= 8)
   {
      int nb = _nn >> 3;
      (maxpool3x3_fntab_08[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 3);
      yloc += (nb << 3);
      _nn  -= (nb << 3);
   }

   {  \
      cnn_type_t wloc[8];             \
      \
      for (int p = 0; p < _nn; p++){wloc[p] = xloc[p];};                                                              \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Lx +       p]) ? (wloc[p]) : (xloc[_Lx +       p]);};  \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Lx + _Lx + p]) ? (wloc[p]) : (xloc[_Lx + _Lx + p]);};  \
      \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln +             p]) ? (wloc[p]) : (xloc[_Ln +             p]);};  \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Lx +       p]) ? (wloc[p]) : (xloc[_Ln + _Lx +       p]);};  \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Lx + _Lx + p]) ? (wloc[p]) : (xloc[_Ln + _Lx + _Lx + p]);};  \
      \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Ln +             p]) ? (wloc[p]) : (xloc[_Ln + _Ln +             p]);};  \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Ln + _Lx +       p]) ? (wloc[p]) : (xloc[_Ln + _Ln + _Lx +       p]);};  \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Ln + _Lx + _Lx + p]) ? (wloc[p]) : (xloc[_Ln + _Ln + _Lx + _Lx + p]);};  \
      \
      for (int p = 0; p < _nn; p++){yloc[p] = wloc[p];};  \
   }
}

/////////////////////////////////////////////////////////////////////////////////////
#define maxpool3x2_08(_Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = xloc[p];};                                                  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Lx + p]) ? (wloc[p]) : (xloc[_Lx + p]);};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln +       p]) ? (wloc[p]) : (xloc[_Ln +       p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Lx + p]) ? (wloc[p]) : (xloc[_Ln + _Lx + p]);};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Ln +       p]) ? (wloc[p]) : (xloc[_Ln + _Ln +       p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Ln + _Lx + p]) ? (wloc[p]) : (xloc[_Ln + _Ln + _Lx + p]);};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = wloc[p];};  \
   xloc += CNN_BCHSIZ; \
   yloc += CNN_BCHSIZ; \
}

#define maxpool3x2_16(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool3x2_08(_Lx, _Ln, xloc, yloc) \
   maxpool3x2_08(_Lx, _Ln, xloc, yloc) \
}

#define maxpool3x2_32(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool3x2_16(_Lx, _Ln, xloc, yloc) \
   maxpool3x2_16(_Lx, _Ln, xloc, yloc) \
}

#define maxpool3x2_64(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool3x2_32(_Lx, _Ln, xloc, yloc) \
   maxpool3x2_32(_Lx, _Ln, xloc, yloc) \
}

#define maxpool3x2_128(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool3x2_64(_Lx, _Ln, xloc, yloc) \
   maxpool3x2_64(_Lx, _Ln, xloc, yloc) \
}

void _maxpool3x2_00(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
}

void _maxpool3x2_96(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x2_64(_Lx, _Ln, xloc, yloc);
   maxpool3x2_32(_Lx, _Ln, xloc, yloc);   
}

void _maxpool3x2_64(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x2_64(_Lx, _Ln, xloc, yloc);
}

void _maxpool3x2_32(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x2_32(_Lx, _Ln, xloc, yloc);
}

void _maxpool3x2_24(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x2_16(_Lx, _Ln, xloc, yloc);
   maxpool3x2_08(_Lx, _Ln, xloc, yloc);   
}

void _maxpool3x2_16(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x2_16(_Lx, _Ln, xloc, yloc);
}

void _maxpool3x2_08(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x2_08(_Lx, _Ln, xloc, yloc);
}

typedef void (*maxpool3x2_func_t)(int, int, const cnn_type_t *, cnn_type_t *);
static maxpool3x2_func_t maxpool3x2_fntab_32[] = {
   _maxpool3x2_00,
   _maxpool3x2_32,
   _maxpool3x2_64,
   _maxpool3x2_96,
};

static maxpool3x2_func_t maxpool3x2_fntab_08[] = {
   _maxpool3x2_00,
   _maxpool3x2_08,
   _maxpool3x2_16,
   _maxpool3x2_24,
};

static void _maxpool3x2_nn(
         int                   _nn,
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_nn >= 128)
   {
      int nb = (_nn >> 7);
      for (int i = 0; i < nb; i++)
      {
         (maxpool3x2_fntab_32[1])(_Lx, _Ln, xloc, yloc);
         xloc += 64;
         yloc += 64;

         (maxpool3x2_fntab_32[1])(_Lx, _Ln, xloc, yloc);
         xloc += 64;
         yloc += 64;
      }
      _nn -= (nb << 7);
   }

   if (_nn >= 32)
   {
      int nb = _nn >> 5;
      (maxpool3x2_fntab_32[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 5);
      yloc += (nb << 5);
      _nn  -= (nb << 5);
   }

   if (_nn >= 8)
   {
      int nb = _nn >> 3;
      (maxpool3x2_fntab_08[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 3);
      yloc += (nb << 3);
      _nn  -= (nb << 3);
   }

   {  \
      cnn_type_t wloc[8];             \
      \
      for (int p = 0; p < _nn; p++){wloc[p] = xloc[p];};                                                  \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Lx + p]) ? (wloc[p]) : (xloc[_Lx + p]);};  \
      \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln +       p]) ? (wloc[p]) : (xloc[_Ln +       p]);};  \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Lx + p]) ? (wloc[p]) : (xloc[_Ln + _Lx + p]);};  \
      \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Ln +       p]) ? (wloc[p]) : (xloc[_Ln + _Ln +       p]);};  \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Ln + _Lx + p]) ? (wloc[p]) : (xloc[_Ln + _Ln + _Lx + p]);};  \
      \
      for (int p = 0; p < _nn; p++){yloc[p] = wloc[p];};  \
   }
}

/////////////////////////////////////////////////////////////////////////////////////
#define maxpool3x1_08(_Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = xloc[p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln +       p]) ? (wloc[p]) : (xloc[_Ln +       p]);};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Ln + p]) ? (wloc[p]) : (xloc[_Ln + _Ln + p]);};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = wloc[p];};  \
   xloc += CNN_BCHSIZ; \
   yloc += CNN_BCHSIZ; \
}

#define maxpool3x1_16(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool3x1_08(_Lx, _Ln, xloc, yloc) \
   maxpool3x1_08(_Lx, _Ln, xloc, yloc) \
}

#define maxpool3x1_32(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool3x1_16(_Lx, _Ln, xloc, yloc) \
   maxpool3x1_16(_Lx, _Ln, xloc, yloc) \
}

#define maxpool3x1_64(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool3x1_32(_Lx, _Ln, xloc, yloc) \
   maxpool3x1_32(_Lx, _Ln, xloc, yloc) \
}

#define maxpool3x1_128(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool3x1_64(_Lx, _Ln, xloc, yloc) \
   maxpool3x1_64(_Lx, _Ln, xloc, yloc) \
}

void _maxpool3x1_00(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
}

void _maxpool3x1_96(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x1_64(_Lx, _Ln, xloc, yloc);
   maxpool3x1_32(_Lx, _Ln, xloc, yloc);   
}

void _maxpool3x1_64(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x1_64(_Lx, _Ln, xloc, yloc);
}

void _maxpool3x1_32(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x1_32(_Lx, _Ln, xloc, yloc);
}

void _maxpool3x1_24(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x1_16(_Lx, _Ln, xloc, yloc);
   maxpool3x1_08(_Lx, _Ln, xloc, yloc);   
}

void _maxpool3x1_16(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x1_16(_Lx, _Ln, xloc, yloc);
}

void _maxpool3x1_08(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x1_08(_Lx, _Ln, xloc, yloc);
}

typedef void (*maxpool3x1_func_t)(int, int, const cnn_type_t *, cnn_type_t *);
static maxpool3x1_func_t maxpool3x1_fntab_32[] = {
   _maxpool3x1_00,
   _maxpool3x1_32,
   _maxpool3x1_64,
   _maxpool3x1_96,
};

static maxpool3x1_func_t maxpool3x1_fntab_08[] = {
   _maxpool3x1_00,
   _maxpool3x1_08,
   _maxpool3x1_16,
   _maxpool3x1_24,
};

static void _maxpool3x1_nn(
         int                   _nn,
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_nn >= 128)
   {
      int nb = (_nn >> 7);
      for (int i = 0; i < nb; i++)
      {
         (maxpool3x1_fntab_32[1])(_Lx, _Ln, xloc, yloc);
         xloc += 64;
         yloc += 64;

         (maxpool3x1_fntab_32[1])(_Lx, _Ln, xloc, yloc);
         xloc += 64;
         yloc += 64;
      }
      _nn -= (nb << 7);
   }

   if (_nn >= 32)
   {
      int nb = _nn >> 5;
      (maxpool3x1_fntab_32[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 5);
      yloc += (nb << 5);
      _nn  -= (nb << 5);
   }

   if (_nn >= 8)
   {
      int nb = _nn >> 3;
      (maxpool3x1_fntab_08[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 3);
      yloc += (nb << 3);
      _nn  -= (nb << 3);
   }

   {  \
      cnn_type_t wloc[8];             \
      \
      for (int p = 0; p < _nn; p++){wloc[p] = xloc[p];}; \
      \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln +       p]) ? (wloc[p]) : (xloc[_Ln +       p]);};  \
      \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Ln + p]) ? (wloc[p]) : (xloc[_Ln + _Ln + p]);};  \
      \
      for (int p = 0; p < _nn; p++){yloc[p] = wloc[p];};  \
   }
}

/////////////////////////////////////////////////////////////////////////////////////
#define maxpool2x3_08(_Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = xloc[p];};                                                              \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Lx +       p]) ? (wloc[p]) : (xloc[_Lx +       p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Lx + _Lx + p]) ? (wloc[p]) : (xloc[_Lx + _Lx + p]);};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln +             p]) ? (wloc[p]) : (xloc[_Ln +             p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Lx +       p]) ? (wloc[p]) : (xloc[_Ln + _Lx +       p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Lx + _Lx + p]) ? (wloc[p]) : (xloc[_Ln + _Lx + _Lx + p]);};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = wloc[p];};  \
   xloc += CNN_BCHSIZ; \
   yloc += CNN_BCHSIZ; \
}

#define maxpool2x3_16(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool2x3_08(_Lx, _Ln, xloc, yloc) \
   maxpool2x3_08(_Lx, _Ln, xloc, yloc) \
}

#define maxpool2x3_32(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool2x3_16(_Lx, _Ln, xloc, yloc) \
   maxpool2x3_16(_Lx, _Ln, xloc, yloc) \
}

#define maxpool2x3_64(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool2x3_32(_Lx, _Ln, xloc, yloc) \
   maxpool2x3_32(_Lx, _Ln, xloc, yloc) \
}

#define maxpool2x3_128(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool2x3_64(_Lx, _Ln, xloc, yloc) \
   maxpool2x3_64(_Lx, _Ln, xloc, yloc) \
}

void _maxpool2x3_00(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
}

void _maxpool2x3_96(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x3_64(_Lx, _Ln, xloc, yloc);
   maxpool2x3_32(_Lx, _Ln, xloc, yloc);   
}

void _maxpool2x3_64(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x3_64(_Lx, _Ln, xloc, yloc);
}

void _maxpool2x3_32(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x3_32(_Lx, _Ln, xloc, yloc);
}

void _maxpool2x3_24(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x3_16(_Lx, _Ln, xloc, yloc);
   maxpool2x3_08(_Lx, _Ln, xloc, yloc);   
}

void _maxpool2x3_16(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x3_16(_Lx, _Ln, xloc, yloc);
}

void _maxpool2x3_08(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x3_08(_Lx, _Ln, xloc, yloc);
}

typedef void (*maxpool2x3_func_t)(int, int, const cnn_type_t *, cnn_type_t *);
static maxpool2x3_func_t maxpool2x3_fntab_32[] = {
   _maxpool2x3_00,
   _maxpool2x3_32,
   _maxpool2x3_64,
   _maxpool2x3_96,
};

static maxpool2x3_func_t maxpool2x3_fntab_08[] = {
   _maxpool2x3_00,
   _maxpool2x3_08,
   _maxpool2x3_16,
   _maxpool2x3_24,
};

static void _maxpool2x3_nn(
         int                   _nn,
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_nn >= 128)
   {
      int nb = (_nn >> 7);
      for (int i = 0; i < nb; i++)
      {
         (maxpool2x3_fntab_32[1])(_Lx, _Ln, xloc, yloc);
         xloc += 64;
         yloc += 64;

         (maxpool2x3_fntab_32[1])(_Lx, _Ln, xloc, yloc);
         xloc += 64;
         yloc += 64;
      }
      _nn -= (nb << 7);
   }

   if (_nn >= 32)
   {
      int nb = _nn >> 5;
      (maxpool2x3_fntab_32[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 5);
      yloc += (nb << 5);
      _nn  -= (nb << 5);
   }

   if (_nn >= 8)
   {
      int nb = _nn >> 3;
      (maxpool2x3_fntab_08[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 3);
      yloc += (nb << 3);
      _nn  -= (nb << 3);
   }

   {  \
      cnn_type_t wloc[8]; \
      \
      for (int p = 0; p < _nn; p++){wloc[p] = xloc[p];};                                                              \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Lx +       p]) ? (wloc[p]) : (xloc[_Lx +       p]);};  \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Lx + _Lx + p]) ? (wloc[p]) : (xloc[_Lx + _Lx + p]);};  \
      \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln +             p]) ? (wloc[p]) : (xloc[_Ln +             p]);};  \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Lx +       p]) ? (wloc[p]) : (xloc[_Ln + _Lx +       p]);};  \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Lx + _Lx + p]) ? (wloc[p]) : (xloc[_Ln + _Lx + _Lx + p]);};  \
      \
      for (int p = 0; p < _nn; p++){yloc[p] = wloc[p];};  \
   }
}

/////////////////////////////////////////////////////////////////////////////////////
#define maxpool1x3_08(_Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = xloc[p];};                                                              \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Lx +       p]) ? (wloc[p]) : (xloc[_Lx +       p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Lx + _Lx + p]) ? (wloc[p]) : (xloc[_Lx + _Lx + p]);};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = wloc[p];};  \
   xloc += CNN_BCHSIZ; \
   yloc += CNN_BCHSIZ; \
}

#define maxpool1x3_16(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool1x3_08(_Lx, _Ln, xloc, yloc) \
   maxpool1x3_08(_Lx, _Ln, xloc, yloc) \
}

#define maxpool1x3_32(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool1x3_16(_Lx, _Ln, xloc, yloc) \
   maxpool1x3_16(_Lx, _Ln, xloc, yloc) \
}

#define maxpool1x3_64(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool1x3_32(_Lx, _Ln, xloc, yloc) \
   maxpool1x3_32(_Lx, _Ln, xloc, yloc) \
}

#define maxpool1x3_128(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool1x3_64(_Lx, _Ln, xloc, yloc) \
   maxpool1x3_64(_Lx, _Ln, xloc, yloc) \
}

void _maxpool1x3_00(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
}

void _maxpool1x3_96(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x3_64(_Lx, _Ln, xloc, yloc);
   maxpool1x3_32(_Lx, _Ln, xloc, yloc);   
}

void _maxpool1x3_64(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x3_64(_Lx, _Ln, xloc, yloc);
}

void _maxpool1x3_32(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x3_32(_Lx, _Ln, xloc, yloc);
}

void _maxpool1x3_24(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x3_16(_Lx, _Ln, xloc, yloc);
   maxpool1x3_08(_Lx, _Ln, xloc, yloc);   
}

void _maxpool1x3_16(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x3_16(_Lx, _Ln, xloc, yloc);
}

void _maxpool1x3_08(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x3_08(_Lx, _Ln, xloc, yloc);
}

typedef void (*maxpool1x3_func_t)(int, int, const cnn_type_t *, cnn_type_t *);
static maxpool1x3_func_t maxpool1x3_fntab_32[] = {
   _maxpool1x3_00,
   _maxpool1x3_32,
   _maxpool1x3_64,
   _maxpool1x3_96,
};

static maxpool1x3_func_t maxpool1x3_fntab_08[] = {
   _maxpool1x3_00,
   _maxpool1x3_08,
   _maxpool1x3_16,
   _maxpool1x3_24,
};

static void _maxpool1x3_nn(
         int                   _nn,
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_nn >= 128)
   {
      int nb = (_nn >> 7);
      for (int i = 0; i < nb; i++)
      {
         (maxpool1x3_fntab_32[1])(_Lx, _Ln, xloc, yloc);
         xloc += 64;
         yloc += 64;

         (maxpool1x3_fntab_32[1])(_Lx, _Ln, xloc, yloc);
         xloc += 64;
         yloc += 64;
      }
      _nn -= (nb << 7);
   }

   if (_nn >= 32)
   {
      int nb = _nn >> 5;
      (maxpool1x3_fntab_32[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 5);
      yloc += (nb << 5);
      _nn  -= (nb << 5);
   }

   if (_nn >= 8)
   {
      int nb = _nn >> 3;
      (maxpool1x3_fntab_08[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 3);
      yloc += (nb << 3);
      _nn  -= (nb << 3);
   }

   {  \
      cnn_type_t wloc[8]; \
      \
      for (int p = 0; p < _nn; p++){wloc[p] = xloc[p];};                                                              \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Lx +       p]) ? (wloc[p]) : (xloc[_Lx +       p]);};  \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Lx + _Lx + p]) ? (wloc[p]) : (xloc[_Lx + _Lx + p]);};  \
      \
      for (int p = 0; p < _nn; p++){yloc[p] = wloc[p];};  \
   }
}


/* (S*C) % 4 = 0 !! */
void maxpool3x3(
         int  M,
         int  N,
         int  C,
         int  S,
   const cnn_type_t * restrict _px,
         cnn_type_t * restrict _py
)
{
   cnn_type_t * px = __builtin_assume_aligned(_px, 16);
   cnn_type_t * py = __builtin_assume_aligned(_py, 16);

   int _Lx = C;
   int _Ln = C * N;

   cnn_type_t * xloc = px;
   cnn_type_t * yloc = py;
   
   assert( ((S*C)&3) == 0 );
   
   int m;
   for (m = 0; m < (M-3+1); m+=S)
   {
      int n;

      xloc = px + m * N * C;

      for (n = 0; n < (N-3+1); n+=S)
      {
         _maxpool3x3_nn(C, _Lx, _Ln, xloc, yloc);
         yloc = yloc + C;
         xloc = xloc + S * C;
      }

      /* skip the reset part if ((N-3)%S == 0) */
      if ((n-S) == (N-3)){continue;};

      {
         switch (N-n)
         {
            case 2:
            _maxpool3x2_nn(C, _Lx, _Ln, xloc, yloc);
            break;
			
            case 1:
            _maxpool3x1_nn(C, _Lx, _Ln, xloc, yloc);
            break;
         }
         yloc = yloc + C;
      }
   }

   /* skip the reset part if ((N-3)%S == 0) */
   if ((m-S) == (M-3)){return;};

   if ((M-m) >= 3) printf("m = %d, M = %d\n", m, M);

   switch (M-m)
   {
      case 2:
      {
         int n;

         xloc = px + m * N * C;

         for (n = 0; n < (N-3+1); n+=S)
         {
            _maxpool2x3_nn(C, _Lx, _Ln, xloc, yloc);
            yloc = yloc + C;
            xloc = xloc + S * C;
         }

         /* skip the reset part if ((N-3)%S == 0) */
         if ((N-n) == 3) printf("n = %d, N = %d\n", n, N);

         if ((n-S) != (N-3))
         {
            switch (N-n)
            {
               case 2:
               _maxpool2x2_nn(C, _Lx, _Ln, xloc, yloc);
               break;

               case 1:
               _maxpool2x1_nn(C, _Lx, _Ln, xloc, yloc);
               break;
            }
            yloc = yloc + C;
         }
      }
      break;

      case 1:
      {
         int n;

         xloc = px + m * N * C;

         for (n = 0; n < (N-3+1); n+=S)
         {
            _maxpool1x3_nn(C, _Lx, _Ln, xloc, yloc);
            yloc = yloc + C;
            xloc = xloc + S * C;
         }

         /* skip the reset part if ((N-3)%S == 0) */
         

         if ((N-n) == 3) printf("n = %d, N = %d\n", n, N);

         if ((n-S) != (N-3))
         {
            switch (N-n)
            {
               case 2:
               _maxpool1x2_nn(C, _Lx, _Ln, xloc, yloc);
               break;

               case 1:
               _maxpool1x1_nn(C, _Lx, _Ln, xloc, yloc);
               break;
            }
            yloc = yloc + C;
         }
      }
      break;
   }
}

//////////////////////////////////////////////////////////////////////////////////////////////
#define softmaxa_04(xloc, uloc, s)  \
{  \
   for (int p = 0; p < 4; p++){uloc[p] = exp(xloc[p]);};  \
   for (int p = 0; p < 4; p++){s[p]   += uloc[p];     };  \
   \
   xloc += 4;  \
   uloc += 4;  \
}

#define softmaxb_08(yloc, uloc, ss)  \
{  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = uloc[p] / ss;};  \
   \
   yloc += CNN_BCHSIZ;  \
   uloc += CNN_BCHSIZ;  \
}

#define softmaxa_08(xloc, uloc, s)  \
{  \
   softmaxa_04(xloc, uloc, s);      \
   softmaxa_04(xloc, uloc, s);      \
}

#define softmaxa_16(xloc, uloc, s)  \
{  \
   softmaxa_08(xloc, uloc, s);      \
   softmaxa_08(xloc, uloc, s);      \
}

#define softmaxa_32(xloc, uloc, s)  \
{  \
   softmaxa_16(xloc, uloc, s);      \
   softmaxa_16(xloc, uloc, s);      \
}

#define softmaxa_64(xloc, uloc, s)  \
{  \
   softmaxa_32(xloc, uloc, s);      \
   softmaxa_32(xloc, uloc, s);      \
}

#define softmaxb_16(xloc, uloc, s)  \
{  \
   softmaxb_08(xloc, uloc, s);      \
   softmaxb_08(xloc, uloc, s);      \
}

#define softmaxb_32(xloc, uloc, s)  \
{  \
   softmaxb_16(xloc, uloc, s);      \
   softmaxb_16(xloc, uloc, s);      \
}

#define softmaxb_64(xloc, uloc, s)  \
{  \
   softmaxb_32(xloc, uloc, s);      \
   softmaxb_32(xloc, uloc, s);      \
}

void _softmaxa_00(
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _uloc,
         cnn_type_t * restrict _sloc
   )
{
}

void _softmaxa_96(
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _uloc,
         cnn_type_t * restrict _sloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * sloc = __builtin_assume_aligned(_sloc, 16);

   softmaxa_64(xloc, uloc, sloc);
   softmaxa_32(xloc, uloc, sloc);   
}

void _softmaxa_64(
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _uloc,
         cnn_type_t * restrict _sloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * sloc = __builtin_assume_aligned(_sloc, 16);

   softmaxa_64(xloc, uloc, sloc);
}

void _softmaxa_32(
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _uloc,
         cnn_type_t * restrict _sloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * sloc = __builtin_assume_aligned(_sloc, 16);

   softmaxa_32(xloc, uloc, sloc);
}

void _softmaxa_24(
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _uloc,
         cnn_type_t * restrict _sloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * sloc = __builtin_assume_aligned(_sloc, 16);

   softmaxa_16(xloc, uloc, sloc);   
   softmaxa_08(xloc, uloc, sloc);   
}

void _softmaxa_16(
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _uloc,
         cnn_type_t * restrict _sloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * sloc = __builtin_assume_aligned(_sloc, 16);

   softmaxa_16(xloc, uloc, sloc);   
}

void _softmaxa_08(
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _uloc,
         cnn_type_t * restrict _sloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * sloc = __builtin_assume_aligned(_sloc, 16);

   softmaxa_08(xloc, uloc, sloc);
}

void _softmaxb_00(
         cnn_type_t * restrict _yloc,
   const cnn_type_t * restrict _uloc,
         cnn_type_t            _ss
   )
{
}

void _softmaxb_96(
         cnn_type_t * restrict _yloc,
   const cnn_type_t * restrict _uloc,
         cnn_type_t            _ss
   )
{
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);

   softmaxb_64(yloc, uloc, _ss);
   softmaxb_32(yloc, uloc, _ss);
}

void _softmaxb_64(
         cnn_type_t * restrict _yloc,
   const cnn_type_t * restrict _uloc,
         cnn_type_t            _ss
   )
{
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);

   softmaxb_64(yloc, uloc, _ss);
}

void _softmaxb_32(
         cnn_type_t * restrict _yloc,
   const cnn_type_t * restrict _uloc,
         cnn_type_t            _ss
   )
{
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);

   softmaxb_32(yloc, uloc, _ss);
}

void _softmaxb_24(
         cnn_type_t * restrict _yloc,
   const cnn_type_t * restrict _uloc,
         cnn_type_t            _ss
   )
{
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);

   softmaxb_16(yloc, uloc, _ss);
   softmaxb_08(yloc, uloc, _ss);
}

void _softmaxb_16(
         cnn_type_t * restrict _yloc,
   const cnn_type_t * restrict _uloc,
         cnn_type_t            _ss
   )
{
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);

   softmaxb_16(yloc, uloc, _ss);
}

void _softmaxb_08(
         cnn_type_t * restrict _yloc,
   const cnn_type_t * restrict _uloc,
         cnn_type_t            _ss
   )
{
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);

   softmaxb_08(yloc, uloc, _ss);
}

typedef void (*softmaxa_func_t)(const cnn_type_t *, cnn_type_t *, cnn_type_t *);
static softmaxa_func_t softmaxa_fntab_32[] = {
   _softmaxa_00,
   _softmaxa_32,
   _softmaxa_64,
   _softmaxa_96,
};

static softmaxa_func_t softmaxa_fntab_08[] = {
   _softmaxa_00,
   _softmaxa_08,
   _softmaxa_16,
   _softmaxa_24,
};

typedef void (*softmaxb_func_t)(cnn_type_t *, const cnn_type_t *, cnn_type_t);
static softmaxb_func_t softmaxb_fntab_32[] = {
   _softmaxb_00,
   _softmaxb_32,
   _softmaxb_64,
   _softmaxb_96,
};

static softmaxb_func_t softmaxb_fntab_08[] = {
   _softmaxb_00,
   _softmaxb_08,
   _softmaxb_16,
   _softmaxb_24,
};

static void _softmaxa_nn(
         int                   _nn,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _uloc,
         cnn_type_t * restrict _sloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * sloc = __builtin_assume_aligned(_sloc, 16);

   if (_nn >= 128)
   {
      int nb = (_nn >> 7);
      for (int i = 0; i < nb; i++)
      {
         (softmaxa_fntab_32[1])(xloc, uloc, sloc);
         xloc += 64;
         uloc += 64;

         (softmaxa_fntab_32[1])(xloc, uloc, sloc);
         xloc += 64;
         uloc += 64;
      }
      _nn -= (nb << 7);
   }

   if (_nn >= 32)
   {
      int nb = _nn >> 5;
      (softmaxa_fntab_32[nb])(xloc, uloc, sloc);
      xloc += (nb << 5);
      uloc += (nb << 5);
      _nn  -= (nb << 5);
   }

   if (_nn >= 8)
   {
      int nb = _nn >> 3;
      (softmaxa_fntab_08[nb])(xloc, uloc, sloc);
      xloc += (nb << 3);
      uloc += (nb << 3);
      _nn  -= (nb << 3);
   }

   for (int n = 0; n < _nn; n++)
   {
      uloc[n    ]  = exp(xloc[n]);
      sloc[n & 3] +=    (uloc[n]);
   }
}

static void _softmaxb_nn(
         int                   _nn,         
         cnn_type_t * restrict _yloc,
   const cnn_type_t * restrict _uloc,
         cnn_type_t            _ss
   )
{
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);

   if (_nn >= 128)
   {
      int nb = (_nn >> 7);
      for (int i = 0; i < nb; i++)
      {
         (softmaxb_fntab_32[1])(yloc, uloc, _ss);
         yloc += 64;
         uloc += 64;

         (softmaxb_fntab_32[1])(yloc, uloc, _ss);
         yloc += 64;
         uloc += 64;
      }
      _nn -= (nb << 7);
   }

   if (_nn >= 32)
   {
      int nb = _nn >> 5;
      (softmaxb_fntab_32[nb])(yloc, uloc, _ss);
      yloc += (nb << 5);
      uloc += (nb << 5);
      _nn  -= (nb << 5);
   }

   if (_nn >= 8)
   {
      int nb = _nn >> 3;
      (softmaxb_fntab_08[nb])(yloc, uloc, _ss);
      yloc += (nb << 3);
      uloc += (nb << 3);
      _nn  -= (nb << 3);
   }

   for (int n = 0; n < _nn; n++)
   {
      yloc[n] = uloc[n] / _ss;
   }
}

void softmaxpool(
         int  M,
         int  N,
         int  C,
   const cnn_type_t * restrict _px,
         cnn_type_t * restrict _py
)
{
   cnn_type_t * px = __builtin_assume_aligned(_px, 16);
   cnn_type_t * py = __builtin_assume_aligned(_py, 16);

   cnn_type_t * xloc = px;
   cnn_type_t * yloc = py;

   cnn_type_t   ss;
   cnn_type_t * uloc = _aligned_malloc(1024 * sizeof(cnn_type_t), 16);
   cnn_type_t * sloc = uloc + 768;


   for (int m = 0; m < M; m++)
   {
      for (int n = 0; n < N; n++)
      {
         sloc[0] = sloc[1] = sloc[2] = sloc[3] = 0;

         _softmaxa_nn(C, xloc, uloc, sloc); 

         ss = sloc[0] + sloc[1] + sloc[2] + sloc[3];
         
         _softmaxb_nn(C, yloc, uloc, ss);

         xloc += C;
         yloc += C;       
      }
   }

   _aligned_free(uloc);
}
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Version
//  1.0.1	Claes Rol�n		2014-11-30	First version
//  1.0.2	Claes Rol�n		2015-01-11	Added 'angle','specgram','fd_filter','gplot'
//                                      Changed file structure
//  1.0.3	Claes Rol�n		2015-04-26	Added 'parser' class, 'err_handler','wrn_handler'
//                                      'freqz','phasez'
//  1.0.4	Claes Rol�n		2015-08-01	Added FFTW class, 'unwrap', 'update_coeffs' + commenting for Doxygen
//  1.0.5	Claes Rol�n		2015-10-11	Added plot to file in gplot
//  1.0.6	Claes Rol�n		2015-12-30	Added support for importing/exporting Wisdom in FFTW
//  1.0.7	Claes Rol�n		2016-10-20	Added support for FFTW 2-D and simple image I/O
//  1.0.8   Claes Rol�n     2016-11-15  Added adaptive filters - LMS, N-LMS and RLS. New line plot function of matrix data.
//  1.1.1   Claes Rol�n     2017-01-20  Cleanup, added Kalman and Newton adaptive filters
//  1.1.2   Claes Rol�n     2017-03-07  Added Kalman tracking and control
//  1.2.1   Claes Rol�n     2017-03-13  Updated for Gnuplot 5.0, plot(..) changes to plot_add(..) plus plot_show()
//  1.2.2   Claes Rol�n     2017-07-13  Added EKF and UKF classes, Non class functions set to arma_inline.
//  1.2.3   Claes Rol�n     2017-08-08  Updated FIR design functions, support for highpass, bandpass and bandstop.
//  1.2.4   Claes Rol�n     2018-03-17  Updated resampling class, added goertzel and timevec functions.


#define SP_VERSION_MAJOR 1
#define SP_VERSION_MINOR 2
#define SP_VERSION_PATCH 4


#ifndef ARMA_INCLUDES
#include <armadillo>
#endif

#ifndef SIGPACK_H
#define SIGPACK_H
#include "base/base.h"
#include "window/window.h"
#include "filter/filter.h"
#include "resampling/resampling.h"
#include "spectrum/spectrum.h"
#include "timing/timing.h"
//#include "gplot/gplot.h"
#include "parser/parser.h"
#ifdef HAVE_FFTW
  #include "fftw/fftw.h"
#endif
#include "image/image.h"
#include "kalman/kalman.h"

#endif

/// \mainpage notitle
///
/// \section intro_sec General
/// \tableofcontents
/// SigPack is a C++ signal processing library using the Armadillo library as a base.
/// The API will be familiar for those who has used IT++ and Octave/Matlab. The intention
/// is to keep it small and only implement the fundamental signal processing algorithms.
///
/// \section features_sec Features
/// \li Easy to use, based on Armadillo library
/// \li API similar to Matlab/Octave and IT++
/// \li FIR/IIR filter
/// \li Window functions - Hanning, Hamming, Bartlett, Kaiser ...
/// \li Spectrum and spectrogram
/// \li Timing/Delay
/// \li Gnuplot support
/// \li Up/Downsampling
/// \li Config file parser
/// \li FFTW support for vector and matrix
/// \li Simple image I/O (.pbm,.pgm and .ppm formats)
/// \li Adaptive filters - LMS, N-LMS, RLS, Kalman and Newton
/// \li Extended and Unscented Kalman nonlinear filters
///
/// \section install_sec Installation
/// Download Armadillo and SigPack and install/extract them to your install directory.
/// Armadillo-7.8 version is used in the examples hereafter.
/// For Windows 64bit users: add the \<Armadillo install dir\>\\examples\\lib_win64
/// to your path in your environment variables.


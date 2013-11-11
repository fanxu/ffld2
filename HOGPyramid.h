//--------------------------------------------------------------------------------------------------
// Implementation of the papers "Exact Acceleration of Linear Object Detectors", 12th European
// Conference on Computer Vision, 2012 and "Deformable Part Models with Individual Part Scaling",
// 24th British Machine Vision Conference, 2013.
//
// Copyright (c) 2013 Idiap Research Institute, <http://www.idiap.ch/>
// Written by Charles Dubout <charles.dubout@idiap.ch>
//
// This file is part of FFLDv2 (the Fast Fourier Linear Detector version 2)
//
// FFLDv2 is free software: you can redistribute it and/or modify it under the terms of the GNU
// Affero General Public License version 3 as published by the Free Software Foundation.
//
// FFLDv2 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
// the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero
// General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with FFLDv2. If
// not, see <http://www.gnu.org/licenses/>.
//--------------------------------------------------------------------------------------------------

#ifndef FFLD_HOGPYRAMID_H
#define FFLD_HOGPYRAMID_H

#include "JPEGImage.h"

#include <Eigen/Core>

namespace FFLD
{
/// The HOGPyramid class computes and stores the HOG features extracted from a jpeg image at
/// multiple scales. The scale of the pyramid level of index @c i is given by the following formula:
/// 2^(1 - @c i / @c interval), so that the first scale is at double the resolution of the original
/// image). Each level is padded with zeros horizontally and vertically by a fixed amount. The last
/// feature is special: it takes the value one in the padding and zero otherwise.
/// @note Define the PASCAL_HOGPYRAMID_DOUBLE to use double scalar values instead of float (slower,
/// uses twice the amount of memory, and the increase in precision is not necessarily useful).
/// @note Define the FFLD_HOGPYRAMID_EXTRA_FEATURES to add extra texture (uniform LBP) and color
/// (hue histogram) features in addition to the original HOG features.
class HOGPyramid
{
public:
	/// Number of HOG features (guaranteed to be even). Fixed at compile time for both ease of use
	/// and optimal performance.
#ifndef FFLD_HOGPYRAMID_EXTRA_FEATURES
	static const int NbFeatures = 32;
#else
	static const int NbFeatures = 48;
#endif
	
	/// Type of a scalar value.
#ifndef FFLD_HOGPYRAMID_DOUBLE
	typedef float Scalar;
#else
	typedef double Scalar;
#endif
	
	/// Type of a matrix.
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
	
	/// Type of a pyramid level cell (fixed-size array of length NbFeatures).
	typedef Eigen::Array<Scalar, NbFeatures, 1> Cell;
	
	/// Type of a pyramid level (matrix of cells).
	typedef Eigen::Matrix<Cell, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Level;
	
	/// Constructs an empty pyramid. An empty pyramid has no level.
	HOGPyramid();
	
	/// Constructs a pyramid from the JPEGImage of a Scene.
	/// @param[in] image The JPEGImage of the Scene.
	/// @param[in] padx Amount of horizontal zero padding (in cells).
	/// @param[in] pady Amount of vertical zero padding (in cells).
	/// @param[in] interval Number of levels per octave in the pyramid.
	/// @note The amount of padding and the interval should be at least 1.
	HOGPyramid(const JPEGImage & image, int padx, int pady, int interval = 5);
	
	/// Constructs a pyramid from parameters and a list of levels.
	/// @param[in] padx Amount of horizontal zero padding (in cells).
	/// @param[in] pady Amount of vertical zero padding (in cells).
	/// @param[in] interval Number of levels per octave in the pyramid.
	/// @param[in] levels List of pyramid levels.
	/// @note The amount of padding and the interval must both be at least 1.
	/// @note The input levels are swapped with empty ones on exit.
	HOGPyramid(int padx, int pady, int interval, std::vector<Level> & levels);
	
	/// Returns whether the pyramid is empty. An empty pyramid has no level.
	bool empty() const;
	
	/// Returns the amount of horizontal zero padding (in cells).
	int padx() const;
	
	/// Returns the amount of vertical zero padding (in cells).
	int pady() const;
	
	/// Returns the number of levels per octave in the pyramid.
	int interval() const;
	
	/// Returns the pyramid levels.
	/// @note Scales are given by the following formula: 2^(1 - @c index / @c interval).
	const std::vector<Level> & levels() const;
	
	/// Returns the convolutions of the pyramid with a filter.
	/// @param[in] filter Filter.
	/// @param[out] convolutions Convolution of each level.
	void convolve(const Level & filter, std::vector<Matrix> & convolutions) const;
	
	/// Returns the flipped version (horizontally) of a level.
	static HOGPyramid::Level Flip(const HOGPyramid::Level & level);
	
	/// Maps a pyramid level to a simple matrix (useful to apply standard matrix operations to it).
	/// @note The size of the matrix will be rows x (cols * NbFeatures).
	static Eigen::Map<Matrix, Eigen::Aligned> Map(Level & level);
	
	/// Maps a const pyramid level to a simple const matrix (useful to apply standard matrix
	/// operations to it).
	/// @note The size of the matrix will be rows x (cols * NbFeatures).
	static const Eigen::Map<const Matrix, Eigen::Aligned> Map(const Level & level);
	
private:
	// Efficiently computes Histogram of Oriented Gradient (HOG) features
	// Code to compute HOG features as described in "Object Detection with Discriminatively Trained
	// Part Based Models" by Felzenszwalb, Girshick, McAllester and Ramanan, PAMI 2010
	static void Hog(const JPEGImage & image, Level & level, int padx = 1, int pady = 1,
					int cellSize = 8);
	
	// Computes the 2D convolution of a pyramid level with a filter
	static void Convolve(const Level & x, const Level & y, Matrix & z);
	
	int padx_;
	int pady_;
	int interval_;
	std::vector<Level> levels_;
};

/// Serializes a pyramid to a stream.
std::ostream & operator<<(std::ostream & os, const HOGPyramid & pyramid);

/// Unserializes a pyramid from a stream.
std::istream & operator>>(std::istream & is, HOGPyramid & pyramid);
}

#endif

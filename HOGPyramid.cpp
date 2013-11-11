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

#include "HOGPyramid.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

using namespace Eigen;
using namespace FFLD;
using namespace std;

HOGPyramid::HOGPyramid() : padx_(0), pady_(0), interval_(0)
{
}

HOGPyramid::HOGPyramid(const JPEGImage & image, int padx, int pady, int interval) : padx_(0),
pady_(0), interval_(0)
{
	if (image.empty() || (padx < 1) || (pady < 1) || (interval < 1)) {
		cerr << "Attempting to create an empty pyramid" << endl;
		return;
	}
	
	// Compute the number of scales such that the smallest size of the last level is 5
	const int maxScale = ceil(log(min(image.width(), image.height()) / 40.0) / log(2.0) * interval);
	
	// Cannot compute the pyramid on images too small
	if (maxScale < interval) {
		cerr << "The image is too small to create a pyramid" << endl;
		return;
	}
	
	padx_ = padx;
	pady_ = pady;
	interval_ = interval;
	levels_.resize(maxScale + 1);
	
#pragma omp parallel for
	for (int i = 0; i < interval; ++i) {
		const double scale = pow(2.0, -static_cast<double>(i) / interval);
		
		JPEGImage scaled = image.rescale(scale);
		
		// First octave at twice the image resolution
		Hog(scaled, levels_[i], padx, pady, 4);
		
		// Second octave at the original resolution
		if (i + interval <= maxScale)
			Hog(scaled, levels_[i + interval], padx, pady, 8);
		
		// Remaining octaves
		for (int j = 2; i + j * interval <= maxScale; ++j) {
			scaled = scaled.rescale(0.5);
			Hog(scaled, levels_[i + j * interval], padx, pady, 8);
		}
	}
}

HOGPyramid::HOGPyramid(int padx, int pady, int interval, vector<Level> & levels) : padx_(0),
pady_(0), interval_(0)
{
	if ((padx < 1) || (pady < 1) || (interval < 1)) {
		cerr << "Attempting to create an empty pyramid" << endl;
		return;
	}
	
	padx_ = padx;
	pady_ = pady;
	interval_ = interval;
	levels_.swap(levels);
}

bool HOGPyramid::empty() const
{
	return levels().empty();
}

int HOGPyramid::padx() const
{
	return padx_;
}

int HOGPyramid::pady() const
{
	return pady_;
}

int HOGPyramid::interval() const
{
	return interval_;
}

const vector<HOGPyramid::Level> & HOGPyramid::levels() const
{
	return levels_;
}

void HOGPyramid::convolve(const Level & filter, vector<Matrix> & convolutions) const
{
	convolutions.resize(levels_.size());
	
#pragma omp parallel for
	for (int i = 0; i < levels_.size(); ++i)
		Convolve(levels_[i], filter, convolutions[i]);
}

FFLD::HOGPyramid::Level HOGPyramid::Flip(const HOGPyramid::Level & level)
{
	// Symmetric features
	const int symmetry[NbFeatures] =
	{
		9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 17, 16, 15, 14, 13, 12, 11, 10, // Contrast-sensitive
		18, 26, 25, 24, 23, 22, 21, 20, 19, // Contrast-insensitive
		28, 27, 30, 29, // Texture
#ifndef FFLD_HOGPYRAMID_EXTRA_FEATURES
		31 // Truncation
#else
		31, 32, 33, 34, 35, 36, 37, 38, 39, 40, // Uniform LBP
		41, 42, 43, 44, 45, 46, // Color
		47 // Truncation
#endif
	};
	
	// Symmetric filter
	HOGPyramid::Level result(level.rows(), level.cols());
	
	for (int y = 0; y < level.rows(); ++y)
		for (int x = 0; x < level.cols(); ++x)
			for (int i = 0; i < NbFeatures; ++i)
				result(y, x)(i) = level(y, level.cols() - 1 - x)(symmetry[i]);
	
	return result;
}

Map<HOGPyramid::Matrix, Aligned> HOGPyramid::Map(Level & level)
{
	return Eigen::Map<Matrix, Aligned>(level.data()->data(), level.rows(),
									   level.cols() * NbFeatures);
}

const Map<const HOGPyramid::Matrix, Aligned> HOGPyramid::Map(const Level & level)
{
	return Eigen::Map<const Matrix, Aligned>(level.data()->data(), level.rows(),
											 level.cols() * NbFeatures);
}

namespace FFLD
{
namespace detail
{
struct HOGTable
{
	char bins[512][512][2];
	HOGPyramid::Scalar magnitudes[512][512][2];
	
	// Singleton pattern
	static const HOGTable & Singleton()
	{
		return Singleton_;
	}
	
private:
	// Singleton pattern
	HOGTable() throw ()
	{
		for (int dy = -255; dy <= 255; ++dy) {
			for (int dx = -255; dx <= 255; ++dx) {
				// Magnitude in the range [0, 1]
				const double magnitude = sqrt(dx * dx + dy * dy) / 255.0;
				
				// Angle in the range [-pi, pi]
				double angle = atan2(static_cast<double>(dy), static_cast<double>(dx));
				
				// Convert it to the range [9.0, 27.0]
				angle = angle * (9.0 / M_PI) + 18.0;
				
				// Convert it to the range [0, 18)
				if (angle >= 18.0)
					angle -= 18.0;
				
				// Bilinear interpolation
				const int bin0 = angle;
				const int bin1 = (bin0 < 17) ? (bin0 + 1) : 0;
				const double alpha = angle - bin0;
				
				bins[dy + 255][dx + 255][0] = bin0;
				bins[dy + 255][dx + 255][1] = bin1;
				magnitudes[dy + 255][dx + 255][0] = magnitude * (1.0 - alpha);
				magnitudes[dy + 255][dx + 255][1] = magnitude * alpha;
			}
		}
	}
	
	// Singleton pattern
	HOGTable(const HOGTable &) throw ();
	void operator=(const HOGTable &) throw ();
	
	static const HOGTable Singleton_;
};

const HOGTable HOGTable::Singleton_;
}
}

void HOGPyramid::Hog(const JPEGImage & image, Level & level, int padx, int pady, int cellSize)
{
	// Get all the image members
	const int width = image.width();
	const int height = image.height();
	const int depth = image.depth();
	
	// Make sure the image is big enough
	if ((width < cellSize) || (height < cellSize) || (depth < 1) || (padx < 1) || (pady < 1) ||
		(cellSize < 1)) {
		level.swap(Level());
		cerr << "Attempting to compute an empty pyramid level" << endl;
		return;
	}
	
	// Resize the feature matrix
	level = Level::Constant((height + cellSize / 2) / cellSize + 2 * pady,
							(width + cellSize / 2) / cellSize + 2 * padx, Cell::Zero());
	
	const Scalar invCellSize = static_cast<Scalar>(1) / cellSize;
	
	for (int y = 0; y < height; ++y) {
		const uint8_t * linem = image.scanLine(max(y - 1, 0));
		const uint8_t * line = image.scanLine(y);
		const uint8_t * linep = image.scanLine(min(y + 1, height - 1));
		
		for (int x = 0; x < width; ++x) {
			// Use the channel with the largest gradient magnitude
			int maxMagnitude = 0;
			int argDx = 255;
			int argDy = 255;
			
			for (int i = 0; i < depth; ++i) {
				const int dx = static_cast<int>(line[min(x + 1, width - 1) * depth + i]) -
							   static_cast<int>(line[max(x - 1, 0) * depth + i]);
				const int dy = static_cast<int>(linep[x * depth + i]) -
							   static_cast<int>(linem[x * depth + i]);
				
				if (dx * dx + dy * dy > maxMagnitude) {
					maxMagnitude = dx * dx + dy * dy;
					argDx = dx + 255;
					argDy = dy + 255;
				}
			}
			
			const char bin0 = detail::HOGTable::Singleton().bins[argDy][argDx][0];
			const char bin1 = detail::HOGTable::Singleton().bins[argDy][argDx][1];
			const Scalar magnitude0 = detail::HOGTable::Singleton().magnitudes[argDy][argDx][0];
			const Scalar magnitude1 = detail::HOGTable::Singleton().magnitudes[argDy][argDx][1];
			
			// Bilinear interpolation
			const Scalar xp = (x + static_cast<Scalar>(0.5)) * invCellSize + padx - 0.5f;
			const Scalar yp = (y + static_cast<Scalar>(0.5)) * invCellSize + pady - 0.5f;
			const int ixp = xp;
			const int iyp = yp;
			const Scalar xp0 = xp - ixp;
			const Scalar yp0 = yp - iyp;
			const Scalar xp1 = 1 - xp0;
			const Scalar yp1 = 1 - yp0;
			
			level(iyp    , ixp    )(bin0) += xp1 * yp1 * magnitude0;
			level(iyp    , ixp    )(bin1) += xp1 * yp1 * magnitude1;
			level(iyp    , ixp + 1)(bin0) += xp0 * yp1 * magnitude0;
			level(iyp    , ixp + 1)(bin1) += xp0 * yp1 * magnitude1;
			level(iyp + 1, ixp    )(bin0) += xp1 * yp0 * magnitude0;
			level(iyp + 1, ixp    )(bin1) += xp1 * yp0 * magnitude1;
			level(iyp + 1, ixp + 1)(bin0) += xp0 * yp0 * magnitude0;
			level(iyp + 1, ixp + 1)(bin1) += xp0 * yp0 * magnitude1;
			
#ifdef FFLD_HOGPYRAMID_EXTRA_FEATURES
			// Normalize by the number of pixels
			const Scalar normalization = 2.0 / (cellSize * cellSize);
			
			// Texture (Uniform LBP) features
			const int LBP_TABLE[256] =
			{
				0, 1, 1, 2, 1, 9, 2, 3, 1, 9, 9, 9, 2, 9, 3, 4, 1, 9, 9, 9, 9, 9, 9, 9,
				2, 9, 9, 9, 3, 9, 4, 5, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
				2, 9, 9, 9, 9, 9, 9, 9, 3, 9, 9, 9, 4, 9, 5, 6, 1, 9, 9, 9, 9, 9, 9, 9,
				9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
				2, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 3, 9, 9, 9, 9, 9, 9, 9,
				4, 9, 9, 9, 5, 9, 6, 7, 1, 2, 9, 3, 9, 9, 9, 4, 9, 9, 9, 9, 9, 9, 9, 5,
				9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 6, 9, 9, 9, 9, 9, 9, 9, 9,
				9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 7,
				2, 3, 9, 4, 9, 9, 9, 5, 9, 9, 9, 9, 9, 9, 9, 6, 9, 9, 9, 9, 9, 9, 9, 9,
				9, 9, 9, 9, 9, 9, 9, 7, 3, 4, 9, 5, 9, 9, 9, 6, 9, 9, 9, 9, 9, 9, 9, 7,
				4, 5, 9, 6, 9, 9, 9, 7, 5, 6, 9, 7, 6, 7, 7, 8
			};
			
			// Use the green channel if available
			const uint8_t g = line[x * depth + (depth > 1)];
			
			const int lbp = (static_cast<int>(linem[xm * depth + (depth > 1)] >= g)     ) |
							(static_cast<int>(linem[x  * depth + (depth > 1)] >= g) << 1) |
							(static_cast<int>(linem[xp * depth + (depth > 1)] >= g) << 2) |
							(static_cast<int>(line[ xp * depth + (depth > 1)] >= g) << 3) |
							(static_cast<int>(linep[xp * depth + (depth > 1)] >= g) << 4) |
							(static_cast<int>(linep[x  * depth + (depth > 1)] >= g) << 5) |
							(static_cast<int>(linep[xm * depth + (depth > 1)] >= g) << 6) |
							(static_cast<int>(line[ xm * depth + (depth > 1)] >= g) << 7);
			
			// Bilinear interpolation
			level(iyp    , ixp    )(LBP_TABLE[lbp] + 31) += xp1 * yp1 * normalization;
			level(iyp    , ixp + 1)(LBP_TABLE[lbp] + 31) += xp0 * yp1 * normalization;
			level(iyp + 1, ixp    )(LBP_TABLE[lbp] + 31) += xp1 * yp0 * normalization;
			level(iyp + 1, ixp + 1)(LBP_TABLE[lbp] + 31) += xp0 * yp0 * normalization;
			
			// Color features
			if (depth >= 3) {
				const Scalar r = line[x * depth + 0] * static_cast<Scalar>(1.0 / 255.0);
				const Scalar g = line[x * depth + 1] * static_cast<Scalar>(1.0 / 255.0);
				const Scalar b = line[x * depth + 2] * static_cast<Scalar>(1.0 / 255.0);
				
				const Scalar minRGB = min(r, min(g, b));
				const Scalar maxRGB = max(r, max(g, b));
				const Scalar chroma = maxRGB - minRGB;
				
				if (chroma > 0.05) {
					Scalar hue = 0;
					
					if (r == maxRGB)
						hue = (g - b) / chroma;
					else if (g == maxRGB)
						hue = (b - r) / chroma + 2;
					else
						hue = (r - g) / chroma + 4;
					
					if (hue < 0)
						hue += 6;
					else if (hue >= 6)
						hue = 0;
					
					const Scalar saturation = chroma / maxRGB;
					
					// Bilinear interpolation
					const int bin0 = hue;
					const int bin1 = (hue0 < 5) ? (hue0 + 1) : 0;
					const Scalar alpha = hue - bin0;
					const Scalar magnitude0 = saturation * normalization * (1 - alpha);
					const Scalar magnitude1 = saturation * normalization * alpha;
					
					level(iyp    , ixp    )(bin0 + 41) += xp1 * yp1 * magnitude0;
					level(iyp    , ixp    )(bin1 + 41) += xp1 * yp1 * magnitude1;
					level(iyp    , ixp + 1)(bin0 + 41) += xp0 * yp1 * magnitude0;
					level(iyp    , ixp + 1)(bin1 + 41) += xp0 * yp1 * magnitude1;
					level(iyp + 1, ixp    )(bin0 + 41) += xp1 * yp0 * magnitude0;
					level(iyp + 1, ixp    )(bin1 + 41) += xp1 * yp0 * magnitude1;
					level(iyp + 1, ixp + 1)(bin0 + 41) += xp0 * yp0 * magnitude0;
					level(iyp + 1, ixp + 1)(bin1 + 41) += xp0 * yp0 * magnitude1;
				}
			}
#endif
		}
	}
	
	// Compute the "gradient energy" of each cell, i.e. ||C(i,j)||^2
	for (int y = 0; y < level.rows(); ++y) {
		for (int x = 0; x < level.cols(); ++x) {
			Scalar sumSq = 0;
			
			for (int i = 0; i < 9; ++i)
				sumSq += (level(y, x)(i) + level(y, x)(i + 9)) *
						 (level(y, x)(i) + level(y, x)(i + 9));
			
			level(y, x)(NbFeatures - 1) = sumSq;
		}
	}
	
	// Compute the four normalization factors then normalize and clamp everything
	const Scalar EPS = numeric_limits<Scalar>::epsilon();
	
	for (int y = pady; y < level.rows() - pady; ++y) {
		for (int x = padx; x < level.cols() - padx; ++x) {
			const Scalar n0 = 1 / sqrt(level(y - 1, x - 1)(NbFeatures - 1) +
									   level(y - 1, x    )(NbFeatures - 1) +
									   level(y    , x - 1)(NbFeatures - 1) +
									   level(y    , x    )(NbFeatures - 1) + EPS);
			const Scalar n1 = 1 / sqrt(level(y - 1, x    )(NbFeatures - 1) +
									   level(y - 1, x + 1)(NbFeatures - 1) +
									   level(y    , x    )(NbFeatures - 1) +
									   level(y    , x + 1)(NbFeatures - 1) + EPS);
			const Scalar n2 = 1 / sqrt(level(y    , x - 1)(NbFeatures - 1) +
									   level(y    , x    )(NbFeatures - 1) +
									   level(y + 1, x - 1)(NbFeatures - 1) +
									   level(y + 1, x    )(NbFeatures - 1) + EPS);
			const Scalar n3 = 1 / sqrt(level(y    , x    )(NbFeatures - 1) +
									   level(y    , x + 1)(NbFeatures - 1) +
									   level(y + 1, x    )(NbFeatures - 1) +
									   level(y + 1, x + 1)(NbFeatures - 1) + EPS);
			
			// Contrast-insensitive features
			for (int i = 0; i < 9; ++i) {
				const Scalar sum = level(y, x)(i) + level(y, x)(i + 9);
				const Scalar h0 = min(sum * n0, static_cast<Scalar>(0.2));
				const Scalar h1 = min(sum * n1, static_cast<Scalar>(0.2));
				const Scalar h2 = min(sum * n2, static_cast<Scalar>(0.2));
				const Scalar h3 = min(sum * n3, static_cast<Scalar>(0.2));
				level(y, x)(i + 18) = (h0 + h1 + h2 + h3) * static_cast<Scalar>(0.5);
			}
			
			// Contrast-sensitive features
			Scalar t0 = 0;
			Scalar t1 = 0;
			Scalar t2 = 0;
			Scalar t3 = 0;
			
			for (int i = 0; i < 18; ++i) {
				const Scalar sum = level(y, x)(i);
				const Scalar h0 = min(sum * n0, static_cast<Scalar>(0.2));
				const Scalar h1 = min(sum * n1, static_cast<Scalar>(0.2));
				const Scalar h2 = min(sum * n2, static_cast<Scalar>(0.2));
				const Scalar h3 = min(sum * n3, static_cast<Scalar>(0.2));
				level(y, x)(i) = (h0 + h1 + h2 + h3) * static_cast<Scalar>(0.5);
				t0 += h0;
				t1 += h1;
				t2 += h2;
				t3 += h3;
			}
			
			// Texture features
			level(y, x)(27) = t0 * static_cast<Scalar>(0.2357);
			level(y, x)(28) = t1 * static_cast<Scalar>(0.2357);
			level(y, x)(29) = t2 * static_cast<Scalar>(0.2357);
			level(y, x)(30) = t3 * static_cast<Scalar>(0.2357);
		}
	}
	
	// Truncation features
	for (int y = 0; y < level.rows(); ++y) {
		for (int x = 0; x < level.cols(); ++x) {
			if ((y < pady) || (y >= level.rows() - pady) || (x < padx) ||
				(x >= level.cols() - padx)) {
				level(y, x).setZero();
				level(y, x)(NbFeatures - 1) = 1;
			}
			else {
				level(y, x)(NbFeatures - 1) = 0;
			}
		}
	}
}

void HOGPyramid::Convolve(const Level & x, const Level & y, Matrix & z)
{
	// Nothing to do if x is smaller than y
	if ((x.rows() < y.rows()) || (x.cols() < y.cols())) {
		z = Matrix();
		return;
	}
	
	z = Matrix::Zero(x.rows() - y.rows() + 1, x.cols() - y.cols() + 1);
	
	for (int i = 0; i < z.rows(); ++i) {
		for (int j = 0; j < y.rows(); ++j) {
			const Eigen::Map<const Matrix, Aligned, OuterStride<NbFeatures> >
				mapx(reinterpret_cast<const Scalar *>(x.row(i + j).data()), z.cols(),
					 y.cols() * NbFeatures);
#ifndef FFLD_HOGPYRAMID_DOUBLE
			const Eigen::Map<const RowVectorXf, Aligned>
#else
			const Eigen::Map<const RowVectorXd, Aligned>
#endif
				mapy(reinterpret_cast<const Scalar *>(y.row(j).data()), y.cols() * NbFeatures);
			
			z.row(i).noalias() += mapy * mapx.transpose();
		}
	}
}

ostream & FFLD::operator<<(ostream & os, const HOGPyramid & pyramid)
{
	os << pyramid.padx() << ' ' << pyramid.pady() << ' ' << pyramid.interval() << ' '
	   << pyramid.levels().size() << endl;
	
	for (int i = 0; i < pyramid.levels().size(); ++i) {
		os << pyramid.levels()[i].rows() << ' ' << pyramid.levels()[i].cols() << ' '
		   << HOGPyramid::NbFeatures << ' ';
		
		for (int y = pyramid.pady(); y < pyramid.levels()[i].rows() - pyramid.pady(); ++y)
			os.write(reinterpret_cast<const char *>(pyramid.levels()[i].row(y).data() +
													pyramid.padx()),
					 (pyramid.levels()[i].cols() - 2 * pyramid.padx()) * sizeof(HOGPyramid::Cell));
		
		os << endl;
	}
	
	return os;
}

istream & FFLD::operator>>(istream & is, HOGPyramid & pyramid)
{
	int padx, pady, interval, nbLevels;
	
	is >> padx >> pady >> interval >> nbLevels;
	
	if (!is) {
		pyramid = HOGPyramid();
		return is;
	}
	
	vector<HOGPyramid::Level> levels(nbLevels);
	
	for (int i = 0; i < nbLevels; ++i) {
		int rows, cols, nbFeatures;
		
		is >> rows >> cols >> nbFeatures;
		
		is.get(); // Remove the space
		
		if (!is || (nbFeatures > HOGPyramid::NbFeatures)) {
			pyramid = HOGPyramid();
			return is;
		}
		
		levels[i] = HOGPyramid::Level::Constant(rows, cols, HOGPyramid::Cell::Zero());
		
		for (int y = 0; y < rows; ++y)
			for (int x = 0; x < cols; ++x)
				levels[i](y, x)(HOGPyramid::NbFeatures - 1) = 1;
		
		for (int y = pady; y < rows - pady; ++y)
			is.read(reinterpret_cast<char *>(levels[i].row(y).data() + padx),
					(cols - 2 * padx) * nbFeatures * sizeof(HOGPyramid::Scalar));
		
		if (!is) {
			pyramid = HOGPyramid();
			return is;
		}
	}
	
	pyramid = HOGPyramid(padx, pady, interval, levels);
	
	return is;
}

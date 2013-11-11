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

#ifndef FFLD_JPEGIMAGE_H
#define FFLD_JPEGIMAGE_H

#include <iosfwd>
#include <string>
#include <vector>
#include <stdint.h>

namespace FFLD
{
/// The JPEGImage class allows to load/save an image from/to a jpeg file, as well as to resize it.
/// The pixels are always stored contiguously in row-major order:
/// @code
///     scanline 0: RGB RGB RGB
///     scanline 1: RGB RGB RGB
///     ...
/// @endcode
/// without any padding.
class JPEGImage
{
public:
	/// Constructs an empty image. An empty image has zero size.
	JPEGImage();
	
	/// Constructs an image with the given @p width, @p height and @p depth, and initializes it from
	/// the given @p bits.
	/// @note The returned image might be empty if any of the parameters is incorrect.
	JPEGImage(int width, int height, int depth, const uint8_t * bits = 0);
	
	/// Constructs an image and tries to load the image from the jpeg file with the given
	/// @p filename.
	/// @note The returned image might be empty if the image could not be loaded.
	JPEGImage(const std::string & filename);
	
	/// Returns whether the image is empty. An empty image has zero size.
	bool empty() const;
	
	/// Returns the width of the image.
	int width() const;
	
	/// Returns the height of the image.
	int height() const;
	
	/// Returns the depth of the image. The image depth is the number of color channels.
	int depth() const;
	
	/// Returns a pointer to the pixel data. Returns a null pointer if the image is empty.
	const uint8_t * bits() const;
	
	/// Returns a pointer to the pixel data. Returns a null pointer if the image is empty.
	uint8_t * bits();
	
	/// Returns a pointer to the pixel data at the scanline with index y. The first scanline is at
	/// index 0. Returns a null pointer if the image is empty or if y is out of bounds.
	const uint8_t * scanLine(int y) const;
	
	/// Returns a pointer to the pixel data at the scanline with index y. The first scanline is at
	/// index 0. Returns a null pointer if the image is empty or if y is out of bounds.
	uint8_t * scanLine(int y);
	
	/// Saves the image to a jpeg file with the given @p filename and @p quality.
	void save(const std::string & filename, int quality = 100) const;
	
	/// Returns a copy of the image scaled to @scale. If the scale is zero or negative, the method
	/// returns an empty image.
	JPEGImage rescale(double scale) const;
	
private:
	int width_;
	int height_;
	int depth_;
	std::vector<uint8_t> bits_;
};

/// Serializes an image to a stream.
std::ostream & operator<<(std::ostream & os, const JPEGImage & image);

/// Unserializes an image from a stream.
std::istream & operator>>(std::istream & is, JPEGImage & image);
}

#endif

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

#include "JPEGImage.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>

#include <jpeglib.h>
#include <stdio.h>

using namespace FFLD;
using namespace std;

JPEGImage::JPEGImage() : width_(0), height_(0), depth_(0)
{
}

JPEGImage::JPEGImage(int width, int height, int depth, const uint8_t * bits) : width_(0),
height_(0), depth_(0)
{
	if ((width <= 0) || (height <= 0) || (depth <= 0)) {
		cerr << "Attempting to create an empty image" << endl;
		return;
	}
	
	width_ = width;
	height_ = height;
	depth_ = depth;
	bits_.resize(width * height * depth);
	
	if (bits)
		copy(bits, bits + bits_.size(), bits_.begin());
}

JPEGImage::JPEGImage(const string & filename) : width_(0), height_(0), depth_(0)
{
	// Load the image
	FILE * file = fopen(filename.c_str(), "rb");
	
	if (!file) {
		cerr << "Could not open " << filename << endl;
		return;
	}
	
	jpeg_decompress_struct cinfo;
	jpeg_error_mgr jerr;
	
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&cinfo);
	jpeg_stdio_src(&cinfo, file);
	
	if ((jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) || (cinfo.data_precision != 8) ||
		!jpeg_start_decompress(&cinfo)) {
		fclose(file);
		cerr << filename << " is not an 8-bit jpeg image" << endl;
		return;
	}
	
	vector<uint8_t> bits(cinfo.image_width * cinfo.image_height * cinfo.num_components);
	
	for (int y = 0; y < cinfo.image_height; ++y) {
		JSAMPLE * row = static_cast<JSAMPLE *>(&bits[y * cinfo.image_width * cinfo.num_components]);
		
		if (jpeg_read_scanlines(&cinfo, &row, 1) != 1) {
			fclose(file);
			cerr << "Error while loading " << filename << endl;
			return;
		}
	}
	
	jpeg_finish_decompress(&cinfo);
	
	fclose(file);
	
	// Recopy everyting if the loading was successful
	width_ = cinfo.image_width;
	height_ = cinfo.image_height;
	depth_ = cinfo.num_components;
	bits_.swap(bits);
}

bool JPEGImage::empty() const
{
	return (width() <= 0) || (height() <= 0) || (depth() <= 0);
}

int JPEGImage::width() const
{
	return width_;
}

int JPEGImage::height() const
{
	return height_;
}

int JPEGImage::depth() const
{
	return depth_;
}

const uint8_t * JPEGImage::bits() const
{
	return empty() ? 0 : &bits_[0];
}

uint8_t * JPEGImage::bits()
{
	return empty() ? 0 : &bits_[0];
}

const uint8_t * JPEGImage::scanLine(int y) const
{
	return (empty() || (y >= height_)) ? 0 : &bits_[y * width_ * depth_];
}

uint8_t * JPEGImage::scanLine(int y)
{
	return (empty() || (y >= height_)) ? 0 : &bits_[y * width_ * depth_];
}

void JPEGImage::save(const string & filename, int quality) const
{
	if (empty())
		return;
	
	FILE * file = fopen(filename.c_str(), "wb");
	
	if (!file) {
		cerr << "Could not open " << filename << endl;
		return;
	}
	
	jpeg_compress_struct cinfo;
	jpeg_error_mgr jerr;
	
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_compress(&cinfo);
	jpeg_stdio_dest(&cinfo, file);
	
	cinfo.image_width = width_;
	cinfo.image_height = height_;
	cinfo.input_components = depth_;
	cinfo.in_color_space = (depth_ == 1) ? JCS_GRAYSCALE : JCS_RGB;
	
	jpeg_set_defaults(&cinfo);
	jpeg_set_quality(&cinfo, quality, TRUE);
	jpeg_start_compress(&cinfo, TRUE);
	
	for (int y = 0; y < height_; ++y) {
		const JSAMPLE * row = static_cast<const JSAMPLE *>(&bits_[y * width_ * depth_]);
		jpeg_write_scanlines(&cinfo, const_cast<JSAMPARRAY>(&row), 1);
	}
	
	jpeg_finish_compress(&cinfo);
	
	fclose(file);
}

// Bilinear interpolation coefficient
namespace FFLD
{
namespace detail
{
struct Bilinear
{
	int x0;
	int x1;
	double a;
	double b;
};
}
}

JPEGImage JPEGImage::rescale(double scale) const
{
	// Empty image
	if (scale <= 0.0)
		return JPEGImage();
	
	// Same scale
	if (scale == 1.0)
		return *this;
	
	// Scale below 0.5
	if (scale < 0.5)
		return rescale(0.5).rescale(2.0 * scale);
	
	const int width = ceil(width_ * scale);
	const int height = ceil(height_ * scale);
	
	JPEGImage result;
	
	result.width_ = width;
	result.height_ = height;
	result.depth_ = depth_;
	result.bits_.resize(width * height * depth_);
	
	// Half scale
	if (scale == 0.5) {
		for (int i = 0; i < height; ++i) {
			const int i2 = min(2 * i + 1, height_ - 1);
			
			for (int j = 0; j < width; ++j) {
				const int j2 = min(2 * j + 1, width_ - 1);
				
				for (int k = 0; k < depth_; ++k)
					result.bits_[(i * width + j) * depth_ + k] =
						(2 + bits_[(2 * i * width_ + 2 * j) * depth_ + k] +
   							 bits_[(2 * i * width_ + j2   ) * depth_ + k] +
							 bits_[(i2    * width_ + 2 * j) * depth_ + k] +
							 bits_[(i2    * width_ + j2   ) * depth_ + k]) >> 2;
			}
		}
		
		return result;
	}
	
	// Bilinear interpolation coefficients
	vector<detail::Bilinear> cols(width);
	
	for (int j = 0; j < width; ++j) {
		const double x = min(max((j + 0.5) / scale - 0.5, 0.0), width_ - 1.0);
		cols[j].x0 = x;
		cols[j].x1 = min(cols[j].x0 + 1, width_ - 1);
		cols[j].a = x - cols[j].x0;
		cols[j].b = 1.0 - cols[j].a;
	}
	
	for (int i = 0; i < height; ++i) {
		const double y = min(max((i + 0.5) / scale - 0.5, 0.0), height_ - 1.0);
		const int y0 = y;
		const int y1 = min(y0 + 1, height_ - 1);
		const double c = y - y0;
		const double d = 1.0 - c;
		
		for (int j = 0; j < width; ++j)
			for (int k = 0; k < depth_; ++k)
				result.bits_[(i * width + j) * depth_ + k] =
					(bits_[(y0 * width_ + cols[j].x0) * depth_ + k] * cols[j].b +
					 bits_[(y0 * width_ + cols[j].x1) * depth_ + k] * cols[j].a) * d +
					(bits_[(y1 * width_ + cols[j].x0) * depth_ + k] * cols[j].b +
					 bits_[(y1 * width_ + cols[j].x1) * depth_ + k] * cols[j].a) * c + 0.5;
	}
	
	return result;
}

ostream & FFLD::operator<<(ostream & os, const JPEGImage & image)
{
	os << image.width() << ' ' << image.height() << ' ' << image.depth() << ' ';
	
	os.write(reinterpret_cast<const char *>(image.bits()),
			 image.width() * image.height() * image.depth());
	
	return os << endl;
}

istream & FFLD::operator>>(istream & is, JPEGImage & image)
{
	int width, height, depth;
	
	is >> width >> height >> depth;
	
	is.get(); // Remove the space
	
	if (!is || (width <= 0) || (height <= 0) || (depth <= 0)) {
		image = JPEGImage();
		return is;
	}
	
	image = JPEGImage(width, height, depth);
	
	is.read(reinterpret_cast<char *>(image.bits()), width * height * depth);
	
	if (!is)
		image = JPEGImage();
	
	return is;
}

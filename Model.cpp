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

#include "Model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <iostream>

using namespace Eigen;
using namespace FFLD;
using namespace std;

Model::Model() : parts_(1), bias_(0.0)
{
	parts_[0].offset.setZero();
	parts_[0].deformation.setZero();
}

Model::Model(pair<int, int> rootSize, int nbParts, pair<int, int> partSize) : parts_(1), bias_(0.0)
{
	parts_[0].offset.setZero();
	parts_[0].deformation.setZero();
	
	// Create an empty model if any of the given parameters is invalid
	if ((rootSize.first <= 0) || (rootSize.second <= 0) || (nbParts < 0) ||
		(nbParts && ((partSize.first <= 0) || (partSize.second <= 0)))) {
		cerr << "Attempting to create an empty model" << endl;
		return;
	}
	
	parts_.resize(nbParts + 1);
	
	parts_[0].filter = HOGPyramid::Level::Constant(rootSize.first, rootSize.second,
												   HOGPyramid::Cell::Zero());
	
	for (int i = 0; i < nbParts; ++i) {
		parts_[i + 1].filter = HOGPyramid::Level::Constant(partSize.first, partSize.second,
														   HOGPyramid::Cell::Zero());
		parts_[i + 1].offset.setZero();
		parts_[i + 1].deformation.setZero();
	}
}

Model::Model(const vector<Part> & parts, double bias) : parts_(parts), bias_(bias)
{
	// Create an empty model if the parts are empty
	if (parts.empty()) {
		parts_.resize(1);
		parts_[0].offset.setZero();
		parts_[0].deformation.setZero();
	}
}

bool Model::empty() const
{
	return !parts_[0].filter.size() && (parts_.size() == 1);
}

const vector<Model::Part> & Model::parts() const
{
	return parts_;
}

vector<Model::Part> & Model::parts()
{
	return parts_;
}

double Model::bias() const
{
	return bias_;
}

double & Model::bias()
{
	return bias_;
}

pair<int, int> Model::rootSize() const
{
	return pair<int, int>(static_cast<int>(parts_[0].filter.rows()),
						  static_cast<int>(parts_[0].filter.cols()));
}

pair<int, int> Model::partSize() const
{
	if (parts_.size() > 1)
		return pair<int, int>(static_cast<int>(parts_[1].filter.rows()),
							  static_cast<int>(parts_[1].filter.cols()));
	else
		return pair<int, int>(0, 0);
}

void Model::initializeParts(int nbParts, pair<int, int> partSize)
{
	// The model stay unmodified if any of the parameter is invalid
	if (empty() || (nbParts <= 0) || (partSize.first <= 0) || (partSize.second <= 0)) {
		cerr << "Attempting to initialize parts in an empty model" << endl;
		return;
	}
	
	// Upsample the root filter by a factor 2 using bicubic interpolation
	const HOGPyramid::Level & root = parts_[0].filter;
	
	HOGPyramid::Level root2x = HOGPyramid::Level::Constant(2 * root.rows(), 2 * root.cols(),
														   HOGPyramid::Cell::Zero());
	
	// Bicubic interpolation matrix for x = y = 0.25
	const double bicubic[4][4] =
	{
		{ 0.004943847656,-0.060974121094,-0.015930175781, 0.001647949219},
		{-0.060974121094, 0.752014160156, 0.196472167969,-0.020324707031},
		{-0.015930175781, 0.196472167969, 0.051330566406,-0.005310058594},
		{ 0.001647949219,-0.020324707031,-0.005310058594, 0.000549316406}
	};
	
	for (int y = 0; y < root.rows(); ++y) {
		for (int x = 0; x < root.cols(); ++x) {
			for (int i = 0; i < 4; ++i) {
				for (int j = 0; j < 4; ++j) {
					const int y2 = min(max(y + i - 2, 0), static_cast<int>(root.rows()) - 1);
					const int y1 = min(max(y + i - 1, 0), static_cast<int>(root.rows()) - 1);
					const int x2 = min(max(x + j - 2, 0), static_cast<int>(root.cols()) - 1);
					const int x1 = min(max(x + j - 1, 0), static_cast<int>(root.cols()) - 1);
					
					root2x(y * 2    , x * 2    ) += bicubic[3 - i][3 - j] * root(y2, x2);
					root2x(y * 2    , x * 2 + 1) += bicubic[3 - i][    j] * root(y2, x1);
					root2x(y * 2 + 1, x * 2    ) += bicubic[    i][3 - j] * root(y1, x2);
					root2x(y * 2 + 1, x * 2 + 1) += bicubic[    i][    j] * root(y1, x1);
				}
			}
		}
	}
	
	// Compute the energy of each cell
	HOGPyramid::Matrix energy(root2x.rows(), root2x.cols());
	
	for (int y = 0; y < root2x.rows(); ++y) {
		for (int x = 0; x < root2x.cols(); ++x) {
			root2x(y, x).cwiseMax(0);
			
			energy(y, x) = 0;
			
			for (int i = 0; i < HOGPyramid::NbFeatures; ++i)
				energy(y, x) += root2x(y, x)(i) * root2x(y, x)(i);
		}
	}
	
	// Assign each part greedily to the region of maximum energy
	parts_.resize(nbParts + 1);
	
	for (int i = 0; i < nbParts; ++i) {
		double maxEnergy = 0.0;
		int argX = 0;
		int argY = 0;
		
		for (int y = 0; y <= energy.rows() - partSize.first; ++y) {
			for (int x = 0; x <= energy.cols() - partSize.second; ++x) {
				const double e = energy.block(y, x, partSize.first, partSize.second).sum();
				
				if (e > maxEnergy) {
					maxEnergy = e;
					argX = x;
					argY = y;
				}
			}
		}
		
		// Initialize the part
		parts_[i + 1].filter = root2x.block(argY, argX, partSize.first, partSize.second);
		parts_[i + 1].offset(0) = argX;
		parts_[i + 1].offset(1) = argY;
		parts_[i + 1].offset(2) = 0;
		
		// Set the energy of the part to zero
		energy.block(argY, argX, partSize.first, partSize.second).setZero();
	}
	
	// Retry 10 times from randomized starting points
	double bestCover = 0.0;
	vector<Part> best(parts_); // The best so far is the current one
	
	for (int i = 0; i < 10; ++i) {
		vector<Part> tmp(parts_); // Try from the current one
		
		// Remove a part at random and look for the best place to put it
		for (int j = 0; j < 100 * nbParts; ++j) {
			// Recompute the energy
			for (int y = 0; y < root2x.rows(); ++y) {
				for (int x = 0; x < root2x.cols(); ++x) {
					energy(y, x) = 0;
					
					for (int i = 0; i < HOGPyramid::NbFeatures; ++i)
						energy(y, x) += root2x(y, x)(i) * root2x(y, x)(i);
				}
			}
			
			// Select a part at random
			const int part = rand() % nbParts;
			
			// Zero out the energy covered by the other parts
			for (int k = 0; k < nbParts; ++k)
				if (k != part)
					energy.block(tmp[k + 1].offset(1), tmp[k + 1].offset(0), partSize.first,
								 partSize.second).setZero();
			
			// Find the region of maximum energy
			double maxEnergy = 0.0;
			int argX = 0;
			int argY = 0;
			
			for (int y = 0; y <= energy.rows() - partSize.first; ++y) {
				for (int x = 0; x <= energy.cols() - partSize.second; ++x) {
					const double e = energy.block(y, x, partSize.first, partSize.second).sum();
					
					if (e > maxEnergy) {
						maxEnergy = e;
						argX = x;
						argY = y;
					}
				}
			}
			
			// Initialize the part
			tmp[part + 1].filter = root2x.block(argY, argX, partSize.first, partSize.second);
			tmp[part + 1].offset(0) = argX;
			tmp[part + 1].offset(1) = argY;
			tmp[part + 1].offset(2) = 0;
		}
		
		// Compute the energy covered by this part arrangement
		double cover = 0.0;
		
		// Recompute the energy
		for (int y = 0; y < root2x.rows(); ++y) {
			for (int x = 0; x < root2x.cols(); ++x) {
				energy(y, x) = 0;
				
				for (int i = 0; i < HOGPyramid::NbFeatures; ++i)
					energy(y, x) += root2x(y, x)(i) * root2x(y, x)(i);
			}
		}
		
		for (int j = 0; j < nbParts; ++j) {
			// Add the energy of the part
			cover += energy.block(tmp[j + 1].offset(1), tmp[j + 1].offset(0), partSize.first,
								  partSize.second).sum();
			
			// Set the energy of the part to zero
			energy.block(tmp[j + 1].offset(1), tmp[j + 1].offset(0), partSize.first,
						 partSize.second).setZero();
		}
		
		if (cover > bestCover) {
			bestCover = cover;
			best = tmp;
		}
	}
	
	parts_.swap(best);
	
	// Initialize the deformations
	for (int i = 0; i < nbParts; ++i)
		parts_[i + 1].deformation << -0.01, 0.0, -0.01, 0.0, -0.01, 0.0;
}

void Model::initializeSample(const HOGPyramid & pyramid, int x, int y, int z, Model & sample,
							 const vector<vector<Positions> > * positions) const
{
	// All the constants relative to the model and the pyramid
	const int nbFilters = static_cast<int>(parts_.size());
	const int nbParts = nbFilters - 1;
	const int padx = pyramid.padx();
	const int pady = pyramid.pady();
	const int interval = pyramid.interval();
	const int nbLevels = static_cast<int>(pyramid.levels().size());
	
	// Invalid parameters
	if (empty() || (x < 0) || (y < 0) || (z < 0) || (z >= nbLevels) ||
		(x + rootSize().second > pyramid.levels()[z].cols()) ||
		(y + rootSize().first > pyramid.levels()[z].rows()) ||
		(nbParts && (!positions || (positions->size() != nbParts)))) {
		sample = Model();
		cerr << "Attempting to initialize an empty sample" << endl;
		return;
	}
	
	// Resize the sample to have the same number of filters as the model
	sample.parts_.resize(nbFilters);
	
	// Extract the root filter
	sample.parts_[0].filter = pyramid.levels()[z].block(y, x, rootSize().first, rootSize().second);
	sample.parts_[0].offset.setZero();
	sample.parts_[0].deformation.setZero();
	
	for (int i = 0; i < nbParts; ++i) {
		// Position of the part
		if ((z >= (*positions)[i].size()) || (x >= (*positions)[i][z].cols()) ||
			(y >= (*positions)[i][z].rows())) {
			sample = Model();
			cerr << "Attempting to initialize an empty sample" << endl;
			return;
		}
		
		const Position position = (*positions)[i][z](y, x);
		
		// Level of the part
		if ((position(2) < 0) || (position(2) >= nbLevels)) {
			sample = Model();
			cerr << "Attempting to initialize an empty sample" << endl;
			return;
		}
		
		const HOGPyramid::Level & level = pyramid.levels()[position(2)];
		
		if ((position(0) < 0) || (position(1) < 0) ||
			(position(0) + partSize().second > level.cols()) ||
			(position(1) + partSize().first > level.rows())) {
			sample = Model();
			cerr << "Attempting to initialize an empty sample" << endl;
			return;
		}
		
		// Extract the part filter
		sample.parts_[i + 1].filter = level.block(position(1), position(0), partSize().first,
												  partSize().second);
		
		// Set the part offset to the position
		sample.parts_[i + 1].offset = position;
		
		// Compute the deformation gradient at the level of the part
		const double scale = pow(2.0, static_cast<double>(z - position(2)) / interval);
		const double xr = (x + (parts_[i + 1].offset(0) + partSize().second * 0.5) * 0.5 - padx) *
						  scale + padx - partSize().second * 0.5;
		const double yr = (y + (parts_[i + 1].offset(1) + partSize().first * 0.5) * 0.5 - pady) *
						  scale + pady - partSize().first * 0.5;
		const double dx = xr - position(0);
		const double dy = yr - position(1);
		const int dz = z - interval - position(2);
		
		sample.parts_[i + 1].deformation(0) = dx * dx;
		sample.parts_[i + 1].deformation(1) = dx;
		sample.parts_[i + 1].deformation(2) = dy * dy;
		sample.parts_[i + 1].deformation(3) = dy;
		sample.parts_[i + 1].deformation(4) = dz * dz;
		sample.parts_[i + 1].deformation(5) = dz;
	}
	
	sample.bias_ = 1.0;
}

void Model::convolve(const HOGPyramid & pyramid, vector<HOGPyramid::Matrix> & scores,
					 vector<vector<Positions> > * positions,
					 vector<vector<HOGPyramid::Matrix> > * convolutions) const
{
	// Invalid parameters
	if (empty() || pyramid.empty() ||
#ifdef FFLD_MODEL_3D
		!positions ||
#endif
		(convolutions && (convolutions->size() != parts_.size()))) {
		scores.clear();
		
		if (positions)
			positions->clear();
		
		return;
	}
	
	// All the constants relative to the model and the pyramid
	const int nbFilters = static_cast<int>(parts_.size());
	const int nbParts = nbFilters - 1;
	const int padx = pyramid.padx();
	const int pady = pyramid.pady();
	const int interval = pyramid.interval();
	const int nbLevels = static_cast<int>(pyramid.levels().size());
	
	// Convolve the pyramid with all the filters
	vector<vector<HOGPyramid::Matrix> > tmpConvolutions;
	
	if (convolutions) {
		for (int i = 0; i < nbFilters; ++i) {
			if ((*convolutions)[i].size() != nbLevels) {
				scores.clear();
				
				if (positions)
					positions->clear();
				
				return;
			}
		}
	}
	else {
		tmpConvolutions.resize(nbFilters);
		
#pragma omp parallel for
		for (int i = 0; i < nbFilters; ++i)
			pyramid.convolve(parts_[i].filter, tmpConvolutions[i]);
		
		convolutions = &tmpConvolutions;
	}
	
	// Resize the positions
	if (positions) {
		positions->resize(nbParts);
		
		for (int i = 0; i < nbParts; ++i)
			(*positions)[i].resize(nbLevels);
	}
	
	// Temporary data needed by the distance transforms
	HOGPyramid::Matrix tmp;
	
#ifndef FFLD_MODEL_3D
	// For each root level in reverse order
	for (int z = nbLevels - 1; z >= interval; --z) {
		// For each part
		for (int i = 0; i < nbParts; ++i) {
			// Transform the part one octave below
			DT2D((*convolutions)[i + 1][z - interval], parts_[i + 1], tmp,
				 positions ? &(*positions)[i][z - interval] : 0);
			
			if (positions)
				(*positions)[i][z] = Positions::Constant((*convolutions)[0][z].rows(),
														 (*convolutions)[0][z].cols(),
														 Position::Zero());
			
			// Add the distance transforms of the part one octave below
			for (int y = 0; y < (*convolutions)[0][z].rows(); ++y) {
				for (int x = 0; x < (*convolutions)[0][z].cols(); ++x) {
					const int xr = 2 * x - padx + parts_[i + 1].offset(0);
					const int yr = 2 * y - pady + parts_[i + 1].offset(1);
					
					if ((xr >= 0) && (yr >= 0) &&
						(xr < (*convolutions)[i + 1][z - interval].cols()) &&
						(yr < (*convolutions)[i + 1][z - interval].rows())) {
						(*convolutions)[0][z](y, x) += (*convolutions)[i + 1][z - interval](yr, xr);
						
						if (positions)
							(*positions)[i][z](y, x) << (*positions)[i][z - interval](yr, xr)(0),
														(*positions)[i][z - interval](yr, xr)(1),
														z - interval;
					}
					else {
						(*convolutions)[0][z](y, x) =
							-numeric_limits<HOGPyramid::Scalar>::infinity();
					}
				}
			}
			
			if (positions)
				(*positions)[i][z - interval] = Positions();
		}
	}
	
	scores.swap((*convolutions)[0]);
	
	for (int i = 0; i < interval; ++i) {
		scores[i] = HOGPyramid::Matrix();
		
		for (int j = 0; j < nbParts; ++j)
			(*positions)[j][i] = Positions();
	}
	
	// Add the bias if necessary
	if (bias_) {
#pragma omp parallel for
		for (int i = interval; i < nbLevels; ++i)
			scores[i].array() += bias_;
	}
#else
	// Range of scales to consider
	const int interval2 = (interval + 1) / 2;
	
	// Precompute all the necessary distance transforms
	for (int i = 0; i < nbLevels - interval + interval2; ++i) {
		for (int j = 0; j < nbParts; ++j) {
			HOGPyramid::Matrix copy = (*convolutions)[j + 1][i];
			DT2D(copy, parts_[j + 1], tmp, positions ? &(*positions)[j][i] : 0);
		}
	}
	
	// Precompute the scale ratios
	vector<double> scales(2 * interval);
	
	for (int i = 0; i < 2 * interval; ++i)
		scales[i] = pow(2.0, static_cast<double>(i) / interval);
	
	// Half part size
	const double hpx = 0.5 * partSize().second;
	const double hpy = 0.5 * partSize().first;
	
	// Resize the scores
	scores.resize(nbLevels);
	
	// For each root level in reverse order
	for (int i = nbLevels - 1; i >= interval - interval2; --i) {
		// Set the scores to those of the root + bias
		scores[i].swap((*convolutions)[0][i]);
		
		for (int j = 0; j < nbParts; ++j)
			(*positions)[j][i] = Model::Positions::Constant(scores[i].rows(), scores[i].cols(),
															Model::Position::Zero());
		
		// Add the scores of each part
		for (int j = 0; j < nbParts; ++j) {
			const Deformation & d = parts_[j + 1].deformation;
			
			for (int y = 0; y < scores[i].rows(); ++y) {
				for (int x = 0; x < scores[i].cols(); ++x) {
					// Score of the best part position across scales
					HOGPyramid::Scalar best = -numeric_limits<HOGPyramid::Scalar>::infinity();
					
					// Coordinates of the center of the part at the root level
					const double cxr = x - padx + 0.5 * (parts_[j + 1].offset(0) + hpx);
					const double cyr = y - pady + 0.5 * (parts_[j + 1].offset(1) + hpy);
					
					for (int zp = max(i - interval - interval2, 0);
						 zp <= i - interval + interval2; ++zp) {
						const double xr = scales[i - zp] * cxr + padx - hpx;
						const double yr = scales[i - zp] * cyr + pady - hpy;
						const int ixr = xr + 0.5;
						const int iyr = yr + 0.5;
						
						if ((ixr >= 0) && (iyr >= 0) && (ixr < (*convolutions)[j + 1][zp].cols()) &&
							(iyr < (*convolutions)[j + 1][zp].rows())) {
							const int xp = (*positions)[j][zp](iyr, ixr)(0);
							const int yp = (*positions)[j][zp](iyr, ixr)(1);
							const double dx = xr - xp;
							const double dy = yr - yp;
							const double dz = i - interval - zp;
							const double cost = (d(0) * dx + d(1)) * dx +
											    (d(2) * dy + d(3)) * dy +
											    (d(4) * dz + d(5)) * dz;
							
							if ((*convolutions)[j + 1][zp](yp, xp) + cost > best) {
								best = (*convolutions)[j + 1][zp](yp, xp) + cost;
								(*positions)[j][i](y, x) << xp, yp, zp;
							}
						}
					}
					
					scores[i](y, x) += best;
				}
			}
		}
	}
	
	for (int i = 0; i < interval - interval2; ++i)
		for (int j = 0; j < nbParts; ++j)
			(*positions)[j][i] = Positions();
	
	// Add the bias if necessary
	if (bias_) {
#pragma omp parallel for
		for (int i = interval - interval2; i < nbLevels; ++i)
			scores[i].array() += bias_;
	}
#endif
}

double Model::dot(const Model & sample) const
{
	double d = bias_ * sample.bias_;
	
	if (parts_.size() != sample.parts_.size())
		return numeric_limits<double>::quiet_NaN();
	
	for (int i = 0; i < parts_.size(); ++i) {
		if ((parts_[i].filter.rows() != sample.parts_[i].filter.rows()) ||
			(parts_[i].filter.cols() != sample.parts_[i].filter.cols()))
			return numeric_limits<double>::quiet_NaN();
		
		for (int y = 0; y < parts_[i].filter.rows(); ++y)
			d += HOGPyramid::Map(parts_[i].filter).row(y).dot(
					HOGPyramid::Map(sample.parts_[i].filter).row(y));
		
		if (i)
			d += parts_[i].deformation.dot(sample.parts_[i].deformation);
	}
	
	return d;
}

double Model::norm() const
{
	double n = 0.0;
	
	for (int i = 0; i < parts_.size(); ++i) {
		n += HOGPyramid::Map(parts_[i].filter).squaredNorm();
		
		if (i)
			n += 10.0 * parts_[i].deformation.squaredNorm();
	}
	
	return sqrt(n);
}

Model & Model::operator+=(const Model & sample)
{
	if (parts_.size() != sample.parts_.size())
		return *this;
	
	Model copy(*this);
	
	for (int i = 0; i < parts_.size(); ++i) {
		if ((parts_[i].filter.rows() != sample.parts_[i].filter.rows()) ||
			(parts_[i].filter.cols() != sample.parts_[i].filter.cols())) {
			*this = copy; // Restore the copy
			return *this;
		}
		
		parts_[i].filter += sample.parts_[i].filter;
		parts_[i].deformation += sample.parts_[i].deformation;
	}
	
	bias_ += sample.bias_;
	
	return *this;
}

Model & Model::operator-=(const Model & sample)
{
	if (parts_.size() != sample.parts_.size())
		return *this;
	
	Model copy(*this);
	
	for (int i = 0; i < parts_.size(); ++i) {
		if ((parts_[i].filter.rows() != sample.parts_[i].filter.rows()) ||
			(parts_[i].filter.cols() != sample.parts_[i].filter.cols())) {
			*this = copy; // Restore the copy
			return *this;
		}
		
		parts_[i].filter -= sample.parts_[i].filter;
		parts_[i].deformation -= sample.parts_[i].deformation;
	}
	
	bias_ -= sample.bias_;
	
	return *this;
}

Model & Model::operator*=(double a)
{
	for (int i = 0; i < parts_.size(); ++i) {
		HOGPyramid::Map(parts_[i].filter) *= a;
		parts_[i].deformation *= a;
	}
	
	bias_ *= a;
	
	return *this;
}

Model Model::flip() const
{
	Model model;
	
	if (!empty()) {
		model.parts_.resize(parts_.size());
		
		// Flip the root
		model.parts_[0].filter = HOGPyramid::Flip(parts_[0].filter);
		model.parts_[0].offset = parts_[0].offset;
		model.parts_[0].deformation = parts_[0].deformation;
		
		// Flip the parts
		for (int i = 1; i < parts_.size(); ++i) {
			model.parts_[i].filter = HOGPyramid::Flip(parts_[i].filter);
			model.parts_[i].offset(0) = 2 * static_cast<int>(parts_[0].filter.cols()) -
										static_cast<int>(parts_[i].filter.cols()) -
										parts_[i].offset(0);
			model.parts_[i].offset(1) = parts_[i].offset(1);
			model.parts_[i].offset(2) = parts_[i].offset(2);
			model.parts_[i].deformation = parts_[i].deformation;
			model.parts_[i].deformation(1) = -model.parts_[i].deformation(1);
		}
	}
	
	model.bias_ = bias_;
	
	return model;
}

template <typename Scalar>
static void dt1d(const Scalar * x, int n, Scalar a, Scalar b, Scalar * z, int * v, Scalar * y,
				 int * m, const Scalar * t, int incx, int incy, int incm)
{
	assert(x && (y || m));
	assert(n > 0);
	assert(a < 0);
	assert(z && v);
	assert(t);
	assert(incx && incy && (m ? incm : true));
	
	z[0] =-numeric_limits<Scalar>::infinity();
	z[1] = numeric_limits<Scalar>::infinity();
	v[0] = 0;
	
	// Use a lookup table to replace the division by (a * (i - v[k]))
	int k = 0;
	Scalar xvk = x[0];
	
	for (int i = 1; i < n;) {
		const Scalar s = (x[i * incx] - xvk) * t[i - v[k]] + (i + v[k]) - b / a;
		
		if (s <= z[k]) {
			--k;
			xvk = x[v[k] * incx];
		}
		else {
			++k;
			v[k] = i;
			z[k] = s;
			xvk = x[i * incx];
			++i;
		}
	}
	
	z[k + 1] = numeric_limits<Scalar>::infinity();
	
	if (y && m) {
		for (int i = 0, k = 0; i < n; ++i) {
			while (z[k + 1] < 2 * i)
				++k;
			
			y[i * incy] = x[v[k] * incx] + (a * (i - v[k]) + b) * (i - v[k]);
			m[i * incm] = v[k];
		}
	}
	else if (y) {
		for (int i = 0, k = 0; i < n; ++i) {
			while (z[k + 1] < 2 * i)
				++k;
			
			y[i * incy] = x[v[k] * incx] + (a * (i - v[k]) + b) * (i - v[k]);
		}
	}
	else {
		for (int i = 0, k = 0; i < n; ++i) {
			while (z[k + 1] < 2 * i)
				++k;
			
			m[i * incm] = v[k];
		}
	}
}

void Model::DT2D(HOGPyramid::Matrix & matrix, const Part & part, HOGPyramid::Matrix & tmp,
				 Positions * positions)
{
	// Nothing to do if the matrix is empty
	if (!matrix.size())
		return;
	
	const int rows = static_cast<int>(matrix.rows());
	const int cols = static_cast<int>(matrix.cols());
	
	if (positions)
		positions->resize(rows, cols);
	
	tmp.resize(rows, cols);
	
	// Temporary vectors
	vector<HOGPyramid::Scalar> z(max(rows, cols) + 1);
	vector<int> v(max(rows, cols) + 1);
	vector<HOGPyramid::Scalar> t(max(rows, cols));
	
	t[0] = numeric_limits<HOGPyramid::Scalar>::infinity();
	
	for (int x = 1; x < cols; ++x)
		t[x] = 1 / (part.deformation(0) * x);
	
	// Filter the rows in tmp
	for (int y = 0; y < rows; ++y)
		dt1d<HOGPyramid::Scalar>(matrix.row(y).data(), cols, part.deformation(0),
								 part.deformation(1), &z[0], &v[0], tmp.row(y).data(),
								 positions ? positions->row(y).data()->data() : 0, &t[0], 1, 1, 3);
	
	for (int y = 1; y < rows; ++y)
		t[y] = 1 / (part.deformation(2) * y);
	
	// Filter the columns back to the original matrix
	for (int x = 0; x < cols; ++x)
		dt1d<HOGPyramid::Scalar>(tmp.data() + x, rows, part.deformation(2), part.deformation(3),
								 &z[0], &v[0],
#ifndef FFLD_MODEL_3D
								 matrix.data() + x,
#else
								 0,
#endif
								 positions ? ((positions->data() + x)->data() + 1) : 0, &t[0], cols,
								 cols, 3 * cols);
	
	// Re-index the best x positions now that the best y changed
	if (positions) {
		for (int y = 0; y < rows; ++y)
			for (int x = 0; x < cols; ++x)
				tmp(y, x) = (*positions)(y, x)(0);
		
		for (int y = 0; y < rows; ++y)
			for (int x = 0; x < cols; ++x)
				(*positions)(y, x)(0) = tmp((*positions)(y, x)(1), x);
	}
}

ostream & FFLD::operator<<(ostream & os, const Model & model)
{
	// Save the number of parts and the bias
	os << model.parts().size() << ' ' << model.bias() << endl;
	
	// Save the parts themselves
	for (int i = 0; i < model.parts().size(); ++i) {
		os << model.parts()[i].filter.rows() << ' ' << model.parts()[i].filter.cols() << ' '
		   << HOGPyramid::NbFeatures << ' ' << model.parts()[i].offset(0) << ' '
		   << model.parts()[i].offset(1) << ' ' << model.parts()[i].offset(2) << ' '
		   << model.parts()[i].deformation(0) << ' ' << model.parts()[i].deformation(1) << ' '
		   << model.parts()[i].deformation(2) << ' ' << model.parts()[i].deformation(3) << ' '
		   << model.parts()[i].deformation(4) << ' ' << model.parts()[i].deformation(5)
		   << endl;
		
		for (int y = 0; y < model.parts()[i].filter.rows(); ++y) {
			os << model.parts()[i].filter(y, 0)(0);
			
			for (int j = 1; j < HOGPyramid::NbFeatures; ++j)
				os << ' ' << model.parts()[i].filter(y, 0)(j);
			
			for (int x = 1; x < model.parts()[i].filter.cols(); ++x)
				for (int j = 0; j < HOGPyramid::NbFeatures; ++j)
					os << ' ' << model.parts()[i].filter(y, x)(j);
			
			os << endl;
		}
	}
	
	return os;
}

istream & FFLD::operator>>(istream & is, Model & model)
{
	int nbParts;
	double bias;
	
	is >> nbParts >> bias;
	
	if (!is) {
		model = Model();
		return is;
	}
	
	vector<Model::Part> parts(nbParts);
	
	for (int i = 0; i < nbParts; ++i) {
		int rows, cols, nbFeatures;
		
		is >> rows >> cols >> nbFeatures >> parts[i].offset(0) >> parts[i].offset(1)
		   >> parts[i].offset(2) >> parts[i].deformation(0) >> parts[i].deformation(1)
		   >> parts[i].deformation(2) >> parts[i].deformation(3) >> parts[i].deformation(4)
		   >> parts[i].deformation(5);
		
		if (!is || (nbFeatures > HOGPyramid::NbFeatures)) {
			model = Model();
			return is;
		}
		
		// Always set the offset and deformation of the root to zero
		if (!i) {
			parts[0].offset.setZero();
			parts[0].deformation.setZero();
		}
		// Always set the z offset of a part to zero
		else {
			parts[i].offset(2) = 0;
		}
		
		parts[i].filter = HOGPyramid::Level::Constant(rows, cols, HOGPyramid::Cell::Zero());
		
		for (int y = 0; y < rows; ++y) {
			for (int x = 0; x < cols; ++x) {
				for (int j = 0; j < nbFeatures; ++j)
					is >> parts[i].filter(y, x)(j);
				
				// Always put the truncation feature at the end
				if (nbFeatures < HOGPyramid::NbFeatures)
					swap(parts[i].filter(y, x)(nbFeatures - 1),
						 parts[i].filter(y, x)(HOGPyramid::NbFeatures - 1));
			}
		}
		
		if (!is) {
			model = Model();
			return is;
		}
	}
	
	model = Model(parts, bias);
	
	return is;
}

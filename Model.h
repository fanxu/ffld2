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

#ifndef FFLD_MODEL_H
#define FFLD_MODEL_H

#include "HOGPyramid.h"

namespace FFLD
{
/// The Model class can represent both a deformable part-based model or a training sample with
/// fixed latent variables (parts' positions). In both cases the members are the same: a list of
/// parts and a bias. If it is a sample, for each part the filter is set to the corresponding
/// features, the offset is set to the part's position relative to the root, and the deformation is
/// set to the deformation gradient (<tt>dx^2 dx dy^2 dy dz^2 dz</tt>), where (<tt>dx dy dz<tt>) are
/// the differences between the reference part location and the part position. The dot product
/// between the deformation gradient and the model deformation then computes the deformation cost.
/// @note Define the PASCAL_MODEL_3D to also deform parts across scales.
class Model
{
public:
	/// Type of a 3d position (x y z).
	typedef Eigen::Vector3i Position;
	
	/// Type of a matrix of 3d positions.
	typedef Eigen::Matrix<Position, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Positions;
	
	/// Type of a 3d quadratic deformation (dx^2 dx dy^2 dy dz^2 dz).
	typedef Eigen::Matrix<double, 6, 1> Deformation;
	
	/// The part structure stores all the information about a part (or the root).
	struct Part
	{
		HOGPyramid::Level filter;	///< Part filter.
		Position offset;			///< Part offset (dx dy dz) relative to the root.
		Deformation deformation;	///< Deformation cost (dx^2 dx dy^2 dy dz^2 dz).
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};
	
	/// Constructs an empty model. An empty model has an empty root and no part.
	Model();
	
	/// Constructs a model with the specified dimensions and initializes all the filters and the
	/// bias to zero.
	/// @param[in] rootSize Size of the root filter (<tt>rows x cols</tt>).
	/// @param[in] nbParts Number of parts (without the root).
	/// @param[in] partSize Size of all the parts (<tt>rows x cols</tt>).
	/// @note The model will be empty if any of the parameter is invalid.
	explicit Model(std::pair<int, int> rootSize, int nbParts = 0,
				   std::pair<int, int> partSize = std::make_pair(0, 0));
	
	/// Constructs a model from a list of parts and a bias.
	explicit Model(const std::vector<Part> & parts, double bias = 0.0);
	
	/// Returns whether the model is empty. An empty model has an empty root and no part.
	bool empty() const;
	
	/// Returns the model parts (the first one is the root).
	const std::vector<Part> & parts() const;
	
	/// Returns the model parts (the first one is the root).
	std::vector<Part> & parts();
	
	/// Returns the model bias.
	double bias() const;
	
	/// Returns the model bias.
	double & bias();
	
	/// Returns the size of the root (<tt>rows x cols</tt>).
	/// @note Equivalent to std::pair<int, int>(parts()[0].rows(), parts()[0].cols()).
	std::pair<int, int> rootSize() const;
	
	/// Returns the size of the parts (<tt>rows x cols</tt>).
	/// @note Equivalent to make_pair(parts()[1].rows(), parts()[1].cols()) if the model has parts.
	std::pair<int, int> partSize() const;
	
	/// Initializes the specidied number of parts from the root.
	/// @param[in] nbParts Number of parts (without the root).
	/// @param[in] partSize Size of each part (<tt>rows x cols</tt>).
	/// @note The model stay unmodified if any of the parameter is invalid.
	/// @note The parts are always initialized at twice the root resolution.
	void initializeParts(int nbParts, std::pair<int, int> partSize);
	
	/// Initializes a training sample with fixed latent variables from a specified position in
	/// a pyramid of features.
	/// @param[in] pyramid Pyramid of features.
	/// @param[in] x, y, z Coordinates of the root.
	/// @param[out] sample Initialized training sample.
	/// @param[in] positions Positions of each part for each pyramid level
	/// (<tt>parts x levels</tt>, only required if the model has parts).
	/// @note The sample will be empty if any of the parameter is invalid or if any of the part
	/// filter is unreachable.
	void initializeSample(const HOGPyramid & pyramid, int x, int y, int z, Model & sample,
						  const std::vector<std::vector<Positions> > * positions = 0) const;
	
	/// Returns the scores of the convolutions + distance transforms of the parts with a pyramid of
	/// features.
	/// @param[in] pyramid Pyramid of features.
	/// @param[out] scores Scores for each pyramid level.
	/// @param[out] positions Positions of each part and each pyramid level.
	/// @param[in] Precomputed convolutions of each part and each pyramid level.
	void convolve(const HOGPyramid & pyramid, std::vector<HOGPyramid::Matrix> & scores,
				  std::vector<std::vector<Positions> > * positions = 0,
				  std::vector<std::vector<HOGPyramid::Matrix> > * convolutions = 0) const;
	
	/// Returns the dot product between the model and a fixed training @p sample.
	/// @note Returns NaN if the sample and the model are not compatible.
	/// @note Do not compute dot products between two models or between two samples.
	double dot(const Model & sample) const;
	
	/// Returns the norm of a model or a fixed training sample.
	double norm() const;
	
	/// Adds the filters, deformation costs, and bias of the fixed sample with the ones of
	/// @p sample.
	/// @note Do nothing if the models are incompatible.
	/// @note Do not use with models, only with fixed samples.
	Model & operator+=(const Model & sample);
	
	/// Subtracts the filters, deformation costs, and bias of the fixed sample with the ones of
	/// @p sample.
	/// @note Do nothing if the models are incompatible.
	/// @note Do not use with models, only with fixed samples.
	Model & operator-=(const Model & sample);
	
	/// Multiplies the filters, deformation costs, and bias of the fixed sample by @p a.
	/// @note Do not use with models, only with fixed samples.
	Model & operator*=(double a);
	
	/// Returns the flipped version (horizontally) of a model or a fixed sample.
	Model flip() const;
	
	/// Computes an in-place 2D quadratic distance transform.
	/// @param[in,out] matrix Matrix to tranform in-place.
	/// @param[in] part Part from which to read the deformation cost and offset.
	/// @param tmp Temporary matrix.
	/// @param[out] positions Optimal position of each part for each root location.
	static void DT2D(HOGPyramid::Matrix & matrix, const Part & part, HOGPyramid::Matrix & tmp,
					 Positions * positions = 0);
	
private:
	std::vector<Part> parts_;
	double bias_;
};

/// Serializes a model to a stream.
std::ostream & operator<<(std::ostream & os, const Model & model);

/// Unserializes a model from a stream.
std::istream & operator>>(std::istream & is, Model & model);
}

#endif

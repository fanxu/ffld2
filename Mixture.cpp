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

#include "Intersector.h"
#include "LBFGS.h"
#include "Mixture.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

using namespace Eigen;
using namespace FFLD;
using namespace std;

Mixture::Mixture() : cached_(false), zero_(true)
{
}

Mixture::Mixture(const vector<Model> & models) : models_(models), cached_(false), zero_(true)
{
}

Mixture::Mixture(int nbComponents, const vector<Scene> & scenes, Object::Name name) :
cached_(false), zero_(true)
{
	// Create an empty mixture if any of the given parameters is invalid
	if ((nbComponents <= 0) || scenes.empty()) {
		cerr << "Attempting to create an empty mixture" << endl;
		return;
	}
	
	// Compute the root filters' sizes using Felzenszwalb's heuristic
	const vector<pair<int, int> > sizes = FilterSizes(nbComponents, scenes, name);
	
	// Early return in case the root filters' sizes could not be determined
	if (sizes.size() != nbComponents)
		return;
	
	// Initialize the models (with symmetry) to those sizes
	models_.resize(2 * nbComponents);
	
	for (int i = 0; i < nbComponents; ++i) {
		models_[2 * i    ] = Model(sizes[i]);
		models_[2 * i + 1] = Model(sizes[i]);
	}
}

bool Mixture::empty() const
{
	return models_.empty();
}

const vector<Model> & Mixture::models() const
{
	return models_;
}

vector<Model> & Mixture::models()
{
	return models_;
}

pair<int, int> Mixture::minSize() const
{
	pair<int, int> size(0, 0);
	
	if (!models_.empty()) {
		size = models_[0].rootSize();
		
		for (int i = 1; i < models_.size(); ++i) {
			size.first = min(size.first, models_[i].rootSize().first);
			size.second = min(size.second, models_[i].rootSize().second);
		}
	}
	
	return size;
}

pair<int, int> Mixture::maxSize() const
{
	pair<int, int> size(0, 0);
	
	if (!models_.empty()) {
		size = models_[0].rootSize();
		
		for (int i = 1; i < models_.size(); ++i) {
			size.first = max(size.first, models_[i].rootSize().first);
			size.second = max(size.second, models_[i].rootSize().second);
		}
	}
	
	return size;
}

double Mixture::train(const vector<Scene> & scenes, Object::Name name, int padx, int pady,
					  int interval, int nbRelabel, int nbDatamine, int maxNegatives, double C,
					  double J, double overlap)
{
	if (empty() || scenes.empty() || (padx < 1) || (pady < 1) || (interval < 1) ||
		(nbRelabel < 1) || (nbDatamine < 1) || (maxNegatives < models_.size()) || (C <= 0.0) ||
		(J <= 0.0) || (overlap <= 0.0) || (overlap >= 1.0)) {
		cerr << "Invalid training parameters" << endl;
		return numeric_limits<double>::quiet_NaN();
	}
	
	// Test if the models are really zero by looking at the first cell of the first filter of the
	// first model
	if (!models_[0].empty() && models_[0].parts()[0].filter.size() &&
		!models_[0].parts()[0].filter(0, 0).isZero())
		zero_ = false;
	
	double loss = numeric_limits<double>::infinity();
	
	for (int relabel = 0; relabel < nbRelabel; ++relabel) {
		// Sample all the positives
		vector<pair<Model, int> > positives;
		
		posLatentSearch(scenes, name, padx, pady, interval, overlap, positives);
		
		// Left-right clustering at the first iteration
		if (zero_)
			Cluster(static_cast<int>(models_.size()), positives);
		
		// Cache of hard negative samples of maximum size maxNegatives
		vector<pair<Model, int> > negatives;
		
		// Previous loss on the cache
		double prevLoss = -numeric_limits<double>::infinity();
		
		for (int datamine = 0; datamine < nbDatamine; ++datamine) {
			// Remove easy samples (keep hard ones)
			int j = 0;
			
			for (int i = 0; i < negatives.size(); ++i)
				if ((negatives[i].first.parts()[0].deformation(3) =
					 models_[negatives[i].second].dot(negatives[i].first)) > -1.01)
					negatives[j++] = negatives[i];
			
			negatives.resize(j);
			
			// Sample new hard negatives
			negLatentSearch(scenes, name, padx, pady, interval, maxNegatives, negatives);
			
			// Stop if there are no new hard negatives
			if (datamine && (negatives.size() == j))
				break;
			
			// Merge the left / right samples for more efficient training
			vector<int> posComponents(positives.size());
			
			for (int i = 0; i < positives.size(); ++i) {
				posComponents[i] = positives[i].second;
				
				if (positives[i].second & 1)
					positives[i].first = positives[i].first.flip();
				
				positives[i].second >>= 1;
			}
			
			vector<int> negComponents(negatives.size());
			
			for (int i = 0; i < negatives.size(); ++i) {
				negComponents[i] = negatives[i].second;
				
				if (negatives[i].second & 1)
					negatives[i].first = negatives[i].first.flip();
				
				negatives[i].second >>= 1;
			}
			
			// Merge the left / right models for more efficient training
			for (int i = 1; i < models_.size() / 2; ++i)
				models_[i] = models_[i * 2];
			
			models_.resize(models_.size() / 2);
			
			const int maxIterations =
				min(max(10.0 * sqrt(static_cast<double>(positives.size())), 100.0), 1000.0);
			
			loss = train(positives, negatives, C, J, maxIterations);
			
			cout << "Relabel: " << relabel << ", datamine: " << datamine
				 << ", # positives: " << positives.size() << ", # hard negatives: " << j
				 << " (already in the cache) + " << (negatives.size() - j) << " (new) = "
				 << negatives.size() << ", loss (cache): " << loss << endl;
			
			// Unmerge the left / right samples
			for (int i = 0; i < positives.size(); ++i) {
				positives[i].second = posComponents[i];
				
				if (positives[i].second & 1)
					positives[i].first = positives[i].first.flip();
			}
			
			for (int i = 0; i < negatives.size(); ++i) {
				negatives[i].second = negComponents[i];
				
				if (negatives[i].second & 1)
					negatives[i].first = negatives[i].first.flip();
			}
			
			// Unmerge the left / right models
			models_.resize(models_.size() * 2);
			
			for (int i = static_cast<int>(models_.size()) / 2 - 1; i >= 0; --i) {
				models_[i * 2    ] = models_[i];
				models_[i * 2 + 1] = models_[i].flip();
			}
			
			// The filters definitely changed
			filterCache_.clear();
			cached_ = false;
			zero_ = false;
			
			// Save the latest model so as to be able to look at it while training
			ofstream out("tmp.txt");
			
			out << (*this);
			
			// Stop if we are not making progress
			if ((0.999 * loss < prevLoss) && (negatives.size() < maxNegatives))
				break;
			
			prevLoss = loss;
		}
	}
	
	return loss;
}

void Mixture::initializeParts(int nbParts, pair<int, int> partSize)
{
	for (int i = 0; i < models_.size(); i += 2) {
		models_[i].initializeParts(nbParts, partSize);
		models_[i + 1] = models_[i].flip();
	}
	
	// The filters definitely changed
	filterCache_.clear();
	cached_ = false;
	zero_ = false;
}

void Mixture::convolve(const HOGPyramid & pyramid, vector<HOGPyramid::Matrix> & scores,
					   vector<Indices> & argmaxes,
					   vector<vector<vector<Model::Positions> > > * positions) const
{
	if (empty() || pyramid.empty()) {
		scores.clear();
		argmaxes.clear();
		
		if (positions)
			positions->clear();
		
		return;
	}
	
	const int nbModels = static_cast<int>(models_.size());
	const int nbLevels = static_cast<int>(pyramid.levels().size());
	
	// Convolve with all the models
	vector<vector<HOGPyramid::Matrix> > convolutions;
	
	convolve(pyramid, convolutions, positions);
	
	// In case of error
	if (convolutions.empty()) {
		scores.clear();
		argmaxes.clear();
		
		if (positions)
			positions->clear();
		
		return;
	}
	
	// Resize the scores and argmaxes
	scores.resize(nbLevels);
	argmaxes.resize(nbLevels);
	
#pragma omp parallel for
	for (int z = 0; z < nbLevels; ++z) {
		int rows = static_cast<int>(convolutions[0][z].rows());
		int cols = static_cast<int>(convolutions[0][z].cols());
		
		for (int i = 1; i < nbModels; ++i) {
			rows = min(rows, static_cast<int>(convolutions[i][z].rows()));
			cols = min(cols, static_cast<int>(convolutions[i][z].cols()));
		}
		
		scores[z].resize(rows, cols);
		argmaxes[z].resize(rows, cols);
		
		for (int y = 0; y < rows; ++y) {
			for (int x = 0; x < cols; ++x) {
				int argmax = 0;
				
				for (int i = 1; i < nbModels; ++i)
					if (convolutions[i][z](y, x) > convolutions[argmax][z](y, x))
						argmax = i;
				
				scores[z](y, x) = convolutions[argmax][z](y, x);
				argmaxes[z](y, x) = argmax;
			}
		}
	}
}

void Mixture::cacheFilters() const
{
	// Count the number of filters
	int nbFilters = 0;
	
	for (int i = 0; i < models_.size(); ++i)
		nbFilters += models_[i].parts().size();
	
	// Transform all the filters
	filterCache_.resize(nbFilters);
	
	for (int i = 0, j = 0; i < models_.size(); ++i) {
#pragma omp parallel for
		for (int k = 0; k < models_[i].parts().size(); ++k)
			Patchwork::TransformFilter(models_[i].parts()[k].filter, filterCache_[j + k]);
		
		j += models_[i].parts().size();
	}
	
	cached_ = true;
}

static inline void clipBndBox(Rectangle & bndbox, const Scene & scene, double alpha = 0.0)
{
	// Compromise between clamping the bounding box to the image and penalizing bounding boxes
	// extending outside the image
	if (bndbox.left() < 0)
		bndbox.setLeft(bndbox.left() * alpha - 0.5);
	
	if (bndbox.top() < 0)
		bndbox.setTop(bndbox.top() * alpha - 0.5);
	
	if (bndbox.right() >= scene.width())
		bndbox.setRight(scene.width() - 1 + (bndbox.right() - scene.width() + 1) * alpha + 0.5);
	
	if (bndbox.bottom() >= scene.height())
		bndbox.setBottom(scene.height() - 1 + (bndbox.bottom() - scene.height() + 1) * alpha + 0.5);
}

void Mixture::posLatentSearch(const vector<Scene> & scenes, Object::Name name, int padx, int pady,
							  int interval, double overlap,
							  vector<pair<Model, int> > & positives) const
{
	if (scenes.empty() || (padx < 1) || (pady < 1) || (interval < 1) || (overlap <= 0.0) ||
		(overlap >= 1.0)) {
		positives.clear();
		cerr << "Invalid training paramters" << endl;
		return;
	}
	
	positives.clear();
	
	for (int i = 0; i < scenes.size(); ++i) {
		// Skip negative scenes
		bool negative = true;
		
		for (int j = 0; j < scenes[i].objects().size(); ++j)
			if ((scenes[i].objects()[j].name() == name) && !scenes[i].objects()[j].difficult())
				negative = false;
		
		if (negative)
			continue;
		
		const JPEGImage image(scenes[i].filename());
		
		if (image.empty()) {
			positives.clear();
			return;
		}
		
		const HOGPyramid pyramid(image, padx, pady, interval);
		
		if (pyramid.empty()) {
			positives.clear();
			return;
		}
		
		vector<HOGPyramid::Matrix> scores;
		vector<Indices> argmaxes;
		vector<vector<vector<Model::Positions> > > positions;
		
		if (!zero_)
			convolve(pyramid, scores, argmaxes, &positions);
		
		// For each object, set as positive the best (highest score or else most intersecting)
		// position
		for (int j = 0; j < scenes[i].objects().size(); ++j) {
			// Ignore objects with a different name or difficult objects
			if ((scenes[i].objects()[j].name() != name) || scenes[i].objects()[j].difficult())
				continue;
			
			const Intersector intersector(scenes[i].objects()[j].bndbox(), overlap);
			
			// The model, level, position, score, and intersection of the best example
			int argModel = -1;
			int argX = -1;
			int argY = -1;
			int argZ = -1;
			double maxScore = -numeric_limits<double>::infinity();
			double maxInter = 0.0;
			
			for (int z = 0; z < pyramid.levels().size(); ++z) {
				const double scale = pow(2.0, static_cast<double>(z) / interval + 2);
				int rows = 0;
				int cols = 0;
				
				if (!zero_) {
					rows = static_cast<int>(scores[z].rows());
					cols = static_cast<int>(scores[z].cols());
				}
				else if (z >= interval) {
					rows = static_cast<int>(pyramid.levels()[z].rows()) - maxSize().first + 1;
					cols = static_cast<int>(pyramid.levels()[z].cols()) - maxSize().second + 1;
				}
				
				for (int y = 0; y < rows; ++y) {
					for (int x = 0; x < cols; ++x) {
						// Find the best matching model (highest score or else most intersecting)
						int model = zero_ ? 0 : argmaxes[z](y, x);
						double intersection = 0.0;
						
						// Try all models and keep the most intersecting one
						if (zero_) {
							for (int k = 0; k < models_.size(); ++k) {
								// The bounding box of the model at this position
								Rectangle bndbox;
								bndbox.setX((x - padx) * scale + 0.5);
								bndbox.setY((y - pady) * scale + 0.5);
								bndbox.setWidth(models_[k].rootSize().second * scale + 0.5);
								bndbox.setHeight(models_[k].rootSize().first * scale + 0.5);
								
								// Trade-off between clipping and penalizing
								clipBndBox(bndbox, scenes[i], 0.5);
								
								double inter = 0.0;
								
								if (intersector(bndbox, &inter)) {
									if (inter > intersection) {
										model = k;
										intersection = inter;
									}
								}
							}
						}
						// Just take the model with the best score
						else {
							// The bounding box of the model at this position
							Rectangle bndbox;
							bndbox.setX((x - padx) * scale + 0.5);
							bndbox.setY((y - pady) * scale + 0.5);
							bndbox.setWidth(models_[model].rootSize().second * scale + 0.5);
							bndbox.setHeight(models_[model].rootSize().first * scale + 0.5);
							
							clipBndBox(bndbox, scenes[i]);
							intersector(bndbox, &intersection);
						}
						
						if ((intersection > maxInter) && (zero_ || (scores[z](y, x) > maxScore))) {
							argModel = model;
							argX = x;
							argY = y;
							argZ = z;
							
							if (!zero_)
								maxScore = scores[z](y, x);
							
							maxInter = intersection;
						}
					}
				}
			}
			
			if (maxInter >= overlap) {
				Model sample;
				
				models_[argModel].initializeSample(pyramid, argX, argY, argZ, sample,
												   zero_ ? 0 : &positions[argModel]);
				
				if (!sample.empty())
					positives.push_back(make_pair(sample, argModel));
			}
		}
	}
}

static inline bool operator==(const Model & a, const Model & b)
{
	return (a.parts()[0].offset == b.parts()[0].offset) &&
		   (a.parts()[0].deformation(0) == b.parts()[0].deformation(0)) &&
		   (a.parts()[0].deformation(1) == b.parts()[0].deformation(1));
}

static inline bool operator<(const Model & a, const Model & b)
{
	return (a.parts()[0].offset(0) < b.parts()[0].offset(0)) ||
		   ((a.parts()[0].offset(0) == b.parts()[0].offset(0)) &&
			((a.parts()[0].offset(1) < b.parts()[0].offset(1)) ||
			 ((a.parts()[0].offset(1) == b.parts()[0].offset(1)) &&
			  ((a.parts()[0].deformation(0) < b.parts()[0].deformation(0)) ||
			   ((a.parts()[0].deformation(0) == b.parts()[0].deformation(0)) &&
			    ((a.parts()[0].deformation(1) < b.parts()[0].deformation(1))))))));
}

void Mixture::negLatentSearch(const vector<Scene> & scenes, Object::Name name, int padx, int pady,
							  int interval, int maxNegatives,
							  vector<pair<Model, int> > & negatives) const
{
	// Sample at most (maxNegatives - negatives.size()) negatives with a score above -1.0
	if (scenes.empty() || (padx < 1) || (pady < 1) || (interval < 1) || (maxNegatives <= 0) ||
		(negatives.size() >= maxNegatives)) {
		negatives.clear();
		cerr << "Invalid training paramters" << endl;
		return;
	}
	
	// The number of negatives already in the cache
	const int nbCached = static_cast<int>(negatives.size());
	
	for (int i = 0, j = 0; i < scenes.size(); ++i) {
		// Skip positive scenes
		bool positive = false;
		
		for (int k = 0; k < scenes[i].objects().size(); ++k)
			if (scenes[i].objects()[k].name() == name)
				positive = true;
		
		if (positive)
			continue;
		
		const JPEGImage image(scenes[i].filename());
		
		if (image.empty()) {
			negatives.clear();
			return;
		}
		
		const HOGPyramid pyramid(image, padx, pady, interval);
		
		if (pyramid.empty()) {
			negatives.clear();
			return;
		}
		
		vector<HOGPyramid::Matrix> scores;
		vector<Indices> argmaxes;
		vector<vector<vector<Model::Positions> > > positions;
		
		if (!zero_)
			convolve(pyramid, scores, argmaxes, &positions);
		
		for (int z = 0; z < pyramid.levels().size(); ++z) {
			int rows = 0;
			int cols = 0;
			
			if (!zero_) {
				rows = static_cast<int>(scores[z].rows());
				cols = static_cast<int>(scores[z].cols());
			}
			else if (z >= interval) {
				rows = static_cast<int>(pyramid.levels()[z].rows()) - maxSize().first + 1;
				cols = static_cast<int>(pyramid.levels()[z].cols()) - maxSize().second + 1;
			}
			
			for (int y = 0; y < rows; ++y) {
				for (int x = 0; x < cols; ++x) {
					const int argmax = zero_ ? (rand() % models_.size()) : argmaxes[z](y, x);
					
					if (zero_ || (scores[z](y, x) > -1)) {
						Model sample;
						
						models_[argmax].initializeSample(pyramid, x, y, z, sample,
														 zero_ ? 0 : &positions[argmax]);
						
						if (!sample.empty()) {
							// Store all the information about the sample in the offset and
							// deformation of its root
							sample.parts()[0].offset(0) = i;
							sample.parts()[0].offset(1) = z;
							sample.parts()[0].deformation(0) = y;
							sample.parts()[0].deformation(1) = x;
							sample.parts()[0].deformation(2) = argmax;
							sample.parts()[0].deformation(3) = zero_ ? 0.0 : scores[z](y, x);
							
							// Look if the same sample was already sampled
							while ((j < nbCached) && (negatives[j].first < sample))
								++j;
							
							// Make sure not to put the same sample twice
							if ((j >= nbCached) || !(negatives[j].first == sample)) {
								negatives.push_back(make_pair(sample, argmax));
								
								if (negatives.size() == maxNegatives)
									return;
							}
						}
					}
				}
			}
		}
	}
}

namespace FFLD
{
namespace detail
{
class Loss : public LBFGS::IFunction
{
public:
	Loss(vector<Model> & models, const vector<pair<Model, int> > & positives,
		 const vector<pair<Model, int> > & negatives, double C, double J, int maxIterations) :
	models_(models), positives_(positives), negatives_(negatives), C_(C), J_(J),
	maxIterations_(maxIterations)
	{
	}
	
	virtual int dim() const
	{
		int d = 0;
		
		for (int i = 0; i < models_.size(); ++i) {
			for (int j = 0; j < models_[i].parts().size(); ++j) {
				d += models_[i].parts()[j].filter.size() * HOGPyramid::NbFeatures; // Filter
				
				if (j)
					d += 6; // Deformation
			}
			
			++d; // Bias
		}
		
		return d;
	}
	
	virtual double operator()(const double * x, double * g = 0) const
	{
		// Recopy the features into the models
		ToModels(x, models_);
		
		// Compute the loss and gradient over the samples
		double loss = 0.0;
		
		vector<Model> gradients;
		
		if (g) {
			gradients.resize(models_.size());
			
			for (int i = 0; i < models_.size(); ++i)
				gradients[i] = Model(models_[i].rootSize(),
									 static_cast<int>(models_[i].parts().size()) - 1,
									 models_[i].partSize());
		}
		
		vector<double> posMargins(positives_.size());
		
#pragma omp parallel for
		for (int i = 0; i < positives_.size(); ++i)
			posMargins[i] = models_[positives_[i].second].dot(positives_[i].first);
		
		for (int i = 0; i < positives_.size(); ++i) {
			if (posMargins[i] < 1.0) {
				loss += 1.0 - posMargins[i];
				
				if (g)
					gradients[positives_[i].second] -= positives_[i].first;
			}
		}
		
		// Reweight the positives
		if (J_ != 1.0) {
			loss *= J_;
			
			if (g) {
				for (int i = 0; i < models_.size(); ++i)
					gradients[i] *= J_;
			}
		}
		
		vector<double> negMargins(negatives_.size());
		
#pragma omp parallel for
		for (int i = 0; i < negatives_.size(); ++i)
			negMargins[i] = models_[negatives_[i].second].dot(negatives_[i].first);
		
		for (int i = 0; i < negatives_.size(); ++i) {
			if (negMargins[i] > -1.0) {
				loss += 1.0 + negMargins[i];
				
				if (g)
					gradients[negatives_[i].second] += negatives_[i].first;
			}
		}
		
		// Add the loss and gradient of the regularization term
		double maxNorm = 0.0;
		int argNorm = 0;
		
		for (int i = 0; i < models_.size(); ++i) {
			if (g)
				gradients[i] *= C_;
			
			const double norm = models_[i].norm();
			
			if (norm > maxNorm) {
				maxNorm = norm;
				argNorm = i;
			}
		}
		
		// Recopy the gradient if needed
		if (g) {
			// Regularization gradient
			gradients[argNorm] += models_[argNorm];
			
			// Regularize the deformation 10 times more
			for (int i = 1; i < gradients[argNorm].parts().size(); ++i)
				gradients[argNorm].parts()[i].deformation +=
					9.0 * models_[argNorm].parts()[i].deformation;
			
			// Do not regularize the bias
			gradients[argNorm].bias() -= models_[argNorm].bias();
			
			// In case minimum constraints were applied
			for (int i = 0; i < models_.size(); ++i) {
				for (int j = 1; j < models_[i].parts().size(); ++j) {
					if (models_[i].parts()[j].deformation(0) >= -0.005)
						gradients[i].parts()[j].deformation(0) =
							max(gradients[i].parts()[j].deformation(0), 0.0);
					
					if (models_[i].parts()[j].deformation(2) >= -0.005)
						gradients[i].parts()[j].deformation(2) =
							max(gradients[i].parts()[j].deformation(2), 0.0);
					
					if (models_[i].parts()[j].deformation(4) >= -0.005)
						gradients[i].parts()[j].deformation(4) =
							max(gradients[i].parts()[j].deformation(4), 0.0);
				}
			}
			
			FromModels(gradients, g);
		}
		
		return 0.5 * maxNorm * maxNorm + C_ * loss;
	}
	
	static void ToModels(const double * x, vector<Model> & models)
	{
		for (int i = 0, j = 0; i < models.size(); ++i) {
			for (int k = 0; k < models[i].parts().size(); ++k) {
				const int nbFeatures = static_cast<int>(models[i].parts()[k].filter.size()) *
									   HOGPyramid::NbFeatures;
				
				copy(x + j, x + j + nbFeatures, models[i].parts()[k].filter.data()->data());
				
				j += nbFeatures;
				
				if (k) {
					// Apply minimum constraints
					models[i].parts()[k].deformation(0) = min((x + j)[0],-0.005);
					models[i].parts()[k].deformation(1) = (x + j)[1];
					models[i].parts()[k].deformation(2) = min((x + j)[2],-0.005);
					models[i].parts()[k].deformation(3) = (x + j)[3];
					models[i].parts()[k].deformation(4) = min((x + j)[4],-0.005);
					models[i].parts()[k].deformation(5) = (x + j)[5];
					
					j += 6;
				}
			}
			
			models[i].bias() = x[j];
			
			++j;
		}
	}
	
	static void FromModels(const vector<Model> & models, double * x)
	{
		for (int i = 0, j = 0; i < models.size(); ++i) {
			for (int k = 0; k < models[i].parts().size(); ++k) {
				const int nbFeatures = static_cast<int>(models[i].parts()[k].filter.size()) *
									   HOGPyramid::NbFeatures;
				
				copy(models[i].parts()[k].filter.data()->data(),
					 models[i].parts()[k].filter.data()->data() + nbFeatures, x + j);
				
				j += nbFeatures;
				
				if (k) {
					copy(models[i].parts()[k].deformation.data(),
						 models[i].parts()[k].deformation.data() + 6, x + j);
					
					j += 6;
				}
			}
			
			x[j] = models[i].bias();
			
			++j;
		}
	}
	
private:
	vector<Model> & models_;
	const vector<pair<Model, int> > & positives_;
	const vector<pair<Model, int> > & negatives_;
	double C_;
	double J_;
	int maxIterations_;
};}
}

double Mixture::train(const vector<pair<Model, int> > & positives,
					  const vector<pair<Model, int> > & negatives, double C, double J,
					  int maxIterations)
{
	detail::Loss loss(models_, positives, negatives, C, J, maxIterations);
	LBFGS lbfgs(&loss, 0.001, maxIterations, 20, 20);
	
	// Start from the current models
	VectorXd x(loss.dim());
	
	detail::Loss::FromModels(models_, x.data());
	
	const double l = lbfgs(x.data());
	
	detail::Loss::ToModels(x.data(), models_);
	
	return l;
}

void Mixture::convolve(const HOGPyramid & pyramid,
					   vector<vector<HOGPyramid::Matrix> > & scores,
					   vector<vector<vector<Model::Positions> > > * positions) const
{
	if (empty() || pyramid.empty()) {
		scores.clear();
		
		if (positions)
			positions->clear();
		
		return;
	}
	
	const int nbModels = static_cast<int>(models_.size());
	
	// Resize the scores and positions
	scores.resize(nbModels);
	
	if (positions)
		positions->resize(nbModels);
	
	// Transform the filters if needed
#ifndef FFLD_MIXTURE_STANDARD_CONVOLUTION
#pragma omp critical
	if (!cached_)
		cacheFilters();
	
	while (!cached_);
	
	// Create a patchwork
	const Patchwork patchwork(pyramid);
	
	// Convolve the patchwork with the filters
	vector<vector<HOGPyramid::Matrix> > convolutions(filterCache_.size());
	
	patchwork.convolve(filterCache_, convolutions);
	
	// In case of error
	if (convolutions.empty()) {
		scores.clear();
		
		if (positions)
			positions->clear();
		
		return;
	}
	
	// Save the offsets of each model in the filter list
	vector<int> offsets(nbModels);
	
	for (int i = 0, j = 0; i < nbModels; ++i) {
		offsets[i] = j;
		j += models_[i].parts().size();
	}
	
	// For each model
#pragma omp parallel for
	for (int i = 0; i < nbModels; ++i) {
		vector<vector<HOGPyramid::Matrix> > tmp(models_[i].parts().size());
		
		for (int j = 0; j < tmp.size(); ++j)
			tmp[j].swap(convolutions[offsets[i] + j]);
		
		models_[i].convolve(pyramid, scores[i], positions ? &(*positions)[i] : 0, &tmp);
	}
	
	// In case of error
	for (int i = 0; i < nbModels; ++i) {
		if (scores[i].empty()) {
			scores.clear();
			
			if (positions)
				positions->clear();
		}
	}
#else
#pragma omp parallel for
	for (int i = 0; i < nbModels; ++i)
		models_[i].convolve(pyramid, scores[i], positions ? &(*positions)[i] : 0);
#endif
}

vector<pair<int, int> > Mixture::FilterSizes(int nbComponents, const vector<Scene> & scenes,
											 Object::Name name)
{
	// Early return in case the filters or the dataset are empty
	if ((nbComponents <= 0) || scenes.empty())
		return vector<pair<int, int> >();
	
	// Sort the aspect ratio of all the (non difficult) samples
	vector<double> ratios;
	
	for (int i = 0; i < scenes.size(); ++i) {
		for (int j = 0; j < scenes[i].objects().size(); ++j) {
			const Object & obj = scenes[i].objects()[j];
			
			if ((obj.name() == name) && !obj.difficult())
				ratios.push_back(static_cast<double>(obj.bndbox().width()) / obj.bndbox().height());
		}
	}
	
	// Early return if there is no object
	if (ratios.empty())
		return vector<pair<int, int> >();
	
	// Sort the aspect ratio of all the samples
	sort(ratios.begin(), ratios.end());
	
	// For each mixture model
	vector<double> references(nbComponents);
	
	for (int i = 0; i < nbComponents; ++i)
		references[i] = ratios[(i * ratios.size()) / nbComponents];
	
	// Store the areas of the objects associated to each component
	vector<vector<int> > areas(nbComponents);
	
	for (int i = 0; i < scenes.size(); ++i) {
		for (int j = 0; j < scenes[i].objects().size(); ++j) {
			const Object & obj = scenes[i].objects()[j];
			
			if ((obj.name() == name) && !obj.difficult()) {
				const double r = static_cast<double>(obj.bndbox().width()) / obj.bndbox().height();
				
				int k = 0;
				
				while ((k + 1 < nbComponents) && (r >= references[k + 1]))
					++k;
				
				areas[k].push_back(obj.bndbox().width() * obj.bndbox().height());
			}
		}
	}
	
	// For each component in reverse order
	vector<pair<int, int> > sizes(nbComponents);
	
	for (int i = nbComponents - 1; i >= 0; --i) {
		if (!areas[i].empty()) {
			sort(areas[i].begin(), areas[i].end());
			
			const int area = min(max(areas[i][(areas[i].size() * 2) / 10], 3000), 5000);
			const double ratio = ratios[(ratios.size() * (i * 2 + 1)) / (nbComponents * 2)];
			
			sizes[i].first = sqrt(area / ratio) / 8.0 + 0.5;
			sizes[i].second = sqrt(area * ratio) / 8.0 + 0.5;
		}
		else {
			sizes[i] = sizes[i + 1];
		}
	}
	
	return sizes;
}

void Mixture::Cluster(int nbComponents, vector<pair<Model, int> > & samples)
{
	// Early return in case the filters or the dataset are empty
	if ((nbComponents <= 0) || samples.empty())
		return;
	
	// For each model
	for (int i = nbComponents - 1; i >= 0; --i) {
		// Indices of the positives
		vector<int> permutation;
		
		// For each positive
		for (int j = 0; j < samples.size(); ++j)
			if (samples[j].second / 2 == i)
				permutation.push_back(j);
		
		// Next model if this one has no associated positives
		if (permutation.empty())
			continue;
		
		// Score of the best split so far
		double best = 0.0;
		
		// Do 1000 clustering trials
		for (int j = 0; j < 1000; ++j) {
			random_shuffle(permutation.begin(), permutation.end());
			
			vector<bool> assignment(permutation.size(), false);
			Model left = samples[permutation[0]].first;
			
			for (int k = 1; k < permutation.size(); ++k) {
				const Model & positive = samples[permutation[k]].first;
				
				if (positive.dot(left) > positive.dot(left.flip())) {
					left += positive;
				}
				else {
					left += positive.flip();
					assignment[k] = true;
				}
			}
			
			left *= 1.0 / permutation.size();
			
			const Model right = left.flip();
			double dots = 0.0;
			
			for (int k = 0; k < permutation.size(); ++k)
				dots += samples[permutation[k]].first.dot(assignment[k] ? right : left);
			
			if (dots > best) {
				for (int k = 0; k < permutation.size(); ++k)
					samples[permutation[k]].second = 2 * i + assignment[k];
				
				best = dots;
			}
		}
	}
}

ostream & FFLD::operator<<(ostream & os, const Mixture & mixture)
{
	// Save the number of models (mixture components)
	os << mixture.models().size() << endl;
	
	// Save the models themselves
	for (int i = 0; i < mixture.models().size(); ++i)
		os << mixture.models()[i] << endl;
	
	return os;
}

istream & FFLD::operator>>(istream & is, Mixture & mixture)
{
	int nbModels;
	
	is >> nbModels;
	
	if (!is || (nbModels <= 0)) {
		mixture = Mixture();
		return is;
	}
	
	vector<Model> models(nbModels);
	
	for (int i = 0; i < nbModels; ++i) {
		is >> models[i];
		
		if (!is || models[i].empty()) {
			mixture = Mixture();
			return is;
		}
	}
	
	mixture.models().swap(models);
	
	return is;
}

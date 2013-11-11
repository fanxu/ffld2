//--------------------------------------------------------------------------------------------------
// Limited-memory BFGS (L-BFGS) algorithm implementation as described by Nocedal.
// L-BFGS is an unconstrained quasi-Newton optimization method that uses a limited memory variation
// of the Broyden–Fletcher–Goldfarb–Shanno (BFGS) update to approximate the inverse Hessian matrix.
// The implementation is robust as it uses a simple line-search technique (backtracking in one
// direction only) and still works even if the L-BFGS algorithm returns a non descent direction (as 
// it will then restart the optimization starting from the current solution).
// Its robustness enables it to minimize non-smooth functions, such as the hinge loss.
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

#include "LBFGS.h"

#include <Eigen/Core>

#include <algorithm>
#include <cassert>

LBFGS::IFunction::~IFunction()
{
}

bool LBFGS::IFunction::progress(const double * x, const double * g, int n, double fx, double xnorm,
								double gnorm, double step, int t, int ls) const
{
	return false;
}

LBFGS::LBFGS(const IFunction * function, double epsilon, int maxIterations, int maxLineSearches,

			 int maxHistory) : function_(function), epsilon_(epsilon),
	maxIterations_(maxIterations), maxLineSearches_(maxLineSearches), maxHistory_(maxHistory)
{
	assert(!function || (function->dim() > 0));
	assert(epsilon > 0.0);
	assert(maxIterations > 0);
	assert(maxLineSearches > 0);
	assert(maxHistory >= 0);
}

double LBFGS::operator()(double * argx) const
{
	// Define the types ourselves to make sure that the matrices are col-major
	typedef Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor> VectorXd;
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixXd;
	
	assert(function_);
	assert(argx);
	
	// Convert the current solution to an Eigen::Map
	Eigen::Map<VectorXd> x(argx, function_->dim());
	
	// Initial value of the objective function and gradient
	VectorXd g(x.rows());
	double fx = (*function_)(x.data(), g.data());
	function_->progress(argx, g.data(), static_cast<int>(x.rows()), fx, x.norm(), g.norm(), 0.0, 0,
						1);
	
	// Histories of the previous solution (required by L-BFGS)
	VectorXd px; // Previous solution x_{t-1}
	VectorXd pg; // Previous gradient g_{t-1}
	MatrixXd dxs(x.rows(), maxHistory_); // History of the previous dx's = x_{t-1} - x_{t-2}, ...
	MatrixXd dgs(x.rows(), maxHistory_); // History of the previous dg's = g_{t-1} - g_{t-2}, ...
	
	// Number of iterations remaining
	for (int i = 0, j = 0; j < maxIterations_; ++i, ++j) {
		// Relative tolerance
		const double relativeEpsilon = epsilon_ * std::max(1.0, x.norm());
		
		// Check the norm of the gradient against convergence threshold
		if (g.norm() < relativeEpsilon)
			return fx;
		
		// Get a new descent direction using the L-BFGS algorithm
		VectorXd z = g;
		
		if (i && maxHistory_) {
			// Update the histories
			const int h = std::min(i, maxHistory_); // Current length of the histories
			const int end = (i - 1) % h;
			
			dxs.col(end) = x - px;
			dgs.col(end) = g - pg;
			
			// Initialize the variables
			VectorXd p(h);
			VectorXd a(h);
			
			for (int j = 0; j < h; ++j) {
				const int k = (end - j + h) % h;
				p(k) = 1.0 / dxs.col(k).dot(dgs.col(k));
				a(k) = p(k) * dxs.col(k).dot(z);
				z -= a(k) * dgs.col(k);
			}
			
			// Scaling of initial Hessian (identity matrix)
			z *= dxs.col(end).dot(dgs.col(end)) / dgs.col(end).dot(dgs.col(end));
			
			for (int j = 0; j < h; ++j) {
				const int k = (end + j + 1) % h;
				const double b = p(k) * dgs.col(k).dot(z);
				z += dxs.col(k) * (a(k) - b);
			}
		}
		
		// Save the previous state
		px = x;
		pg = g;
		
		// If z is not a valid descent direction (because of a bad Hessian estimation), restart the
		// optimization starting from the current solution
		double descent = -z.dot(g);
		
		if (descent > -0.0001 * relativeEpsilon) {
			z = g;
			i = 0;
			descent = -z.dot(g);
		}
		
		// Backtracking using Wolfe's first condition (Armijo condition)
		double step = i ? 1.0 : (1.0 / g.norm());
		bool down = false;
		int ls;
		
		for (ls = 0; ls < maxLineSearches_; ++ls) {
			// Tentative solution, gradient and loss
			const VectorXd nx = x - step * z;
			VectorXd ng(x.rows());
			const double nfx = (*function_)(nx.data(), ng.data());
			
			if (nfx <= fx + 0.0001 * step * descent) { // First Wolfe condition
				if ((-z.dot(ng) >= 0.9 * descent) || down) { // Second Wolfe condition
					x = nx;
					g = ng;
					fx = nfx;
					break;
				}
				else {
					step *= 2.0;
				}
			}
			else {
				step *= 0.5;
				down = true;
			}
		}
		
		if (function_->progress(argx, g.data(), static_cast<int>(x.rows()), fx, x.norm(), g.norm(),
								step, j + 1, ls + 1))
			return fx;
		
		if (ls == maxLineSearches_) {
			if (i)
				i = -1;
			else
				return fx;
		}
	}
	
	return fx;
}

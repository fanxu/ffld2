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

#ifndef LBFGS_H
#define LBFGS_H

/// Limited-memory BFGS (L-BFGS) algorithm implementation as described by Nocedal.
/// L-BFGS is an unconstrained quasi-Newton optimization method that uses a limited memory variation
/// of the Broyden–Fletcher–Goldfarb–Shanno (BFGS) update to approximate the inverse Hessian matrix.
/// The implementation is robust as it uses a simple line-search technique (backtracking in one
/// direction only) and still works even if the L-BFGS algorithm returns a non descent direction (as 
/// it will then restart the optimization starting from the current solution).
/// Its robustness enables it to minimize non-smooth functions, such as the hinge loss.
class LBFGS
{
public:
	/// Callback interface to provide objective function and gradient evaluations.
	class IFunction
	{
	public:
		/// Destructor.
		virtual ~IFunction();
		
		/// Returns the number of variables.
		virtual int dim() const = 0;
		
		/// Provides objective function and gradient evaluations.
		/// @param[in] x Current solution.
		/// @param[out] g The gradient vector which must be computed for the current solution.
		/// @returns The value of the objective function for the current solution.
		virtual double operator()(const double * x, double * g = 0) const = 0;
		
		/// Provides information about the current iteration.
		/// @param[in] x The current solution.
		/// @param[in] g The gradient vector of the current solution.
		/// @param[in] n The number of variables.
		/// @param[in] fx The current value of the objective function.
		/// @param[in] xnorm The Euclidean norm of the current solution.
		/// @param[in] gnorm The Euclidean norm of the gradient vector.
		/// @param[in] step The line-search step used for this iteration.
		/// @param[in] t The iteration count.
		/// @param[in] ls The number of evaluations called for this iteration.
		/// @returns whether to stop the optimization process.
		virtual bool progress(const double * x, const double * g, int n, double fx, double xnorm,
							  double gnorm, double step, int t, int ls) const;
	};
	
public:
	/// Constructor.
	/// @param[in] function Callback function to provide objective function and gradient
	/// evaluations.
	/// @param[in] epsilon Accuracy to which the solution is to be found.
	/// @param[in] maxIterations Maximum number of iterations allowed.
	/// @param[in] int maxLineSearches Maximum number of line-searches per iteration allowed.
	/// @param[in] maxHistory Maximum history length of previous solutions and gradients.
	LBFGS(const IFunction * function = 0, double epsilon = 1e-6, int maxIterations = 400,
		  int maxLineSearches = 40, int maxHistory = 10);
	
	/// Starts the L-BFGS optimization process.
	/// @param[in,out] x Initial solution on entry. Receives the optimization result on exit.
	/// @returns The final value of the objective function.
	double operator()(double * x) const;
	
private:
	// Constructor parameters
	const IFunction * function_;
	double epsilon_;
	int maxIterations_;
	int maxLineSearches_;
	int maxHistory_;
};

#endif

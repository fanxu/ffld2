%--------------------------------------------------------------------------
% Implementation of the papers "Exact Acceleration of Linear Object
% Detectors", 12th European Conference on Computer Vision, 2012 and "Deformable
% Part Models with Individual Part Scaling", 24th British Machine Vision
% Conference, 2013.
%
% Copyright (c) 2013 Idiap Research Institute, <http://www.idiap.ch/>
% Written by Charles Dubout <charles.dubout@idiap.ch>
%
% This file is part of FFLDv2 (the Fast Fourier Linear Detector version 2)
%
% FFLDv2 is free software: you can redistribute it and/or modify it under the
% terms of the GNU Affero General Public License version 3 as published by the
% Free Software Foundation.
%
% FFLDv2 is distributed in the hope that it will be useful, but WITHOUT ANY
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
% details.
%
% You should have received a copy of the GNU Affero General Public License
% along with FFLDv2. If not, see <http://www.gnu.org/licenses/>.
%--------------------------------------------------------------------------

function mixture = readmixture(filename)
% Read a mixture of deformable part models.
% mixture is a cell array of size k x 2, where k is the number of mixture
% components. For each component the cell in the first column stores the model
% while the second stores its bias. A model is a n x 3 cell array where n is
% the number of parts. The first column stores the part filter, the second the
% offset (x y z), and the third the deformation cost (x^2 x y^2 y z^2 z).
% The part filter is a 3 dimensional matrix of size height x width x features.

fileID = fopen(filename, 'r');
nbModels = fscanf(fileID, '%d', 1);
mixture = cell(nbModels, 2);

for i = 1:nbModels
    nbParts = fscanf(fileID, '%d', 1);
    mixture{i,2} = fscanf(fileID, '%f', 1); % bias
    model = cell(nbParts, 3);
    
    for j = 1:nbParts
        sz = fscanf(fileID, '%d', 3);
        model{j, 2} = fscanf(fileID, '%d', 3); % offset
        model{j, 3} = fscanf(fileID, '%f', 6); % deformation
        w = zeros(sz(1), sz(2), sz(3)); % filter
        
        for y = 1:sz(1)
            w(y, :, :) = reshape(fscanf(fileID, '%f', sz(2) * sz(3)), ...
                                 [sz(3) sz(2)])';
        end
        
        model{j, 1} = w;
    end
    
    mixture{i, 1} = model;
end

fclose(fileID);

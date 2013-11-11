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

function writemixture(mixture, filename)
% Write a mixture of deformable part models.

fileID = fopen(filename, 'w');
nbModels = size(mixture, 1);
fprintf(fileID, '%d\n', nbModels);

for i = 1:nbModels
    parts = mixture{i, 1};
    nbParts = size(parts, 1);
    fprintf(fileID,'%d %f\n', nbParts, mixture{i, 2});
    
    for j = 1:nbParts
        w = parts{j, 1};
        
        fprintf(fileID,'%d %d %d %d %d %d %f %f %f %f %f %f\n', size(w), ...
                parts{j, 2}, parts{j, 3});
        
        for y = 1:size(w,1)
            for x = 1:size(w,2)
                fprintf(fileID, '%f ', w(y, x, :));
            end
            
            fprintf(fileID, '\n');
        end
    end
end

fclose(fileID);

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

function convertmodel5(model, filename)
%
% convertmodel5(model, filename) convert the models of [1, 2] version 5 into
% a file readable by FFLDv2.
%
% [1] P. Felzenszwalb, R. Girshick, D. McAllester and D. Ramanan.
%     Object Detection with Discriminatively Trained Part Based Models.
%     IEEE Transactions on Pattern Analysis and Machine Intelligence,
%     Vol. 32, No. 9, 2010
%
% [2] R. Girshick, P. Felzenszwalb, and D. McAllester.
%     Discriminatively Trained Deformable Part Models, Release 5.
%     <http://people.cs.uchicago.edu/~rbg/latent-release5/>
%

fileID = fopen(filename, 'w');
nbModels = length(model.rules{model.start});
fprintf(fileID, '%d\n', nbModels);

for i = 1:nbModels
    rhs = model.rules{model.start}(i).rhs;
    nbParts = length(rhs);
    
    % Assume the root filter is first on the rhs of the start rules
    if model.symbols(rhs(1)).type == 'T'
        % Handle case where there's no deformation model for the root
        root = model.symbols(rhs(1)).filter;
        bias = 0;
    else
        % Handle case where there is a deformation model for the root
        root = model.symbols(model.rules{rhs(1)}(1).rhs).filter;
        bias = model_get_block(model, model.rules{model.start}(i).offset) * model.features.bias;
    end
    
    fprintf(fileID, '%d %f\n', nbParts, bias);
    
    % FFLD adds instead of subtracting the deformation cost
    def = -model_get_block(model, model.rules{rhs(1)}(1).def);
    
    % Swap features 28 and 31
    w = model_get_block(model, model.filters(root));
    w = w(:, :, [1:27 31 29 30 28 32]);
    
    fprintf(fileID, '%d %d %d 0 0 0 %f %f %f %f 0 0\n', size(w,1), ...
            size(w,2), size(w,3), def);
    
    for y = 1:size(w,1)
        for x = 1:size(w,2)
            fprintf(fileID, '%f ', w(y, x, :));
        end
        
        fprintf(fileID, '\n');
    end
    
    for j = 2:nbParts
        part = model.symbols(model.rules{rhs(j)}(1).rhs).filter;
        anc = model.rules{model.start}(i).anchor{j};
        def = -model_get_block(model, model.rules{rhs(j)}(1).def);
        
        w = model_get_block(model, model.filters(part));
        w = w(:, :, [1:27 31 29 30 28 32]);
        
        fprintf(fileID, '%d %d %d %d %d 0 %f %f %f %f 0 0\n', size(w,1), ...
                size(w,2), size(w,3), anc(1), anc(2), def);
        
        for y = 1:size(w,1)
            for x = 1:size(w,2)
                fprintf(fileID, '%f ', w(y, x, :));
            end
            
            fprintf(fileID, '\n');
        end
    end
end

fclose(fileID);

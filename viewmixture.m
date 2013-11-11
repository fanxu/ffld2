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

function im = viewmixture(mixture, cellSize, pad)
% Create a representation of a mixture of deformable part models.
% cellSize is the size in pixels of a HOG cell.
% pad is the amount of padding in HOG cells to add to the image.

if nargin < 2
    cellSize = 20;
end

if nargin < 3
    pad = 0;
end

row = 1;
col = 1;
scale = inf;

for i = 1:size(mixture, 1)
    root = foldhog(mixture{i, 1}{1, 1});
    scale = min(scale, max(root(:)));
end

for i = 1:size(mixture, 1)
    root = foldhog(mixture{i, 1}{1, 1});
	root = [zeros(pad,size(root, 2) + 2 * pad, 9); ...
            zeros(size(root, 1), pad, 9) root zeros(size(root, 1), pad, 9); ...
            zeros(pad,size(root, 2) + 2 * pad, 9)];
    
    imroot = imhog(root, 2 * cellSize) / scale;
    row = max(row, size(imroot, 1));
    
    if i > 1
        im(:, col:col + 2 * cellSize - 1, :) = 0.5;
        col = col + 2 * cellSize;
    end
    
    im(:, col:col + size(imroot, 2) - 1) = 0.5;
    off = floor((row - size(imroot, 1)) / 2);
    im(off + 1:off + size(imroot, 1), col:col + size(imroot, 2) - 1) = imroot;
    
    if size(mixture{i, 1}, 1) > 1
        parts = zeros(2 * (size(root, 1) + 2 * pad), ...
                      2 * (size(root, 2) + 2 * pad), 9);
        
        for j = 2:size(mixture{i, 1}, 1)
            anchor = mixture{i, 1}{j, 2} + 1;
            x = anchor(1) + 2 * pad;
            y = anchor(2) + 2 * pad;
            w = size(mixture{i, 1}{j, 1}, 2);
            h = size(mixture{i, 1}{j, 1}, 1);
            
            parts(y:y + h - 1, x:x + w - 1, :) = ...
                max(parts(y:y + h - 1, x:x + w - 1, :), ...
                    foldhog(mixture{i, 1}{j, 1}));
        end
        
        imparts = imhog(parts, cellSize) / scale;
        
        for j = 2:size(mixture{i, 1}, 1)
            anchor = mixture{i, 1}{j, 2} + 1;
            x = anchor(1) + 2 * pad;
            y = anchor(2) + 2 * pad;
            w = size(mixture{i, 1}{j, 1}, 2);
            h = size(mixture{i, 1}{j, 1}, 1);
            
            left   = max(cellSize * x + 1, 2);
            top    = max(cellSize * y + 1, 2);
            right  = min(left + cellSize * w, size(imparts, 2) - 1);
            bottom = min(top  + cellSize * h, size(imparts, 1) - 1);
            
            imparts(top - 1:top + 1, left - 1:right + 1) = 1;
            imparts(bottom - 1:bottom + 1, left - 1:right+1) = 1;
            imparts(top - 1:bottom + 1, left - 1:left + 1) = 1;
            imparts(top - 1:bottom + 1, right - 1:right + 1) = 1;
        end
        
        im(row + off + 1:row + 2 * cellSize + off, ...
           col:col + size(imparts, 2) - 1) = 0.5;
        im(row + 2 * cellSize + off + 1: ...
           row + 2 * cellSize + off + size(imparts, 1), ...
           col:col + size(imparts, 2) - 1) = imparts;
    end
    
    col = col + size(imroot, 2);
end

im = 1 - min(max(im, 0), 1).^2;

if ~nargout
    imshow(im);
end

function m = foldhog(m)
    m = m(:, :, 1:9) + m(:, :, 10:18) + m(:, :, 19:27);

function im = imhog(m, cellSize)
    % Construct a "glyph" for each orientation
    glyphs = zeros(cellSize, cellSize, 9);
    
    for i = 1:9
        glyph = zeros(4 * cellSize);
        dx = sin((i - 1) * pi / 9);
        dy =-cos((i - 1) * pi / 9);
        [X, Y] = meshgrid(0.5 - 2 * cellSize:2 * cellSize - 0.5, ...
                          0.5 - 2 * cellSize:2 * cellSize - 0.5);
        d = X * dx + Y * dy;
        d = (X - d * dx).^2 + (Y - d * dy).^2;
        glyph((d < (cellSize / 4)^2) & (X.^2 + Y.^2 < 4 * cellSize^2)) = 1;
        glyphs(:, :, i) = imresize(glyph, 0.25);
    end
    
    % Make an image by adding up weighted glyphs
    im = zeros(cellSize * size(m, 1), cellSize * size(m, 2));
    
    for x = 1:size(m, 2)
        for y = 1:size(m, 1)
            for i = 1:9
                im(cellSize * (y - 1) + 1:cellSize * y, ...
                   cellSize * (x - 1) + 1:cellSize * x) = ...
                    max(im(cellSize * (y - 1) + 1:cellSize * y, ...
                           cellSize * (x - 1) + 1:cellSize * x), ...
                        m(y, x, i) * glyphs(:, :, i));
            end
        end
    end

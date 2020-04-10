function [ patch ] = read_region( image_path, level, patch_rect )
% patch_rect – (x1, x2, y1, y2)
    pixel_region = {[patch_rect(1),patch_rect(2)],[patch_rect(3),patch_rect(4)]};
    patch = imread(image_path,'ReductionLevel',level,'PixelRegion',pixel_region);
end


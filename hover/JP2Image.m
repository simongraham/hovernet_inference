function [ level_dimensions, level_downsamples, level_count] = JP2Image( image_path )
%JP2 Summary of this function goes here
%   Detailed explanation goes here
    info = imfinfo(image_path);
    level_count = log2(info.CodeBlockDims(1))+1;
    for i = 0 : level_count
        level_downsamples(i+1,:) = 2^i;
        level_dimensions(i+1,:) = int32([info.Height/2^i,info.Width/2^i]);
    end
end


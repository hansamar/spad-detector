% READ_TDC_FRAME_CUBE_MATLAB 读取 spad-detector tdc_frame_cube .bin 文件
%
% 用法:
%   tdc = read_tdc_frame_cube_matlab('tdc_frame_cube.bin');
%   tdc = read_tdc_frame_cube_matlab('tdc_frame_cube.bin', 'tdc_frame_cube.metadata.json');

function tdc = read_tdc_frame_cube_matlab(bin_path, metadata_path)
    if nargin < 2
        [fpath, ~, ~] = fileparts(bin_path);
        meta_candidate = fullfile(fpath, 'tdc_frame_cube.metadata.json');
        if isfile(meta_candidate)
            metadata_path = meta_candidate;
        end
    end

    if ~exist('metadata_path', 'var') || isempty(metadata_path)
        error('未找到 metadata.json');
    end

    meta = jsondecode(fileread(metadata_path));
    shape = meta.shape;
    dtype = meta.dtype;
    empty_val = meta.empty_pixel_value;

    fprintf('TDC frame cube:\n');
    fprintf('  shape: [%d, %d, %d]\n', shape(1), shape(2), shape(3));
    fprintf('  empty_pixel_value: %d\n', empty_val);
    if isfield(meta, 'valid_tdc_range')
        fprintf('  valid_tdc_range: [%d, %d]\n', meta.valid_tdc_range(1), meta.valid_tdc_range(2));
    end
    if isfield(meta, 'collision_policy')
        fprintf('  collision_policy: %s\n', meta.collision_policy);
    end
    fprintf('  range_bin_m: %.4f\n', meta.range_bin_m);
    fprintf('  max_unambiguous_range_m: %.1f\n', meta.max_unambiguous_range_m);

    % 读取
    fid = fopen(bin_path, 'r');
    data = fread(fid, inf, dtype);
    fclose(fid);

    tdc = permute(reshape(data, [shape(3), shape(2), shape(1)]), [3, 2, 1]);

    % 统计
    valid_mask = tdc ~= empty_val;
    n_valid = sum(valid_mask(:));
    fprintf('\n统计:\n');
    fprintf('  有效像素数: %d\n', n_valid);
    fprintf('  空像素数: %d\n', sum(~valid_mask(:)));
    if n_valid > 0
        valid_bins = tdc(valid_mask);
        fprintf('  TDC bin 范围: [%d, %d]\n', min(valid_bins), max(valid_bins));
        fprintf('  TDC bin 均值: %.1f\n', mean(double(valid_bins)));
    end
end

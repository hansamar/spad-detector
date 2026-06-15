% READ_COUNT_CUBE_MATLAB 读取 spad-detector count_cube .bin 文件
%
% 用法:
%   cube = read_count_cube_matlab('counts.bin');
%   cube = read_count_cube_matlab('counts.bin', 'counts.metadata.json');

function cube = read_count_cube_matlab(bin_path, metadata_path)
    if nargin < 2
        % 尝试自动查找 metadata.json
        [fpath, fname, ~] = fileparts(bin_path);
        meta_candidate = fullfile(fpath, [fname, '.metadata.json']);
        if isfile(meta_candidate)
            metadata_path = meta_candidate;
        else
            % 尝试 counts.metadata.json
            meta_candidate = fullfile(fpath, 'counts.metadata.json');
            if isfile(meta_candidate)
                metadata_path = meta_candidate;
            end
        end
    end

    if exist('metadata_path', 'var') && ~isempty(metadata_path)
        meta = jsondecode(fileread(metadata_path));
        shape = meta.shape;        % [n_frames, roi_h, roi_w]
        dtype = meta.dtype;        % 'uint16'

        fprintf('从 metadata 读取:\n');
        fprintf('  format: %s\n', meta.format);
        fprintf('  shape: [%d, %d, %d]\n', shape(1), shape(2), shape(3));
        fprintf('  dtype: %s\n', dtype);
        fprintf('  sample_rate_hz: %.1f\n', meta.sample_rate_hz);
        fprintf('  detector_preset: %s\n', meta.detector_preset);
        fprintf('  random_seed: %d\n', meta.random_seed);
    else
        error('未找到 metadata.json，无法确定 shape/dtype');
    end

    % 读取二进制文件
    fid = fopen(bin_path, 'r');
    if fid == -1
        error('无法打开文件: %s', bin_path);
    end
    data = fread(fid, inf, dtype);
    fclose(fid);

    % reshape — frame-major 布局
    % index = frame * roi_h * roi_w + row * roi_w * col
    cube = reshape(data, [shape(3), shape(2), shape(1)])';  % [roi_w, roi_h, n_frames] -> [n_frames, roi_h, roi_w]
    cube = permute(reshape(data, [shape(3), shape(2), shape(1)]), [3, 2, 1]);

    fprintf('\ncount_cube 加载完成:\n');
    fprintf('  总光子数: %d\n', sum(cube(:)));
    fprintf('  均值: %.2f\n', mean(cube(:)));
    fprintf('  最大值: %d\n', max(cube(:)));
    fprintf('  有检测的像素比例: %.4f\n', mean(cube(:) > 0));
end

function runKIFD(input_mat_path, output_mat_path, zeta, tree_num, subsample_percentage, a_threshold)
% runKIFD - wrapper to call KIFD from Python
% Usage:
%  runKIFD('data/porcine4.mat','out/tmp_porcine4.mat', 300, 1000, 3, 1900)
%
% Notes:
% - zeta : principal components for KPCA (if zeta < bands perform KPCA)
% - tree_num : number of iForest trees (q)
% - subsample_percentage : percent used to compute tree_size = floor(subsample_percentage*N/100)
% - a_threshold : area threshold "a" (absolute pixel count). If <=0, we fallback to N/120.


S = load(input_mat_path);
if isfield(S,'data')
    data1 = double(S.data);
else
    error('Input .mat must contain variable "data"');
end

[row, col, bands] = size(data1);
N = row * col;

% normalize
data2 = NormalizeData(data1);

% KPCA decision: perform KPCA only if zeta < bands (keeps original behavior from your script)
if zeta < bands
    disp(['Performing KPCA: reducing ', num2str(bands), ' -> ', num2str(zeta), ' components...']);
    data_kpca = kpca(data2, 10000, zeta, 'Gaussian', 1);
    data = NormalizeData(data_kpca);
else
    disp(['Skipping KPCA because zeta (', num2str(zeta), ') >= number of bands (', num2str(bands), ').']);
    data = data2;
end

data = ToVector(data);

% compute tree_size
if subsample_percentage <= 0
    subsample_percentage = 3; % default
end
tree_size = floor(subsample_percentage * row * col / 100);

% default a_threshold fallback
if isempty(a_threshold) || a_threshold <= 0
    a_threshold = max(1, round(N / 120));
end

% Run global iForest
s = iforest(data, tree_num, tree_size);

% Run local iForest iterative refinement (keeps your loop)
img = reshape(s, row, col);
stop_flag = 0;
index = [];
num_iter = 1;
r0 = img;
lev = graythresh(r0);

while stop_flag == 0
    [r1, flag, s1, index1] = Local_iforest(r0, data, s, index, lev, a_threshold);
    r0 = r1;
    s = s1;
    index = index1;
    stop_flag = flag;
    num_iter = num_iter + 1;
    if num_iter > 100
        break;
    end
end

img = zeros(row, col);
img(index1) = 1;
index = (1:row*col)';
index(index1, :) = [];
Data_d = data(:, :);
Data_d(index1, :) = [];
s_d = iforest(Data_d, tree_num, tree_size);
r1(index) = s_d;
r2 = 10.^r1;

show = mat2gray(r2);

save(output_mat_path, 'show');

end
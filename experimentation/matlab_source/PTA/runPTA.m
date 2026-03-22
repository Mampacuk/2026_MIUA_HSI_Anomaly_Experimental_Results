function runPTA(input_mat_path, output_mat_path, truncate_rank, alphia, mu, beta, tau, maxiter)
% runPTA - wrapper that loads input .mat, runs PTA, saves 'show' to output_mat_path
% Usage:
%   runPTA('data/porcine4.mat', 'out/tmp_porcine4.mat', 5, 1.0, 0.1, 1e-2, 1, 400)

% load
S = load(input_mat_path);
if isfield(S,'data')
    DataTest = double(S.data);
else
    error('Input .mat does not contain variable "data"');
end

if isfield(S,'map')
    mask = double(S.map);
elseif isfield(S,'mask')
    mask = double(S.mask);
else
    error('Input .mat must contain ground truth variable "map" or "mask"');
end

[H,W,Dim] = size(DataTest);
num = H*W;

% normalize per band (same as your original)
for i=1:Dim
    band = DataTest(:,:,i);
    mn = min(band(:));
    mx = max(band(:));
    if mx>mn
        DataTest(:,:,i) = (band - mn) / (mx - mn);
    else
        DataTest(:,:,i) = band - mn; % constant band -> zero
    end
end

% reshape masks
mask_reshape = reshape(mask, 1, num);
anomaly_map = logical(double(mask_reshape) > 0);
normal_map = logical(double(mask_reshape) == 0);

Y = reshape(DataTest, num, Dim)';

% Parameters passed in: truncate_rank, alphia, mu, beta, tau, maxiter

tol1 = 1e-4;
tol2 = 1e-6;

% call original PTA solver (your AD_Tensor_LILU1)
[X,S_anom,area] = AD_Tensor_LILU1(DataTest, alphia, beta, tau, mu, truncate_rank, maxiter, tol1, tol2, normal_map, anomaly_map);

show = sqrt(sum(S_anom.^2, 3));
if max(show(:)) > min(show(:))
    show = (show - min(show(:))) / (max(show(:)) - min(show(:)));
else
    show = zeros(size(show));
end

save(output_mat_path, 'show');

end
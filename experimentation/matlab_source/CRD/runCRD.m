function runCRD(input_mat_path, output_mat_path, w_in, w_out, lamda)

data = load(input_mat_path);
hsi = double(data.data);

[rows,cols,bands]=size(hsi);

disp('Running CRD, Please wait...')

show = func_CRD(hsi, w_out, w_in, lamda);

show=(show-min(show(:)))/(max(show(:))-min(show(:)));

save(output_mat_path,'show')

end
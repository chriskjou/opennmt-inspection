function [] = calculate_llh_significance_across_layers(batch_num, total_batches)
clear alpha exp_r xp pxp bor

% get model
model = 'all_best_llh_by_voxel.mat';

vals = load(model); 
vals_shape = size(vals.metric);
num_voxels = vals_shape(1);
num_subjects = vals_shape(2);
num_layers = vals_shape(3);

pvals = [];
pxps = [];

% get voxel numbers
if mod(num_voxels, total_batches) == 0
	chunk_size = floor(num_voxels / total_batches);
else
	chunk_size = floor(num_voxels / total_batches) + 1;
end

if batch_num == 0
	start_voxel = 1
else
	start_voxel = batch_num * chunk_size + 1
end

if batch_num ~= total_batches - 1
	end_voxel = batch_num * chunk_size + chunk_size;
else
	end_voxel = num_voxels;
end

tic
for i=start_voxel:end_voxel
    lme = vals.metric(i,:,:);
    lme = squeeze(lme);
	% lme_reshape = reshape(lme,[num_subjects,num_layers]);
    [alpha,exp_r,xp,pxp,bor] = bms(lme_reshape);
    pvals = [pvals, bor];
    pxps = [pxps, pxp];
end
toc

llh.bor = pvals;
pxps_reshape = reshape(pxps,[end_voxel-start_voxel+1,num_layers]);
llh.pxp = pxps_reshape;
file_name = strcat('../llh_mat/all_batch', num2str(batch_num), 'of', num2str(total_batches), '_bor_pxp.mat');
save(file_name,'llh')

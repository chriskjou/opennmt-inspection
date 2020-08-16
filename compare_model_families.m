function [] = compare_model_families(batch_num, total_batches)

% add paths
addpath(genpath('/n/home09/cjou/projects/VBA-toolbox/'))

% get model
model = '../mfit/nested_all_best_llh_by_voxel.mat';

vals = load(model); 
vals_shape = size(vals.metric);
num_voxels = vals_shape(1);
num_subjects = vals_shape(2);
num_layers = vals_shape(3);
num_families = 3;

pvals = [];
pxps = [];
families_pxps = [];

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

% example 
options.families = {[1,2,3,4,5,6,7,8,9,10,11,12], [13,14], [15,16,17,18]} ;
options.DisplayWin = false;
options.verbose = false;
for i=start_voxel:end_voxel
     lme = vals.metric(i,:,:);
     L = double(reshape(lme,[num_layers,num_subjects]));
     [pos, out] = VBA_groupBMC(L, options) ;
     pvals = [pvals, out.bor];
     pxps = [pxps, out.ep];
     families_pxps = [families_pxps, out.families.ep];
end

llh.bor = pvals;
pxps_reshape = reshape(pxps,[end_voxel-start_voxel+1,num_layers]);
llh.pxp = pxps_reshape;
family_reshape = reshape(families_pxps,[end_voxel-start_voxel+1,num_families]);
llh.family = families_pxps;
file_name = strcat('../nested_llh_mat/family_bor_pxp', num2str(batch_num), 'of', num2str(total_batches), '_bor_pxp.mat');
save(file_name,'llh')

disp('done.');
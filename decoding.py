import scipy.io
from tqdm import tqdm 
import pickle
import scipy.spatial as spatial
import numpy as np
from scipy.linalg import solve
import sys
from scipy.spatial import distance_matrix

# get initial information from MATLAB
def get_activations(info):
	print("getting activations...")
	mat = scipy.io.loadmat(info)

	activations = mat["examples_sentences"]
	volmask = mat["volmask"]
	atlasvals = mat["multimask_aal"]
	roi_labels = mat["labels_langloc"]
	atlas_labels = mat["labels_aal"]


	print("writing to file...")
	pickle.dump( activations, open( "activations.p", "wb" ) )
	pickle.dump( volmask, open( "volmask.p", "wb" ) )

	print("finished.")
	return

# https://stackoverflow.com/questions/32424604/find-all-nearest-neighbors-within-a-specific-distance
# get masks for every nonzero point
def get_all_spheres(volmask, title, radius=5, saved=False):
	if not saved:
		nonzero_pts = np.transpose(np.nonzero(volmask))
		pts_mask = []

		print("finding all masks...")
		for pt in tqdm(nonzero_pts):
			# only extract points with radius
			# within_radius = []
			# x1,y1,z1 = pt
			# for pt2 in nonzero_pts:
			# 	x2,y2,z2 = pt2
			# 	if abs(x1-x2) + abs(y1-y2) + abs(z1-z2) <= 5:
			# 		within_radius.append(pt2)

			# generate possible space
			x1,y1,z1 = pt
			within_radius = []
			for i in range(radius+1):
				for j in range(radius+1):
					for k in range(radius+1):
						pt2 = [x1+i,y1+j,z1+k]
						if pt2 in nonzero_pts:
							within_radius.append(pt2)
						
			point_tree = spatial.cKDTree(within_radius)
			nbr = point_tree.data[point_tree.query_ball_point(pt, radius)]
			pts_mask.append(nbr)

		pickle.dump( neighbors, open( "masks-" + str(title)+ ".p", "wb" ) )
	else:
		pts_mask = pickle.load( open( "activations.p", "rb" ) )
	return pts_mask

# reshape activations per sentence

### CHECK IF NEEDED FOR RESHAPING FIX?
def reshape_sentence_activations(activations, volmask):
	print("reshaping sentence activations...")
	i,j,k = volmask.shape
	sentences_activations = []
	for st in tqdm(activations):
		st_act = np.reshape(st, (i,j,k))
		sentences_activations.append(st_act)
	return sentences_activations
### CHECK IF NEEDED FOR RESHAPING FIX?

# get activations for all spotlights in one sentence
def get_activations_for_single_sentence(sent_activations, pts_mask, volmask):
	i,j,k = volmask.shape

	# over all points in a spotlight
	mask = np.zeros((i,j,k))
	for pt in pts_mask:
		x,y,z = pt
		mask[x][y][z] = 1
	flat_mask = mask.flatten()

		# get average activation per mask
	final_acts = sent_activations[flat_mask]
	avg_activation = np.true_divide(final.sum(1),(final!=0).sum(1))
	return avg_activation

# get activations of spotlights x sentence
def activations_for_all_sentences(activations, pt_mask, volmask):
	print("getting activations for all sentences...")
	per_sentence = []
	for mask in tqdm(pt_mask):
		spotlights = []
		for st in tqdm(activations):
			spot = get_activations_for_single_sentence(st, mask, volmask)
			spotlights.append(spot)
		per_sentence.append(spotlights)
	return per_sentence

def linear_model(embedding, spotlight_activation):
	coeffs = solve(embedding, spotlight_activation)
	return coeffs

def main():
	### GET MAIN INPUT BELOW
	if len(sys.argv) != 4:
		print("usage: python decoding.py -embedding_layer -examplesGLM.mat -title")
		exit()

	embedding = sys.argv[1]
	info = sys.argv[2]
	title = sys.argv[3]
	### GET MAIN INPUT ABOVE

	# CHECK COORDINATES - NEED TO

	### PROCESSING FROM BRAIN BELOW
	saved = True
	if not saved:
		get_activations(info)

		print("saved activations.")
	else:
		print("loading activations and mask...")
		activations = pickle.load( open( "activations.p", "rb" ) )
		volmask = pickle.load( open( "volmask.p", "rb" ) )

	saved = False
	pts_mask = get_all_spheres(volmask, title, saved)
	spotlight_acts_per_sentence = activations_for_all_sentences(activations, pts_mask, volmask)
	### PROCESSING FROM BRAIN ABOVE

	### DECODING BELOW 
	coeffs_per_spotlight = []
	for spotlight in spotlight_acts_per_sentence:
		coeff = linear_model(embedding, spotlight)
		coeffs_per_spotlight.append(coeff)
	### DECODING ABOVE

	### RUN SIGNIFICANT TESTS BELOW

	### RUN SIGNIFICANCE TESTS ABOVE
	return

if __name__ == "__main__":
    main()
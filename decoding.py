import scipy.io
from tqdm import tqdm 
import pickle
import numpy as np
from scipy.linalg import solve
import sys
import math

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
	return activations, volmask

# https://stackoverflow.com/questions/32424604/find-all-nearest-neighbors-within-a-specific-distance
# get masks for every nonzero point
def get_all_spheres(volmask, title, saved=False, radius=5):
	if not saved:
		a,b,c = volmask.shape
		nonzero_pts = np.transpose(np.nonzero(volmask))
		pts_mask = []

		print("finding spotlights...")
		for pt in tqdm(nonzero_pts):
			sphere_mask = np.zeros((a,b,c))
			x1,y1,z1 = pt
			within_radius = [pt]
			count = 0
			for i in range(-radius, radius+1):
				for j in range(-radius, radius+1):
					for k in range(-radius, radius+1):
						xp = x1 + i
						yp = y1 + j
						zp = z1 + k
						pt2 = [xp,yp,zp]
						if 0 <= xp and 0 <= yp and 0 <= zp and xp < a and yp < b and zp < c:
							dist = math.sqrt(i ** 2 + j ** 2 + k ** 2) #distance.euclidean(pt, pt2) # can remove
							if pt2 in nonzero_pts and dist <= radius:
								sphere_mask[x1+i][y1+j][z1+k] = 1
								within_radius.append(pt2)
								count+=1

			pts_mask.append(sphere_mask)

			# if len(pts_mask) == 10:
			# 	pickle.dump( pts_mask, open( "test-" + str(title)+ ".p", "wb" ) )
			# 	exit(1)

			# no neighbors in spotlight
			# if len(within_radius) == 1:
			# 	pts_mask.append([pt])
			# else:
			# 	# print("HAS NEIGHBORS")
			# 	# print("PT")
			# 	# print(pt)
			# 	point_tree = spatial.cKDTree(within_radius)
			# 	nbr = point_tree.data[point_tree.query_ball_point(pt, radius)]
			# 	# print("NEIGHBORS")
			# 	# print(len(nbr))
			# 	pts_mask.append([nbr])

		# pickle.dump( pts_mask, open( "spotlight-" + str(title)+ ".p", "wb" ) )
	else:
		print("loading spotlights...")
		pts_mask = pickle.load( open( "test-" + str(title)+ ".p", "rb" ) )
		# pts_mask = pickle.load( open( "spotlight-" + str(title)+ ".p", "rb" ) )
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
def get_activations_for_single_sentence(sentence_act, pts_mask, volmask):
	print("getting activation for single sentence...")
	i,j,k = volmask.shape
	nonzero_pts = np.transpose(np.nonzero(volmask))
	# print(len(nonzero_pts))

	# MAKE ACTIVATION MASK BELOW
	act_mask = np.zeros((i,j,k))
	for pt in range(len(nonzero_pts)):
		x,y,z = nonzero_pts[pt]
		act_mask[int(x)][int(y)][int(z)] = sentence_act[pt]
	# MAKE ACTIVATION MASK ABOVE

	# GET ACTIVATIONS FOR SPECIFIC SPOTLIGHT BELOW
	num_of_activations_in_spotlight = np.sum(pts_mask[0])
	spotlight_act = act_mask[pts_mask.astype(bool)]
	avg_activation = np.nansum(spotlight_act) * 1. / num_of_activations_in_spotlight
	print("AVERAGE ACTIVATIONS")
	print(avg_activation)
	# GET ACTIVATIONS FOR SPECIFIC SPOTLIGHT ABOVE


	# print(sent_activations.shape)
	# if len(pts_mask) == 1:
	# 	print("HERE")
	# 	print(pts_mask[0])
	# 	return sent_activations[pts_mask[0]]

	# corresponding mask

	# print("SHAPE")
	

	# over all points in a spotlight
	# mask = np.zeros((i,j,k), dtype=bool)
	# # print(pts_mask)
	# # print(len(pts_mask))
	# for pt in pts_mask:
	# 	print(len(pt))
	# 	print("IN MASK")
	# 	print(pt)
	# 	x,y,z = pt
	# 	mask[int(x)][int(y)][int(z)] = 1

	# flat_mask = mask.flatten()

	# # get average activation per mask
	# final_acts = flat_act_mask[flat_mask]
	# # print(final_acts)
	# avg_activation = np.mean(final_acts)
	# # print(avg_activation)
	# # avg_activation = np.true_divide(final.sum(1),(final!=0).sum(1))
	return avg_activation

# get activations of spotlights x sentence
def activations_for_all_sentences(activations, neighbors_mask, volmask):
	print("getting activations for all sentences...")
	per_sentence = []

	print("NUMBER OF SPOTLIGHT MASKS: ", len(neighbors_mask))
	# iterate over spotlight
	for mask in tqdm(neighbors_mask):
		print("CHECKING MASK SHAPE: ", mask.shape)
		spotlights = []
		# iterate over each sentence
		for sentence_act in activations:
			print("ACTIVATIONS")
			print(sentence_act)
			spot = get_activations_for_single_sentence(sentence_act, mask, volmask)
			spotlights.append(spot)
		print("NUMBER OF SPOTLIGHT ACTIVATIONS: ", len(spotlights))
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
		activations, volmask = get_activations(info)
		print("saved activations.")
	else:
		print("loading activations and mask...")
		activations = pickle.load( open( "activations.p", "rb" ) )
		volmask = pickle.load( open( "volmask.p", "rb" ) )

	neighbors_mask = get_all_spheres(volmask, title, saved=False)

	# RANDOM DELETE BELOW
	print("VOLMASK")
	a,b,c = volmask.shape
	neighbors_mask = [np.random.randint(2, size=(a,b,c))]
	# RANDOM DELETE ABOVE
	print("PROCESSING")
	spotlight_acts_per_sentence = activations_for_all_sentences(activations, neighbors_mask, volmask)
	### PROCESSING FROM BRAIN ABOVE

	### DECODING BELOW 
	# coeffs_per_spotlight = []
	# for spotlight in spotlight_acts_per_sentence:
	# 	coeff = linear_model(embedding, spotlight)
	# 	coeffs_per_spotlight.append(coeff)
	### DECODING ABOVE

	### RUN SIGNIFICANT TESTS BELOW

	### RUN SIGNIFICANCE TESTS ABOVE
	return

if __name__ == "__main__":
    main()
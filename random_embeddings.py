import pickle
import os
import numpy as np 

def get_random_embeddings(file, dim=500):
	embed_matrix = np.random.uniform(low=-0.5, high=0.5, size=(len(file), dim))
	return embed_matrix

def main():
	file = open("cleaned_sentencesGLM.txt","r").read().splitlines()

	embed_matrix = get_random_embeddings(file)
	print(np.shape(embed_matrix))

	bool_labels = [1]*len(file)

	if not os.path.exists('../embeddings/rand_embed/'):
		os.makedirs('../embeddings/rand_embed/')

	print("saving files...")
	pickle.dump(embed_matrix, open("../embeddings/rand_embed/rand_embed.p", "wb"))


	print("done.")

if __name__ == "__main__":
	main()
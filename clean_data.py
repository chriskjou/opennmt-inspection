import scipy.io
from tqdm import tqdm 

print("loading sentences...")
mat = scipy.io.loadmat('examplesGLM.mat')

sentences = mat["sentences"]

file = open("cleaned_sentencesGLM.txt","w") 

print("writing to file...")

for s in tqdm(sentences):
	line = str(s[0][0]).replace("_", " ")
	file.write(line + "\n")
file.close()

print("finished.")
import sys
from itertools import izip
from tqdm import tqdm

def main():
	if len(sys.argv) != 3:
		print("usage: python find_common_sentence.py -src.txt -newfilename")
		exit()

	src = open(sys.argv[1])
	newfile = open(sys.argv[2],"w") 

	print("cleaning spaces...")
	with src as file:
		for line in tqdm(file):
			line = line.strip()
			if line != "":
				newfile.write(line + "\n")

	print("finished writing to file.")
	return

if __name__ == "__main__":
    main()
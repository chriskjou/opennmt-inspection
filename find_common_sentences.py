import sys
from itertools import izip
from tqdm import tqdm

def main():
	if len(sys.argv) != 5:
		print("usage: python find_common_sentence.py -srclanguageA.txt -tgtlanguageB.txt -sortedsrclanguageA.txt -newfilename")
		exit()

	link_dict = {}

	# open files
	srclangA = open(sys.argv[1])
	tgtlangB = open(sys.argv[2])
	sortedsrclangA = open(sys.argv[3])
	newfile = open(sys.argv[4],"w") 

	print("creating dictionary...")
	with srclangA as file1, tgtlangB as file2: 
	    for x, y in tqdm(izip(file1, file2)):
			x = x.strip()
			y = y.strip()
			link_dict.setdefault(x, 0)
			link_dict[x] = y

	print("finding sentences...")
	with sortedsrclangA as sortedfile:
		for line in tqdm(sortedfile):
			line = line.strip()
			if line != "":
				newfile.write(link_dict[line] + "\n")

	file1.close()
	file2.close()
	sortedfile.close()
	newfile.close()

	print("finished writing to new file.")

	return

if __name__ == "__main__":
    main()
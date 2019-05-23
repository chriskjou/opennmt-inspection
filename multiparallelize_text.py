import sys
import subprocess
import os
from itertools import zip_longest as zip
from tqdm import tqdm

def find_common_sentences(srclangABfile, tgtlangABfile, sortedsrclangAfile, newfilename):
	link_dict = {}

	# open files
	srclangA = open(srclangABfile)
	tgtlangB = open(tgtlangABfile)
	sortedsrclangA = open(sortedsrclangAfile)
	newfile = open(newfilename,"w") 

	with srclangA as file1, tgtlangB as file2: 
		for x, y in tqdm(zip(file1, file2)):
			x = x.strip()
			y = y.strip()
			link_dict.setdefault(x, 0)
			link_dict[x] = y

	with sortedsrclangA as sortedfile:
		for line in tqdm(sortedfile):
			line = line.strip()
			if line != "":
				newfile.write(link_dict[line] + "\n")

	file1.close()
	file2.close()
	sortedfile.close()
	newfile.close()
	return

def clean_all_english(srcfile, newfilename):
	src = open(srcfile)
	newfile = open(newfilename,"w") 

	with src as file:
		for line in tqdm(file):
			line = line.strip()
			if line != "":
				newfile.write(line + "\n")
	return

def main():
	print("usage: python multiparallelize_text.py -srclangAB.txt -tgtlangAB.txt -srclangAC.txt -tgtlangAC.txt (as many src languages texts of src-tgt pairs)")
	print("RETURNS: srclangA.txt tgtlangB.txt tgtlangC.txt\n")

	texts = []
	targets = []

	# get texts
	for index in range(1, len(sys.argv)):
		if index % 2:
			texts.append(sys.argv[index])
		else:
			targets.append(sys.argv[index])

	print("SOURCES: " + str(texts))
	print("TARGETS: " + str(targets))

	if not os.path.exists('../multiparallelize/'):
		os.makedirs('../multiparallelize/')

	files_to_remove = []
	unique_sorted = []
	annotated = []

	# get unique sorted of texts
	print("\ngetting unique sorted texts...\n")
	for txt in texts:
		file_name = txt.split("/")[-1].split(".")[0]
		sort_cmd = "sort " + str(txt) + " > ../multiparallelize/sorted-" + str(file_name) + ".txt"
		files_to_remove.append("../multiparallelize/sorted-" + str(file_name) + ".txt")
		os.system(sort_cmd)
		uniq_cmd = "uniq -u ../multiparallelize/sorted-" + str(file_name) + ".txt > ../multiparallelize/"+ str(file_name) + "-unique-sorted.txt"
		os.system(uniq_cmd)
		files_to_remove.append("../multiparallelize/"+ str(file_name) + "-unique-sorted.txt")
		unique_sorted.append("../multiparallelize/"+ str(file_name) + "-unique-sorted.txt")

	# merge to get unique of all
	print("merging to get common unique...\n")
	for counter in range(1, len(texts)):
		file_name_counter = texts[counter].split("/")[-1].split(".")[0]
		file_name_counter_prev = texts[counter-1].split("/")[-1].split(".")[0]
		if counter == 1:
			cmd = "comm -12 ../multiparallelize/" + str(file_name_counter_prev) + "-unique-sorted.txt ../multiparallelize/" + str(file_name_counter) + "-unique-sorted.txt > ../multiparallelize/merge" + str(counter) + ".txt"
			files_to_remove.append("../multiparallelize/merge" + str(counter) + ".txt")
		elif counter == len(texts) - 1:
			cmd = "comm -12 ../multiparallelize/" + str(file_name_counter_prev) + "-unique-sorted.txt ../multiparallelize/merge" + str(counter-1) + ".txt > ../multiparallelize/common.txt"
			files_to_remove.append("../multiparallelize/common.txt")
		else:
			cmd = "comm -12 ../multiparallelize/" + str(file_name_counter_prev) + "-unique-sorted.txt ../multiparallelize/merge" + str(counter-1) + ".txt > ../multiparallelize/merge" + str(counter) + ".txt"
			files_to_remove.append("../multiparallelize/merge" + str(counter) + ".txt")
		os.system(cmd)

	# annotate languages
	print("annotating languages...\n")
	for txt in texts:
		file_name = txt.split("/")[-1].split(".")[0]
		cmd = "./annotate.sh ../multiparallelize/" + str(file_name) + "-unique-sorted.txt ../multiparallelize/merge" + str(len(texts)-1) + ".txt > ../multiparallelize/" + str(file_name) + "-annotated.txt"
		os.system(cmd)
		files_to_remove.append("../multiparallelize/" + str(file_name) + "-annotated.txt")
		annotated.append("../multiparallelize/" + str(file_name) + "-annotated.txt")

	# get corresponding target language based on common sentences
	new_srcfile_name = "../multiparallelize/parallelized-src.txt"
	clean_all_english(annotated[0], new_srcfile_name)

	for index in range(len(texts)):
		file_name = texts[index].split("/")[-1].split(".")[0]
		new_tgtfile_name = "../multiparallelize/parallelized-target-" + str(file_name) + ".txt"
		find_common_sentences(texts[index], targets[index], "../multiparallelize/parallelized-src.txt", new_tgtfile_name)
	
	print("\ndeleting files made in progress...\n")
	for file in files_to_remove:
		print(file)
		cmd = "rm " + str(file)
		os.system(cmd)
	print("done.")
	return

if __name__ == "__main__":
    main()
import sys
import subprocess

def main():
	print("usage: python multiparallelize.py -srclangA.txt -srclangB.txt (as many src languages texts of src-tgt pairs)")
	print("RETURNS: annotated languages of 'srclangX-annotated.txt' and txt of common sentences of 'common.txt'")
	print("PASS to find_common_sentences.py: original_srclang, original_tgtlang, srclangX-annotated -> desired file with corresponding tgt sentences")
		
	texts = []

	# get texts
	for arg in sys.argv:  
		texts.append(arg)

	# get unique sorted of texts
	for txt in texts:
		cmd = "uniq -u <(sort " + str(txt) + " > "+ str(txt) + "-unique-sorted.txt"
		process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
		output, error = process.communicate()

	# merge to get unique of all
	for counter in range(1, len(texts)):
		if counter == 1:
			cmd = "comm -12 " + str(texts[counter-1]) + "-unique-sorted.txt" + str(texts[counter]) + "-unique-sorted.txt > merge" + str(counter) + ".txt"
		elif counter == len(texts) - 1:
			cmd = "comm -12 " + str(texts[counter-1]) + "-unique-sorted.txt merge" + str(counter-1) + ".txt > common.txt"
		else:
			cmd = "comm -12 " + str(texts[counter-1]) + "-unique-sorted.txt merge" + str(counter-1) + ".txt > merge" + str(counter) + ".txt"
		process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
		output, error = process.communicate()

	# annotate languages
	for txt in texts:
		cmd = "./annotate.sh " + str(txt) + "-unique-sorted.txt" + " merge" + str(len(texts)-1) + ".txt" > str(txt) + "-annotated.txt"

	return

if __name__ == "__main__":
    main()
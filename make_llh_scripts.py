import argparse 
# import os

def save_script(args):
	# if not os.path.exists(str(args.path) + 'calculate_llh_significance/'):
	# 	os.makedirs(str(args.path) + 'calculate_llh_significance/')

	folder_name = str(args.path) + "calculate_llh_significance"
	script_to_open = str(folder_name) + ".sh"

	# make master script
	with open(script_to_open, "w") as rsh:
		rsh.write('''\
#!/bin/bash
for i in `seq 0 99`; do
  sbatch "calculate_llh_significance_batch""$i""of{}.sh" -H
done
'''.format( 
		args.total_batches
	)
)

	# break into batches
	for batch_num in range(args.total_batches):
		job_id = "calculate_llh_significance_batch{}of{}".format(
			batch_num,
			args.total_batches
		)

		fname = str(job_id) + '.sh'

		with open(fname, 'w') as rsh:
			mem = "2000"
			timelimit = "0-3:00"
			rsh.write('''\
#!/bin/bash
#SBATCH -J {0}  								# Job name
#SBATCH -p serial_requeue 						# partition (queue)
#SBATCH --mem {1} 								# memory pool for all cores
#SBATCH -t {2}									# time (D-HH:MM)
#SBATCH -o {6}outpt_{0}.txt 			# File that STDOUT writes to
#SBATCH -e {6}err_{0}.txt				# File that STDERR writes to
#SBATCH --mail-type=ALL
#SBATCH --mail-user={5}

module load matlab

matlab -nodisplay -nosplash -nojvm -r $'calculate_llh_significance_across_layers({3}, {4})'

'''.format(
		job_id, 
		mem, 
		timelimit,
		batch_num, 
		args.total_batches,
		args.email,
		args.slurm_path
	)
)

def main():
	parser = argparse.ArgumentParser("make scripts for Odyssey cluster")
	parser.add_argument("-total_batches", "--total_batches", help="Total number of batches to run", type=int, default=100)
	parser.add_argument("-path", "--path", help="path to mfit directory", type=str, default="../mfit/", required=True)
	parser.add_argument("-email", "--email", help="email associated with Odyssey cluster", type=str, default="ckjou@college.harvard.edu", required=True)
	parser.add_argument("-slurm_path", "--slurm_path", help="path for output/error files (default on home directory)", type=str, default="")
	args = parser.parse_args()

	save_script(args)
	print("done.")

if __name__ == "__main__":
    main()
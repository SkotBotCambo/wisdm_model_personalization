wisdm_v1_url = "http://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz"
wisdm_v2_url = "http://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_at_latest.tar.gz"

dataset_path = "./datasets/"

wisdm_v1_tar_path = dataset_path+"wisdm_v1.tar"
wisdm_v2_tar_path = dataset_path+"wisdm_v2.tar"

wisdm_v1_dataset_path = dataset_path + "WISDM_v1/"
wisdm_v2_dataset_path = dataset_path + "WISDM_v2/"

import os
try:
	import urllib.request
except ImportError as ie:
	print("Try Using Python3 instead.")
	raise ie
import tarfile
import shutil

def init_dirs():
	try:
		os.mkdir(dataset_path)
		os.mkdir("./results/")
	except FileExistsError as fee:
		pass

def download_wisdm_to_path():
	print("downloading WISDM v1.1 from %s" % wisdm_v1_url)
	urllib.request.urlretrieve(wisdm_v1_url, filename=wisdm_v1_tar_path)

	print("downloading WISDM v2.0 from %s" % wisdm_v2_url)
	urllib.request.urlretrieve(wisdm_v2_url, filename=wisdm_v2_tar_path)

	print("Extracting data...")
	wisdm_tar = tarfile.open(wisdm_v1_tar_path)
	wisdm_tar.extractall(path=wisdm_v1_dataset_path)
	wisdm_tar.close()

	wisdm_tar = tarfile.open(wisdm_v2_tar_path)
	wisdm_tar.extractall(path=wisdm_v2_dataset_path)
	wisdm_tar.close()

	print("Fixing data paths...")
	wisdm_v1_files = os.listdir(wisdm_v1_dataset_path+"WISDM_ar_v1.1")

	for file in wisdm_v1_files:
		shutil.move(wisdm_v1_dataset_path+"WISDM_ar_v1.1/"+file, wisdm_v1_dataset_path+file)

	wisdm_v2_files = os.listdir(wisdm_v2_dataset_path+"home/share/data/public_sets/WISDM_at_v2.0/")
	for file in wisdm_v2_files:
		shutil.move(wisdm_v2_dataset_path+"home/share/data/public_sets/WISDM_at_v2.0/"+file, wisdm_v2_dataset_path+file)
	shutil.rmtree(wisdm_v2_dataset_path+"home/")

def create_fixed_datasets():
	# dataset v1.1
	with open(wisdm_v1_dataset_path+"WISDM_ar_v1.1_transformed.arff", "r") as fIn:
		old_lines = fIn.readlines()
		new_lines = []
		for ind, line in enumerate(old_lines):
			if ind is 3:
				bracket_index = line.find("{")
				new_line = line[:bracket_index]
				new_line += line[bracket_index:].replace('"', '')
			elif ind is 47:
				bracket_index = line.find("{")
				new_line = line[:bracket_index]
				new_line += " "+line[bracket_index:].replace('"', '')
			else:
				new_line = line
			new_lines.append(new_line)

	with open(wisdm_v1_dataset_path+"WISDM_ar_v1.1_transformed_FIXED.arff", "w") as fOut:
		for line in new_lines:
			fOut.write(line)

	with open(wisdm_v2_dataset_path+"WISDM_at_v2.0_transformed.arff", "r") as fIn:
		old_lines = fIn.readlines()
		new_lines = []
		for ind, line in enumerate(old_lines):
			if ind is 2:
				brack_index = line.find("{")
				new_line = line[:bracket_index]
				new_line += line[bracket_index:].replace('"', '')
			elif ind is 46:
				bracket_index = line.find("{")
				new_line = line[:bracket_index]
				new_line += " "+line[bracket_index:].replace('"', '')
			else:
				new_line = line
			new_lines.append(new_line)

	with open(wisdm_v2_dataset_path+"WISDM_at_v2.0_transformed_FIXED.arff", "w") as fOut:
		for line in new_lines:
			fOut.write(line)

if __name__ == "__main__":
	print("Initializing directories...")
	init_dirs()

	print("Downloading wisdm datasets...")
	download_wisdm_to_path()

	print("Fixing datasets...")
	create_fixed_datasets()

	try:
		from wisdm import wisdm
		wisdm.set_data()
		print("WISDM datasets downloaded and fixed for analysis with python.")
	except:
		print("WISDM datasets appear to have not downloaded correctly")
		raise



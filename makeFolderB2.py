# Lines marked with *** need to be changed for A2,B1,B2 versions
import os, shutil, numpy
exercise = "B2"														#***
source_folder = "/Users/faymita29/Desktop/AMLS/AMLSGitHub/givenData/celeba"										#***
target_folder = "/Users/faymita29/Desktop/AMLS/AMLS_19-20_SN16057922/Datasets"
if not(os.path.isdir(target_folder)): os.mkdir(target_folder)
os.mkdir(target_folder+"/"+exercise)
for use in ["train","validate","test"]:
	os.mkdir(target_folder+"/"+exercise+"/"+use)
	for eyecolour in ["brown","blue","green","grey","black"]:
		os.mkdir(target_folder + "/"+exercise+"/"+use+"/"+eyecolour)
label_file = open(source_folder+"/labels.csv",'r');
label_lines = label_file.readlines()
label_file.close()
file_list = {"brown":[],"blue":[],"green":[],"grey":[],"black":[]}									
for line in range(1,len(label_lines)):
	line_as_list = label_lines[line].split()
	file_name = line_as_list[3]
	eyecolour = line_as_list[1]
	if eyecolour == "0":												#***
		file_list["brown"].append(file_name)						#***
    if eyecolour == "1":												
		file_list["blue"].append(file_name)
    if eyecolour == "2":												
		file_list["green"].append(file_name)
    if eyecolour == "3":												
		file_list["grey"].append(file_name)
	else:															#***
		file_list["black"].append(file_name)							#***

for eyecolour in ["brown","blue","green","grey","black"]:
	numpy.random.shuffle(file_list[eyecolour])
for eyecolour in ["brown","blue","green","grey","black"]:		
	training_size = int(0.6*len(file_list[eyecolour]))
	validation_size = int(0.2*len(file_list[eyecolour]))
	testing_size = len(file_list[eyecolour])-training_size-validation_size
	files_to_copy = {}
	files_to_copy["train"] = file_list[eyecolour][0:training_size]
	files_to_copy["validate"] = file_list[eyecolour][training_size:training_size+validation_size]
	files_to_copy["test"] = file_list[eyecolour][training_size+validation_size:]
	for use in ["train","validate","test"]:
		for file in files_to_copy[use]:
			source_file = source_folder+"/img/"+file
			target_file = target_folder+"/"+exercise+"/"+use+"/"+eyecolour+"/"+file
			shutil.copyfile(source_file,target_file)
	


	

# Lines marked with *** need to be changed for A2,B1,B2 versions
import os, shutil, numpy
exercise = "A1"														#***
source_folder = "/Users/faymita29/Desktop/AMLS/AMLSGitHub/givenData/celeba"										#***
target_folder = "/Users/faymita29/Desktop/AMLS/AMLS_19-20_SN16057922/Datasets"
if not(os.path.isdir(target_folder)): os.mkdir(target_folder)
os.mkdir(target_folder+"/"+exercise)
for use in ["train","validate","test"]:
	os.mkdir(target_folder+"/"+exercise+"/"+use)
	for gender in ["female","male"]:								#***
		os.mkdir(target_folder + "/"+exercise+"/"+use+"/"+gender)
label_file = open(source_folder+"/labels.csv",'r');
label_lines = label_file.readlines()
label_file.close()
file_list = {"female":[],"male":[]}									
for line in range(1,len(label_lines)):
	line_as_list = label_lines[line].split()
	file_name = line_as_list[1]
	gender = line_as_list[2]
	if gender == "-1":												#***
		file_list["female"].append(file_name)						#***
	else:															#***
		file_list["male"].append(file_name)							#***
for gender in ["female","male"]:									#***
	numpy.random.shuffle(file_list[gender])
for gender in ["female","male"]:									#***
	training_size = int(0.6*len(file_list[gender]))
	validation_size = int(0.2*len(file_list[gender]))
	testing_size = len(file_list[gender])-training_size-validation_size
	files_to_copy = {}
	files_to_copy["train"] = file_list[gender][0:training_size]
	files_to_copy["validate"] = file_list[gender][training_size:training_size+validation_size]
	files_to_copy["test"] = file_list[gender][training_size+validation_size:]
	for use in ["train","validate","test"]:
		for file in files_to_copy[use]:
			source_file = source_folder+"/img/"+file
			target_file = target_folder+"/"+exercise+"/"+use+"/"+gender+"/"+file
			shutil.copyfile(source_file,target_file)
	


	

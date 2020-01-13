# Lines marked with *** need to be changed for A2,B1,B2 versions
import os, shutil, numpy
exercise = "B1"														#***
source_folder = "/Users/faymita29/Desktop/AMLS/AMLS_19-20_SN16015140/givenData/cartoon_set"	
target_folder = "/Users/faymita29/Desktop/AMLS/AMLS_19-20_SN16057922/Datasets"
if not(os.path.isdir(target_folder)): os.mkdir(target_folder)
os.mkdir(target_folder+"/"+exercise)
for use in ["train","validate","test"]:
	os.mkdir(target_folder+"/"+exercise+"/"+use)
	for faceshape in ["shape0","shape1","shape2","shape3","shape4"]:
		os.mkdir(target_folder + "/"+exercise+"/"+use+"/"+faceshape)
label_file = open(source_folder+"/labels.csv",'r');
label_lines = label_file.readlines()
label_file.close()
file_list = {"shape0":[],"shape1":[],"shape2":[],"shape3":[],"shape4":[]}									
for line in range(1,len(label_lines)):
	line_as_list = label_lines[line].split()
	file_name = line_as_list[3]
	faceshape = line_as_list[2]
	if faceshape == "0":
        file_list["shape0"].append(file_name)
        elif faceshape == "1":
            file_list["shape1"].append(file_name)
            elif faceshape == "2":
                file_list["shape2"].append(file_name)
                elif faceshape == "3":	
                    file_list["shape3"].append(file_name)
    else:
        file_list["shape4"].append(file_name)


for faceshape in ["shape0","shape1","shape2","shape3","shape4"]:
	numpy.random.shuffle(file_list[faceshape])
for faceshape in ["shape0","shape1","shape2","shape3","shape4"]:		
	training_size = int(0.6*len(file_list[faceshape]))
	validation_size = int(0.2*len(file_list[faceshape]))
	testing_size = len(file_list[faceshape])-training_size-validation_size
	files_to_copy = {}
	files_to_copy["train"] = file_list[faceshape][0:training_size]
	files_to_copy["validate"] = file_list[faceshape][training_size:training_size+validation_size]
	files_to_copy["test"] = file_list[faceshape][training_size+validation_size:]
	for use in ["train","validate","test"]:
		for file in files_to_copy[use]:
			source_file = source_folder+"/img/"+file
			target_file = target_folder+"/"+exercise+"/"+use+"/"+faceshape+"/"+file
			shutil.copyfile(source_file,target_file)
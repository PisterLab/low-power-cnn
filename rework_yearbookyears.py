# Script to change the organization and naming convention of the yearbook folders to make the dataset for identifying year rather than M/F

import os
from shutil import copyfile

home = os.path.expanduser("~")
print(home)

# os.join()

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(file_path):
        print('New Directory!')
        os.makedirs(file_path)


path_orig_M = '/Users/nato/Datasets-Models/yearbook_MF/M'
path_orig_F = '/Users/nato/Datasets-Models/yearbook_MF/F'

path_new = '/Users/nato/Datasets-Models/yearbook_year'

files = os.listdir(path_orig_F)

for f in files:
    year = f[0:3]
    # print(year)
    decade = year+'0'
    # print(decade)
    # print(f)
    save_loc = path_new+'/'+decade + '/'
    save_file = save_loc + f
    load_loc = path_orig_F + '/'+f
    # print(load_loc)
    # print(save_loc)
    ensure_dir(save_loc)
    copyfile(load_loc, save_file)
    # print('---')

    # print(f)

'''
This is to generated training data for chinese ocr.
Source data: https://github.com/chenkenanalytic/handwritting_data_all
'''
import glob, os, shutil
import random
import numpy as np
from PIL import Image

folder_path = '/home/rogersun/handwritting_data_all/cleaned_data' # location of source data
new_folder_name = 'chineseocr_training_dataset' # output folder
concatenate_direction = 1 # 0: verticle, 1: horizontal
max_len = 10 # max length of the words

# Ground Truth File
train_path = 'train_own.txt'
test_path = 'test_own.txt'
f_train = open(train_path,'w')
f_test = open(test_path,'w')


# make destination folder
destination_path = os.path.join(folder_path,new_folder_name)
if not os.path.isdir(destination_path):
	os.mkdir(destination_path)

# get file list
os.chdir(folder_path)
filepath_list = glob.glob('*/*.png') # label/word_randomnum.png, e.g. "1/一_10.png"

counter = 0 #init

while len(filepath_list) > 0:
    counter += 1

    filepath_list_length = len(filepath_list)
    word_len = random.sample(range(max_len-1),1)[0]+1 # decide word length
    # print('word length' + str(word_len)+'\n')
    # shuffle, but i don't know why my random.shuffle didn't work
    filepath_random_index = random.sample(range(filepath_list_length),filepath_list_length)
    filepath_list_shuffle = [x for y, x in sorted(zip(filepath_random_index, filepath_list))]

    # Word candidate
    if word_len >= filepath_list_length:
        filepath_list_candidate = filepath_list_shuffle
        filepath_list = []
    else:
        filepath_list_candidate = filepath_list_shuffle[:word_len]
        filepath_list = filepath_list_shuffle[word_len:]

    # concatenate image
    img_list = [np.array(Image.open(os.path.join(folder_path,file))) for file in filepath_list_candidate]
    img = np.concatenate(img_list,axis = concatenate_direction)
    img = Image.fromarray(img)
    # print('word composeed with following file: \n')
    # print(filepath_list_candidate)
    img_label = [str(int(file.split('/')[0])-1) for file in filepath_list_candidate]
    img_word = [str(file.split('/')[1].split('_')[0]) for file in filepath_list_candidate]
    img_name = ''.join(img_word)+'_'+'_'.join(img_label)+'.png'
    
    img.save(os.path.join(destination_path,img_name))

    # 2/3 of data be training data, 1/3 be validation
    if (counter%3)==0:
        f_test.write(img_name + ' ' + ' '.join(img_label)+'\n')
    else:
        f_train.write(img_name + ' ' + ' '.join(img_label)+'\n')

f_train.close()
f_test.close()
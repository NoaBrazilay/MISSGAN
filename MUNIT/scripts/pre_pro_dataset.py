import io
import os
import imageio
train_dataset_raw_path = '..\datasets\edges2handbags\\train'
train_folder_a = '..\datasets\edges2handbags\\trainA'
train_folder_b = '..\datasets\edges2handbags\\trainB'
os.makedirs(train_folder_a, exist_ok=True)
os.makedirs(train_folder_b, exist_ok=True)

test_dataset_raw_path = '..\datasets\edges2handbags\\val'
test_folder_a = '..\datasets\edges2handbags\\testA'
test_folder_b = '..\datasets\edges2handbags\\testB'
os.makedirs(test_folder_a, exist_ok=True)
os.makedirs(test_folder_b, exist_ok=True)

def load_images_from_folder(folder, folder_a, folder_b):

    for filename in os.listdir(folder):

        img = imageio.imread(os.path.join(folder,filename))
        if img is not None:
            # h,w = img.shape[0], img.shape[1]
            imgA = img[:, :256]
            imgB = img[:, 256:]
            imageio.imwrite(folder_a+'\\' + filename, imgA)
            imageio.imwrite(folder_b+'\\' + filename, imgB)



load_images_from_folder(train_dataset_raw_path, train_folder_a, train_folder_b)
load_images_from_folder(test_dataset_raw_path, test_folder_a, test_folder_b)
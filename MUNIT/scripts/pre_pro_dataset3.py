import io
import os
import imageio
# import cv2
from PIL import Image

# illustrator = 'Axel Scheffler'
# illustrator = 'David Mckee'
# illustrator = 'Kevin Henkes'
# illustrator = 'Korky Paul'
# illustrator = 'Marc Brown'
# illustrator = 'Patricia Polacco'
#illustrator = 'Stephen Cartwright'


train_dataset_raw_path = '..\..\Ganilla\datasets\illustrations\\'

train_folder_a = '..\datasets\illustrations2landscapes\\' + 'total' + '\\trainA'
os.makedirs(train_folder_a, exist_ok=True)
def load_images_from_folder(parent_folder, folder_a):
    dest_img_name = 0
    for folder in os.listdir(parent_folder):
        illustrator_folder = parent_folder+'\\' + folder
        for book in os.listdir(illustrator_folder):
            book_folder = illustrator_folder+'\\' + book
            for filename in os.listdir(book_folder):

                # img = imageio.imread(os.path.join(folder,filename))
                img = Image.open(os.path.join(book_folder, filename))
                if img is not None:
                    # h,w = img.shape[0], img.shape[1]
                    imgA = img.resize((256, 256), Image.ANTIALIAS)

                    # imageio.imwrite(folder_a+'\\' + str(dest_img_name), imgA)
                    imgA.save(folder_a+'\\' + str(dest_img_name) + '.png')
                    dest_img_name += 1



load_images_from_folder(train_dataset_raw_path, train_folder_a)

import numpy as np
import cv2
import os

def read_dataset(dataset_path):
    file_paths = []
    scores = []
    with open(dataset_path, mode='r') as f:
        lines = f.readlines()
        print('Text size:',len(lines))

        for i, line in enumerate(lines):
            token = line.split()
            score = np.array(token[1:11], dtype='float32')
            file_path = token[0]
            
            if os.path.exists(file_path):
                file_paths.append(file_path)
                scores.append(score)
            else:
                print('File not found:', file_path)
                
            count = len(lines) // 20
            if i % count == 0 and i != 0:
                print('Loaded %d percent of the dataset' % (i / len(lines) * 100))
                
    file_paths = np.array(file_paths)
    scores = np.array(scores, dtype='float32')

    return file_paths, scores

def randomCrop(img, width, height):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    return img

def data_generator(file_paths, scores, batchsize, image_size, shuffle=True):
    while True:
        dataset = list(zip(file_paths, scores))
        np.random.shuffle(dataset)
        file_paths, scores = zip(*dataset)
        batch_size = batchsize
        for i in range(0, len(file_paths) - batch_size, batch_size):
            #print('train: i-->',i)
            batch_x_file = file_paths[i:i+batch_size]
            batch_y = np.array(scores[i:i+batch_size])
            batch_x = np.zeros((batch_size, image_size[0], image_size[1], 3),dtype = np.float32)    
            for index, file in enumerate(batch_x_file):
                img = cv2.imread(file)
                img_scaled = cv2.resize(img,(256,256), interpolation=cv2.INTER_CUBIC)
                is_flip = np.random.randint(2)
                if is_flip == 1:
                    img_scaled = cv2.flip( img_scaled, 0 )
                img_cropped = randomCrop(img_scaled,image_size[0], image_size[1])
                batch_x[index] = img_cropped/255.0 # (img - 127.5)/127.5
                
            yield batch_x, batch_y
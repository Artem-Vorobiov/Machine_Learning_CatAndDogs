#	GPU (сокр. от graphics processing unit, графический процессор) – это отдельный процессор расположенный на видеокарте, 
#	который выполняет обработку 2D или 3D графики. Имея процессор на видеокарте, компьютерный процессор освобождается 
#	от лишней работы и может выполнять все другие важные задачи быстрее. Особенностью графического процессора (GPU), 
#	является то, что он максимально нацелен на увеличение скорости расчета именно графической информации (текстур и объектов).
#	 Благодаря своей архитектуре такие процессоры намного эффективнее обрабатывают графическую информацию, 
#	нежели типичный центральный процессор компьютера

import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion

TRAIN_DIR = 'train'
TEST_DIR = 'test'
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic') # just so we remember which saved model is which, sizes must match
#	Подключаем папку с исходными картинками
checking1  = os.listdir(TRAIN_DIR)
# print('\n GO \n\n', MODEL_NAME)
# print('\n checking1 \n\n', checking1)			#	['cat.6.jpg', 'dog.24.jpg', 'dog.18.jpg', 'cat.19.jpg' ...  и так далее]
# print('\n checking1 \n\n', type(checking1))		#	 <class 'list'>

#	Now, our first order of business is to convert the images and labels to array information that we can pass through our network.
#	 To do this, we'll need a helper function to convert the image name to an array
#			ОБЪЯСНЯЛОЧКА-ПРОВЕРОЧКА
# a  = 'OneTwoThree.22.After'
# ch = a.split('.')[-3]
# print('\n 	GO \n\n', ch)

def label_img(img):
    word_label = img.split('.')[-3]
    # print('\n word_label	HERE \n\n', word_label)
    # print('\n img	HERE \n\n', img)
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'cat': return [1,0]
    #                             [no cat, very doggo]
    elif word_label == 'dog': return [0,1]

#	Anyway, the below function converts the data for us into array data of the image and its label.
def create_train_data():
	# 	ШАГ - 2
    training_data = []	# Создали Лист Пустой
    # print('\n os.listdir(TRAIN_DIR) \n\n', os.listdir(TRAIN_DIR))		<class 'list'>

    for img in tqdm(os.listdir(TRAIN_DIR)):
    	# У нас есть Лист внутри которого все картинки, они же элементы Листа(тип строка).
    	# Перебираем по одному каждый элемент Листа
    	# Каждый элемент отправляется в функцию def label_img(img), где делится на три части, разеделенные точкой
    	# Берем нужную нам часть, если это собака присваиваем [0,1], если кошка [1,0]
    	# Выход def label_img(img) - это лист типа label = [1,0]  <class 'list'>

        label = label_img(img)
        # print('\n label \n\n', label)				#	label = [1,0]
        # print('\n label \n\n', type(label))		#	<class 'list'>

        path = os.path.join(TRAIN_DIR,img)
        # print('\n path \n\n', path)				#	train/cat.8.jpg
        # print('\n path \n\n', type(path))			#	<class 'str'>

        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        # print('\n img \n\n', img)					#	Массив с цифрами от 0 до 255, каждая цифра это пиксель
        # print('\n TYPE \n\n', type(img))			#	<class 'numpy.ndarray'>

        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        # print('\n img \n\n', img)					#	Массив с цифрами от 0 до 255, каждая цифра это пиксель
        # print('\n TYPE \n\n', type(img))			#	<class 'numpy.ndarray'>

#	ЭТО Лист внутри листа ряд элементов, эти элементы так же Листы, в каждом подЛисте два элемента
# 	Первый подПодЭлемент = Лист с даннми о картинке(0-255), второй подПодЭлемент Лист содержаший инфу Собака или Кошка
#	training_data = [[[img_1],[0,1]],[[img_2],[1,0]]]
        training_data.append([np.array(img),np.array(label)])
        # print('\n training_data \n\n', training_data)


    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
    	# print('\n testing_data \n\n', os.listdir(TEST_DIR))	#	List['20.jpg', '36.jpg']
    	# print('\n TYPE \n\n', type(os.listdir(TEST_DIR)))		#	<class 'list'>

    	path = os.path.join(TEST_DIR,img)
    	# print('\n path \n\n', path)							#	test/44.jpg
    	# print('\n TYPE \n\n', type(path))						#	class 'str'>

    	img_num = img.split('.')[0]
    	img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    	img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    	testing_data.append([np.array(img), img_num])
#	ЭТО Лист внутри листа ряд элементов, эти элементы так же Листы, в каждом подЛисте два элемента
# 	Первый подПодЭлемент = Лист с даннми о картинке(0-255), второй подПодЭлемент Строка соответвует номеру картинки
#	training_data = [[[img_1],'3',[[img_2],'7']

        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

# 	ШАГ - 1
train_data = create_train_data()
# print('\n training_data \n\n', train_data)
# print('\n TYPE \n\n', type(train_data))
test_data = process_test_data()
# print('\n test_data \n\n', test_data)
# print('\n TYPE \n\n', type(test_data))

# If you have already created the dataset:
#train_data = np.load('train_data.npy')
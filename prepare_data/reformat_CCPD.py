import os
import cv2
import random


def annotation_from_name(file_name):
    file_name = file_name[:-4]
    name_split = file_name.split('-')
    location = name_split[2]
    location = location.split('_')
    left_top = location[0].split('&')
    right_bottom = location[1].split('&')
    x1 = int(left_top[0])
    y1 = int(left_top[1])
    x2 = int(right_bottom[0])
    y2 = int(right_bottom[1])

    x1y1 = name_split[3].split('_')[0].split('&')
    x2y2 = name_split[3].split('_')[1].split('&')
    x3y3 = name_split[3].split('_')[2].split('&')
    x4y4 = name_split[3].split('_')[3].split('&')

    return (x1, y1, x2, y2, int(x1y1[0]), int(x1y1[1]), int(x2y2[0]), int(x2y2[1]), int(x3y3[0]), int(x3y3[1]), int(x4y4[0]), int(x4y4[1]))


def generate_data_list(ccpd_path='/home/hylink/eclipse-workspace/CCPD2019/'):
    
    image_roots = [ccpd_path + 'ccpd_base',
                   ccpd_path + 'ccpd_blur',
                   ccpd_path + 'ccpd_challenge',
                   ccpd_path + 'ccpd_db',
                   ccpd_path + 'ccpd_fn',
                   ccpd_path + 'ccpd_rotate',
                   ccpd_path + 'ccpd_tilt',
                   ccpd_path + 'ccpd_weather']

    train_list_file_path = './data_folder/data_list_CCPD_train.txt'
    test_list_file_path = './data_folder/data_list_CCPD_test.txt'
    if not os.path.exists(os.path.dirname(train_list_file_path)):
        os.makedirs(os.path.dirname(train_list_file_path))
    fout_train = open(train_list_file_path, 'w')
    fout_test = open(test_list_file_path, 'w')

    train_proportion = 1
    train_counter = 0
    test_counter = 0
    for root in image_roots:
        file_name_list = [name for name in os.listdir(root) if name.endswith('.jpg')]
        random.shuffle(file_name_list)

        file_name_list_train = file_name_list[:int(len(file_name_list) * train_proportion)]
        file_name_list_test = file_name_list[int(len(file_name_list) * train_proportion):]

        for file_name in file_name_list_train:
            location_annotation = annotation_from_name(file_name)
            line = os.path.join(root, file_name) + ',1,1,' + str(location_annotation[0]) + ',' + str(location_annotation[1]) + ',' + str(location_annotation[2]) + ',' + str(location_annotation[3])\
             +',' + str(location_annotation[4]) + ',' + str(location_annotation[5]) + ',' + str(location_annotation[6]) + ',' + str(location_annotation[7]) + ',' + str(location_annotation[8])\
             +',' + str(location_annotation[9]) + ',' + str(location_annotation[10]) + ',' + str(location_annotation[11])
            
            fout_train.write(line + '\n')
            train_counter += 1
            print(train_counter)

        for file_name in file_name_list_test:
            location_annotation = annotation_from_name(file_name)
            line = os.path.join(root, file_name) + ',1,1,' + str(location_annotation[0]) + ',' + str(location_annotation[1]) + ',' + str(location_annotation[2]) + ',' + str(location_annotation[3])\
             +',' + str(location_annotation[4]) + ',' + str(location_annotation[5]) + ',' + str(location_annotation[6]) + ',' + str(location_annotation[7]) + ',' + str(location_annotation[8])\
             +',' + str(location_annotation[9]) + ',' + str(location_annotation[10]) + ',' + str(location_annotation[11])            
            
            fout_test.write(line + '\n')
            test_counter += 1
            print(test_counter)

    fout_train.close()
    fout_test.close()


def show_image():
    list_file_path = './data_folder/data_list_CCPD_train.txt'

    fin = open(list_file_path, 'r')
    lines = fin.readlines()
    fin.close()

    random.shuffle(lines)
    for line in lines:
        line = line.strip('\n').split(',')

        im = cv2.imread(line[0])

        bboxes = []
        landmarks = []
        num_bboxes = int(line[2])
        for i in range(num_bboxes):
            xmin = int(line[3 + i * 4])
            ymin = int(line[4 + i * 4])
            x2min = int(line[5 + i * 4])
            y2min = int(line[6 + i * 4])
            bboxes.append((xmin, ymin, x2min + 1, y2min + 1))
            landmarks.append([int(line[7]), int(line[8]), int(line[9]), int(line[10]), int(line[11]), int(line[12]), int(line[13]), int(line[14])])

        for bbox in bboxes:
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)

        for b in landmarks:
            cv2.circle(im, (b[0], b[1]), 1, (0, 0, 255), 10)
            cv2.circle(im, (b[2], b[3]), 1, (0, 255, 255), 15)
            # cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(im, (b[4], b[5]), 1, (0, 255, 0), 20)
            cv2.circle(im, (b[6], b[7]), 1, (255, 0, 0), 30) 

        cv2.imshow('im', im)
        cv2.waitKey()


def dataset_statistics():
    list_file_path = './data_folder/data_list_CCPD_train.txt'

    fin = open(list_file_path, 'r')
    lines = fin.readlines()
    fin.close()

    bin_size = 8
    longer_bin_dict = {}
    shorter_bin_dict = {}
    counter_pos = 0
    counter_neg = 0
    for line in lines:
        line = line.strip('\n').split(',')
        if line[1] == '0':
            counter_neg += 1
            continue
        else:
            counter_pos += 1
        num_bboxes = int(line[2])
        for i in range(num_bboxes):
            xmin = int(line[3 + i * 4])
            ymin = int(line[4 + i * 4])
            x2min = int(line[5 + i * 4])
            y2min = int(line[6 + i * 4])
            
            width = x2min - xmin
            height = y2min - ymin

            longer_side = max(width, height)
            shorter_side = min(width, height)

            key = int(longer_side / bin_size)
            if key in longer_bin_dict:
                longer_bin_dict[key] += 1
            else:
                longer_bin_dict[key] = 1

            key = int(shorter_side / bin_size)
            if key in shorter_bin_dict:
                shorter_bin_dict[key] += 1
            else:
                shorter_bin_dict[key] = 1

    total_pedestrian = 0
    print('shorter side based statistics:')
    shorter_bin_dict_key_list = sorted(shorter_bin_dict)
    for k in shorter_bin_dict_key_list:
        v = shorter_bin_dict[k]
        total_pedestrian += v
        print('[%d-%d): %d' % (k * bin_size, k * bin_size + bin_size, v))

    print('longer side based statistics:')
    longer_bin_dict_key_list = sorted(longer_bin_dict)
    for k in longer_bin_dict_key_list:
        v = longer_bin_dict[k]
        print('[%d-%d): %d' % (k * bin_size, k * bin_size + bin_size, v))

    print('num pos: %d, num neg: %d' % (counter_pos, counter_neg))
    print('total LP: %d' % total_pedestrian)


if __name__ == '__main__':

    # generate_data_list(ccpd_path='/home/hylink/eclipse-workspace/CCPD2019/')
    show_image()
    # dataset_statistics()

'''
This provider accepts an adapter, save dataset in file and load all dataset to memory for data iterators
'''

import cv2
import os
import time
from prepare_data.base_provider import ProviderBaseclass
from prepare_data.text_list_adapter import TextListAdapter


class DataSetProvider(ProviderBaseclass):
    """
    This class provides methods to save and read data.
    By default, images are compressed using JPG format.
    If data_adapter is not None, it means saving data, or it is reading data
    """

    def __init__(self,
                 data_set_list_path,
                 data_adapter=None, data_set_file_path=''):
        ProviderBaseclass.__init__(self)
        self.data_set_file_path = data_set_file_path
        self.data_set_list_path = data_set_list_path
        
        if data_adapter:  # write data

            self.data_adapter = data_adapter
            self.data = {}
            self.counter = 0

    def write(self):
        
        if not os.path.exists(os.path.dirname(self.data_set_file_path)):
            os.makedirs(os.path.dirname(self.data_set_file_path))
            
        if not os.path.exists(os.path.dirname(self.data_set_list_path)):
            os.makedirs(os.path.dirname(self.data_set_list_path))
            
        train = open(self.data_set_list_path, 'w')
        for data_item in self.data_adapter.get_one():

            temp_sample = []
            im, lable = data_item
            tempfile = str(int(time.time() * 1000)) + '.jpg'
            cv2.imwrite(self.data_set_file_path + tempfile, im)
            train.write(self.data_set_file_path + tempfile + ',' + ",".join(lable) + '\n')

            self.data[self.counter] = temp_sample
            print('Successfully save the %d-th data item.' % self.counter)
            self.counter += 1


def write_file(data_set_file_path='/home/hylink/ccpd_dataset/train/'):
    data_list_file_path = './data_folder/data_list_CCPD_train.txt'
    adapter = TextListAdapter(data_list_file_path)

    data_set_list_path = './data_folder/train.txt'
    
    packer = DataSetProvider(data_set_list_path, adapter, data_set_file_path)
    packer.write()


if __name__ == '__main__':
    write_file(data_set_file_path='/home/hylink/ccpd_dataset/train/')


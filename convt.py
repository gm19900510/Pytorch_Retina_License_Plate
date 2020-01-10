import os
f = open('/home/hylink/eclipse-workspace/CCPD2019/train.txt', 'r')
lines = f.readlines()
positive = []
for line in lines:
    line = line.strip('\n').split(',')
    line[0] = '/home/hylink/eclipse-workspace/CCPD2019/detection_dir/' + line[0]
    if os.path.exists(line[0]):
        lable = list(line[0:-1])
        new_lable = []
        new_lable.append(lable[0])
        new_lable.append('3')
        new_lable.append('1')
        new_lable = new_lable + lable[1:]
        positive.append(",".join(new_lable))
        print(new_lable)
print(positive[0:5])    
f = open('/home/hylink/eclipse-workspace/Pytorch_Retina_License_Plate_GM/prepare_data/data_folder/train.txt', 'r')
lines = f.readlines()

negative = []
for line in lines[0:60000]: 
    line = line.strip('\n') 
    lable = line.split(',')
    #if lable[1] == '-1'  :
    negative.append(line) 
        
print(len(negative))
print(negative[0:5])
train = open('train_white', 'a+')
all = positive + negative
for line in all:
    train.write(line + '\n')
      

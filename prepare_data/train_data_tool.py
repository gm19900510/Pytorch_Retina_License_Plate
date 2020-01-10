import os
name_list = os.listdir('/home/hylink/eclipse-workspace/License_Plate_Detection_Pytorch/ccpd/ccpd_dataset/train')
f = open('/home/hylink/eclipse-workspace/License_Plate_Detection_Pytorch/ccpd/ccpd_dataset/train.txt', 'r')
lines = f.readlines()
print(len(lines))
'''
txt_list = []       
for line in lines:
    line = line.strip('\n').split(',')
    txt_list.append(line[0])
    if line[0] not in name_list:
        print(line[0])
 
sub = list(set(txt_list).difference(set(name_list)))
print(sub) 
'''
all_list = []
neg_list = ['1.jpg,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1',
'2.jpg,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1',
'3.jpg,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1',
'4.jpg,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1',
'5.jpg,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1',
'6.jpg,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1',
'7.jpg,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1',
'8.jpg,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1',
'9.jpg,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1',
'10.jpg,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1',
'11.jpg,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1',
'12.jpg,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1',
'13.jpg,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1',
'15.jpg,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1',
'16.jpg,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1',
'17.jpg,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1',
'18.jpg,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1',
'19.jpg,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1',
'20.jpg,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1',
'21.jpg,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1']

for idx, line in enumerate(lines):
    if idx % 333 == 0:
        all_list.append(line) 
        all_list = all_list + neg_list
    else:
        all_list.append(line) 

print(len(all_list))  

train = open('/home/hylink/eclipse-workspace/License_Plate_Detection_Pytorch/ccpd/ccpd_dataset/new_train.txt', 'w')          
for line in all_list:
    train.write(line.strip('\n') + '\n')

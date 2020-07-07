'''
Created on 28-May-2020

@author: Neeraj Badal
'''
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
if __name__ == "__main__":
    
    no_of_samples = 100 
    functionDomain = [-3.0,3.0]
    
    train_X = [np.random.uniform(functionDomain[0],functionDomain[1],2)
                for i_ in range(0,no_of_samples)]
    
    
    
    train_X = np.array(train_X)
    np.random.shuffle(train_X)
    
#     outDataFile = "q2ci.dat"
#     
#     data_ = [line.strip('\n') for line in open(outDataFile, 'r')]
#     
#     train_X = data_[:]
#     train_X = [user_rating.split('\t') for user_rating in train_X]
# #     print(train_data)
#     train_X = np.array(train_X).astype(np.float)
    
    
    
    y_label = train_X[:,0]**2 + (train_X[:,1]**2)/2.0
    
    y_label[y_label <= 2.0] = 1
    
    y_label[y_label > 2.0] = -1
    
    color_label = []
    color_used = ['red','blue']
    
    for dat_ind in range(0,len(y_label)):
        if y_label[dat_ind] == -1:
            color_label.append('red')
        elif y_label[dat_ind] == 1:
            color_label.append('blue')

    classes = ['-1','1']
    plt.figure()
    plt.scatter(train_X[:,0],train_X[:,1],c = color_label)
    box_leg = []
    plt.title("Data set from q2c.i ",fontsize=18)
    for i_ in range(0,len(color_used)):
        box_leg.append(mpatches.Rectangle((0,0),1,1,fc=color_used[i_]))
    plt.legend(box_leg,classes,loc=4,prop={'size': 16})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    
    
    
    outDataFile = "q2ci.dat"
    count_ = 0
    with open(outDataFile, "w") as myfile:
        for data_ in train_X:
            myfile.write(str(data_[0])+"\t"+str(data_[1])+
                         "\t"+str(y_label[count_])+"\n")
             
            count_ = count_ + 1
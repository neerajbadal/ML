'''
Created on 27-May-2020

@author: Neeraj Badal
'''
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

if __name__ == "__main__":
    
    no_of_samples = 100 
    I_ = np.array([[1.0,0.0],[0.0,1.0]])
    w_ = np.random.multivariate_normal([0, 0], I_.T.dot(I_),1)[0]
    
    bias_ = np.random.normal(0.0, 1.0, 1)
    
    print(" w coefficients ",w_)
    print(" bias  ",bias_)
    
    functionDomain = [-3.0,3.0]
    
    train_X = [np.random.uniform(functionDomain[0],functionDomain[1],2)
                for i_ in range(0,no_of_samples)]
    
    train_X = np.array(train_X)
    np.random.shuffle(train_X)
    
    prod_val = np.dot(train_X,w_) + bias_
    
    y_label = np.sign(prod_val)
    
    color_used = ['red','blue']
    
    
    color_label = []
    
    for dat_ind in range(0,len(y_label)):
        if y_label[dat_ind] == -1:
            color_label.append('red')
        elif y_label[dat_ind] == 1:
            color_label.append('blue')

    classes = ['-1','1']
    plt.figure()
    plt.scatter(train_X[:,0],train_X[:,1],c = color_label)
    box_leg = []
    plt.title("Data set from q2a ",fontsize=18)
    for i_ in range(0,len(color_used)):
        box_leg.append(mpatches.Rectangle((0,0),1,1,fc=color_used[i_]))
    plt.legend(box_leg,classes,loc=4,prop={'size': 16})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    
    
    
    outDataFile = "q2i.dat"
    
    with open(outDataFile, "w") as myfile:
        myfile.write(str(w_[0])+"\t"+str(w_[1])+"\t"+
                             str(bias_[0])+"\n")
        for data_ in train_X:
            myfile.write(str(data_[0])+"\t"+str(data_[1])+"\n")
    
    
    
#     quadraticSolver(train_X,y_label)
    
    
    
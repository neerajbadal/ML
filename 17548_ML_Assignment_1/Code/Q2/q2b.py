'''
Created on 27-May-2020

@author: Neeraj Badal
'''
'''
Created on 27-May-2020

@author: Neeraj Badal
'''
import numpy as np
from matplotlib import pyplot as plt
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import matplotlib.patches as mpatches
from sklearn.model_selection import KFold
from sklearn.svm import SVC

def primalSVMCI(comb_data):
    train_set = comb_data[:100]
    test_set = comb_data[100:150]
    y_label_ = train_set[:,2]
    test_lab = test_set[:,2]
    
    m,n = train_set[:,0:2].shape
    H = np.c_[np.ones((1,n)),np.zeros((1,m+1))]
    H = np.diagflat(H)
    P = cvxopt_matrix(H)
    
    temp_mat = np.c_[train_set[:,0:2],np.ones((m,1)),np.zeros((m,m))]
    for row_i in range(0,len(temp_mat)):
        temp_mat[row_i] = -1*y_label_[row_i] * temp_mat[row_i] 
    
    temp_mat_2 = np.c_[np.zeros((m,n+1)),np.eye(m)]
    G = temp_mat - temp_mat_2
    
    temp_mat_3 = -1*np.c_[np.zeros((m,n+1)),np.eye(m)]
    h_mat_3 = np.zeros((m,1)) 
    
    G = np.r_[G,temp_mat_3]
    
    
    G = cvxopt_matrix(G)
    
    h = -1*np.ones((m,1))
    h = np.r_[h,h_mat_3]
    
    h = cvxopt_matrix(h)
    
    C_list = [0.1,0.2,0.4,0.6,0.8,0.9,1.1,2,3,5,10,50]
    
#     print(comb_data)

    models_ = []
    
    train_accuracy_c = []
    test_accuracy_c = []
#     cnt_ =0
    for C_ in C_list:
    
        q = np.c_[np.zeros((1,n+1)),C_*np.ones((1,m))].T
        print("q shape ",q.shape)
        q = cvxopt_matrix(q)
    
        sol = cvxopt_solvers.qp(P, q, G, h,None,None)
        alphas = np.array(sol['x'])
        new_y = np.sign(np.dot(train_set[:,0:2],alphas[0:2]) + alphas[2])
        new_y = np.squeeze(new_y)
        
        cvx_accuracy = new_y * y_label_
        cvx_accuracy[cvx_accuracy < 0] = 0
            
        cvx_accuracy = (np.sum(cvx_accuracy)) / len(cvx_accuracy)
        
        
        test_pred = np.sign(np.dot(test_set[:,0:2],alphas[0:2]) + alphas[2])
        test_pred = np.squeeze(test_pred)
        
#         plt.figure()
#         plt.scatter(test_set[:,0],test_set[:,1],c=test_pred,label=str(C_))
#         plt.legend()
        
        test_accuracy = test_pred * test_lab
        test_accuracy[test_accuracy < 0] = 0
            
        test_accuracy = (np.sum(test_accuracy)) / len(test_accuracy)
         
        train_accuracy_c.append(cvx_accuracy)
        test_accuracy_c.append(test_accuracy)
        
        xrange_pts = np.linspace(-3.0,3.0)
        slope_ = -1.0*alphas[0] / alphas[1]
        offset_ = alphas[2]
        y_vals = slope_ * xrange_pts - offset_ / alphas[1]
        
        plt.plot(xrange_pts,y_vals,label=str(C_),alpha=0.3)
        plt.legend(prop={'size': 16})
        
        models_.append(alphas)
         
         
    
    models_ = np.array(models_)
    
    max_accuracy_occ = np.argmax(train_accuracy_c)
    print("best train accuracy for C : ",C_list[max_accuracy_occ])
    
    print(" w and b params corresponding to best test fit ")
    print(" w : ",models_[max_accuracy_occ,0][0]," ",models_[max_accuracy_occ,1][0])
    print(" b : ",models_[max_accuracy_occ,2][0])
    print(" training accuracy : ",train_accuracy_c[max_accuracy_occ])
    print(" test accuracy : ",test_accuracy_c[max_accuracy_occ])
    
    
    new_y = np.sign(np.dot(train_set[:,0:2],models_[max_accuracy_occ,0:2]) +
                     models_[max_accuracy_occ,2])
    new_y = np.squeeze(new_y)
     
    test_pred = np.sign(np.dot(test_set[:,0:2],models_[max_accuracy_occ,0:2]) +
                        models_[max_accuracy_occ,2])
    test_pred = np.squeeze(test_pred)
     
    colors = ['red','blue']
    new_col_test = []
    new_col_train = []
     
    for i_ in range(0,len(test_pred)):
        if test_pred[i_] == -1:
            new_col_test.append('red')
        elif test_pred[i_] == 1:
            new_col_test.append('blue')
     
    for i_ in range(0,len(new_y)):
        if new_y[i_] == -1:
            new_col_train.append('red')
        elif new_y[i_] == 1:
            new_col_train.append('blue')
    
    xrange_pts = np.linspace(-3.0,3.0)
    slope_ = -1.0*models_[max_accuracy_occ,0] / models_[max_accuracy_occ,1]
    offset_ = models_[max_accuracy_occ,2]
    y_vals = slope_ * xrange_pts - offset_ / models_[max_accuracy_occ,1]
    max_margin_p1 = y_vals + 1.0 
    max_margin_m1 = y_vals - 1.0
    
    plt.scatter(test_set[:,0],test_set[:,1],c=new_col_test)
    plt.plot(xrange_pts,y_vals,c='red',lw=2,ls='--',label="optimal")
    plt.legend(prop={'size': 16})
#     plt.scatter(train_set[:,0],train_set[:,1],c=y_labe)
#     plt.plot(xrange_pts,y_vals)
    plt.show()
    
    
    fig, ax = plt.subplots()
    plt.scatter(train_set[:,0],train_set[:,1],c = new_col_train)
    plt.plot(xrange_pts,y_vals,c='red',lw=2,ls='--',label="wx+b = 0")
    plt.plot(xrange_pts,max_margin_p1,c='cyan',lw=2,ls='--',label="wx+b = 1")
    plt.plot(xrange_pts,max_margin_m1,c='green',lw=2,ls='--',label="wx+b= -1")
    leg = plt.legend(prop={'size': 16})
    box_leg = []
    ax.add_artist(leg)
    plt.title("Max-Margin and Separating Hyperplanes over Train Data ",fontsize=18)
    for i_ in range(0,len(color_used)):
        box_leg.append(mpatches.Rectangle((0,0),1,1,fc=color_used[i_]))
    plt.legend(box_leg,classes,loc=4,prop={'size': 16})
    
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.grid()
    plt.show()
    
    
    
    
    plt.title("Train and Test accuracy for different C ",fontsize=18)
    plt.plot(C_list,train_accuracy_c,marker='o',label='train',alpha=0.5)
    plt.plot(C_list,test_accuracy_c,marker='o',label='test',alpha=0.5)
    plt.plot(C_list[max_accuracy_occ],train_accuracy_c[max_accuracy_occ],
             marker='o',ms=9,label='optimal C from train accuracy')
    plt.plot(C_list[max_accuracy_occ],test_accuracy_c[max_accuracy_occ],
             marker='o',ms=9,label='Test accuracy for optimal C')
    
    plt.xlabel("C values",fontsize=18)
    plt.ylabel("Accuracy",fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(prop={'size': 16})
    plt.show()
    
    


if __name__ == "__main__":
    
    no_of_samples = 100 
    
    outDataFile = "q2i.dat"
    
    data_ = [line.strip('\n') for line in open(outDataFile, 'r')]
    
    line_1 = data_[0].split('\t')
    w_ = np.array([line_1[0],line_1[1]]).astype(np.float)
    bias_ = np.array(line_1[2]).astype(np.float)
    
    
    
    train_data = data_[1:]
    train_data = [user_rating.split('\t') for user_rating in train_data]
#     print(train_data)
    train_data = np.array(train_data).astype(np.float)
    print("loaded train data shape : ",train_data.shape)
    
#     train_data[:, 0],train_data[:, 1] = train_data[:, 1], train_data[:, 0].copy()
    
    prod_val = np.dot(train_data,w_) + bias_
    
    y_label = np.sign(prod_val)
    
    color_used = ['red','blue']
    
    color_label = []
    
    for dat_ind in range(0,len(y_label)):
        if y_label[dat_ind] == -1:
            color_label.append('red')
        elif y_label[dat_ind] == 1:
            color_label.append('blue')

    classes = ['-1','1']
    
    functionDomain = [-3.0,3.0]
    
    test_X = [np.random.uniform(functionDomain[0],functionDomain[1],2)
                for i_ in range(0,50)]
     
    test_X = np.array(test_X)
    np.random.shuffle(test_X)
     
    prod_val = np.dot(test_X,w_) + bias_
     
    test_label = np.sign(prod_val)
    
    train_data = np.r_[train_data,test_X]
    y_label = np.r_[y_label,test_label]

    plt.figure()
#     plt.scatter(test_X[:,0],test_X[:,1],c = test_label)
    box_leg = []
    plt.title("Various separating Hyperplanes over Test Data ",fontsize=18)
    for i_ in range(0,len(color_used)):
        box_leg.append(mpatches.Rectangle((0,0),1,1,fc=color_used[i_]))
    plt.legend(box_leg,classes,loc=4,prop={'size': 16})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    
    comb_data = np.c_[train_data,y_label] 
    primalSVMCI(comb_data)
        
    
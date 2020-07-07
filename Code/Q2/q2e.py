'''
Created on 28-May-2020

@author: Neeraj Badal
'''
'''
Created on 28-May-2020

@author: Neeraj Badal
'''
import numpy as np
from matplotlib import pyplot as plt
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import matplotlib.patches as mpatches

def polyKernel(x_,y_,deg_=2):
    val = (np.dot(x_,y_) + 1)**deg_ 
    return val

def gaussKernel(x_,y_,sigma_=1.0):
    diff_ = x_ - y_
    diff_ = np.linalg.norm(diff_,2)
    diff_ = diff_ / (2.0 * (sigma_**2))
    return np.exp(-1* diff_)
    
    

def dualSVMC(comb_data,k_func,h_param_):
    train_set = comb_data[:100]
    test_set = comb_data[100:150]
    y_label_ = train_set[:,2]
    test_lab = test_set[:,2]
    m,n = train_set[:,0:2].shape
    
    K_gram = np.zeros((m,m),dtype=np.float)
    
    for i_  in range(0,len(train_set)):
        for j_ in range(0,len(train_set)):
            K_gram[i_,j_] = k_func(train_set[i_,0:2],train_set[j_,0:2],h_param_)*(y_label_[i_]*
                                                                             y_label_[j_])
    C_list = [0.1,0.2,0.4,0.6,0.8,0.9,1.1,2,3,5,10,50]

    P = cvxopt_matrix(K_gram)
    q = cvxopt_matrix(-1*np.ones((m, 1)))
    G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
    
    A = cvxopt_matrix(y_label_.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))
    
    train_accuracy_c = []
    test_accuracy_c = []
    
    train_preds_ = []
    test_preds_ = []
    
    for C_ in C_list:
        h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C_)))
        
        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])
        
        S = (alphas > 1e-4).flatten()
        S_ = (alphas <= C_ ).flatten()
        S = S & S_
        
        supportVector_index = np.argwhere(S == True)
        supportVector_index = np.squeeze(supportVector_index)
        
        offset_b = computeOffset(supportVector_index, alphas, train_set,k_func,h_param_)
        
        new_preds = [predSVMC(supportVector_index,alphas,train_set,offset_b,new_dat,k_func,h_param_)
                      for new_dat in train_set ]
        
        new_preds = np.array(new_preds)
    
        new_preds_test = [predSVMC(supportVector_index,alphas,train_set,offset_b,new_dat,k_func,h_param_)
                      for new_dat in test_set ]
        
        new_preds_test = np.array(new_preds_test)
    
        test_new_preds = new_preds_test
        train_new_preds =  new_preds
        
        test_accuracy = test_new_preds * test_lab
        test_accuracy[test_accuracy < 0] = 0
        test_accuracy = (np.sum(test_accuracy)) / len(test_accuracy)
        
        train_acc = train_new_preds * y_label_
        train_acc[train_acc < 0] = 0
            
        train_acc = (np.sum(train_acc)) / len(train_acc)
        
        train_accuracy_c.append(train_acc)
        test_accuracy_c.append(test_accuracy)
        
        train_preds_.append(train_new_preds)
        test_preds_.append(test_new_preds)
    
    
    max_accuracy_occ = np.argmax(train_accuracy_c)
    
    return([C_list[max_accuracy_occ],train_accuracy_c[max_accuracy_occ]
            ,test_accuracy_c[max_accuracy_occ],train_preds_[max_accuracy_occ],
            test_preds_[max_accuracy_occ]])


def computeOffset(support_vectors_index,alphas_,train_data_,k_func_,h_param):
    
    offset_ = []
    for i_ in range(0,len(support_vectors_index)):
        y_val = train_data_[support_vectors_index[i_],2] 
        sum_t = 0.0
        for j_ in range(0,len(support_vectors_index)):
            dot_prod_val = k_func_(train_data_[support_vectors_index[i_],0:2],
                                      train_data_[support_vectors_index[j_],0:2],h_param) 
            
            dot_prod_val = dot_prod_val * alphas_[support_vectors_index[j_]] * train_data_[support_vectors_index[j_],2]  
            sum_t += dot_prod_val
        
        sum_t = y_val - sum_t
        offset_.append(sum_t)
    return np.mean(offset_)
    

def predSVMC(support_vectors_index,alphas_,train_data_,b_val,new_data,k_func_,h_param):
    sum_t = 0.0
    for i_ in range(0,len(support_vectors_index)):
        dot_prod_val = k_func_(train_data_[support_vectors_index[i_],0:2],
                                  new_data[0:2],h_param) 
        
        dot_prod_val = dot_prod_val * alphas_[support_vectors_index[i_]] * train_data_[support_vectors_index[i_],2]  
        sum_t += dot_prod_val
    
    pred_val = np.sign(sum_t + b_val)
    return pred_val[0]
        


def assignColors(labels_):
    color_used = ['red','blue']
    color_label = []
    for dat_ind in range(0,len(labels_)):
        if labels_[dat_ind] == -1:
            color_label.append('red')
        elif labels_[dat_ind] == 1:
            color_label.append('blue')

    classes = ['C1','C2']
    
    box_leg = []
    
    for i_ in range(0,len(color_used)):
        box_leg.append(mpatches.Rectangle((0,0),1,1,fc=color_used[i_]))
    plt.legend(box_leg,classes,loc=4,prop={'size': 16})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    return color_label

    

if __name__ == "__main__":
    
    no_of_samples = 100 
    
    outDataFile = "q2ci.dat"
    
    data_ = [line.strip('\n') for line in open(outDataFile, 'r')]
    
    
    
    
    data_ = [user_rating.split('\t') for user_rating in data_]
#     print(train_data)
    train_data = data_[:no_of_samples]
    train_data = np.array(train_data).astype(np.float)
    
    y_label = train_data[:,2]
    train_data = train_data[:,0:2]
    
    
    functionDomain = [-3.0,3.0]
    
    test_X = [np.random.uniform(functionDomain[0],functionDomain[1],2)
                for i_ in range(0,50)]
     
    test_X = np.array(test_X)
    np.random.shuffle(test_X)
     
    
    test_label = test_X[:,0]**2 + (test_X[:,1]**2)/2.0
    
    test_label[test_label <= 2.0] = 1
    
    test_label[test_label > 2.0] = -1
    

    train_data = np.r_[train_data,test_X]
    y_label = np.r_[y_label,test_label]
    
    train_data = np.c_[train_data,y_label]
    
    kernel_func = [polyKernel,gaussKernel]
    h_param = [[1,2,3,4,5],
               [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,5.0]
        ]
    model_kernel_name = ['poly','RBF']
    model_kernel_tuning_param = ['deg','sig']
    mse_func = []
    model_params = []
    
    mse_comp = []
    mse_model_names = []
    
    for k_ind in range(0,len(kernel_func)):
        h_val_mse = []
        h_val_coll = []
        for h_val in range(0,len(h_param[k_ind])): 
            tmp_mse = dualSVMC(train_data,kernel_func[k_ind],h_param[k_ind][h_val])
            h_val_mse.append(tmp_mse)
            operat_name = model_kernel_name[k_ind] +"_"+model_kernel_tuning_param[k_ind]+"_"+str(h_param[k_ind][h_val])+"_"+str(tmp_mse[0]) 
            mse_comp.append([tmp_mse[0],tmp_mse[1],tmp_mse[2],operat_name,tmp_mse[3],tmp_mse[4]])
#             print(" model : ",operat_name," train accuracy : ",tmp_mse[1]," test accuracy :",
#                   tmp_mse[2]," C value : ",tmp_mse[0])
#             show_projected_classes(tmp_mse[4],tmp_mse[3], tmp_mse[6],train_data,operat_name)
            
            
    mse_comp = np.array(mse_comp)
    best_model_ind = np.argmax(mse_comp[:,1].astype(np.float))
    
    print("models tried out")
    for i_ in range(0,len(mse_comp)):
        print(" model : ",mse_comp[i_,3]," train accuracy : ",mse_comp[i_,1]," test accuracy :",
                  mse_comp[i_,2]," C value : ",mse_comp[i_,0])
        
    
    print("chosen model ")
    print(" model : ",mse_comp[best_model_ind,3]," train accuracy : ",mse_comp[best_model_ind,1]," test accuracy :",
                  mse_comp[best_model_ind,2]," C value : ",mse_comp[best_model_ind,0])
    
    plt.figure()
    plt.title("train & test classification accuracy across models",fontsize=18)
    
    plt.plot(mse_comp[:,3],mse_comp[:,1].astype(np.float),marker='o',label='train accuracy')
    plt.plot(mse_comp[:,3],mse_comp[:,2].astype(np.float),marker='o',label='test accuracy')
    plt.xticks(fontsize=15,rotation=-90)
    plt.yticks(fontsize=16)
    plt.legend(prop={'size': 16})
    plt.show()
    
    
    plt.figure()
    color_label = assignColors(train_data[:no_of_samples,2])
    plt.title(" Train Data ",fontsize=18)
    plt.scatter(train_data[:no_of_samples,0],train_data[:no_of_samples,1],c = color_label)
    
    plt.figure()
    color_label = assignColors(mse_comp[best_model_ind,4])
    plt.title(" Classification over Train Data ",fontsize=18)
    plt.scatter(train_data[:no_of_samples,0],train_data[:no_of_samples,1],c = color_label)
    
    plt.show()
    
    
    
    
    plt.figure()
    color_label = assignColors(test_label)
    plt.title(" Test Data ",fontsize=18)
    plt.scatter(test_X[:,0],test_X[:,1],c = color_label)
    
    plt.figure()
    color_label = assignColors(mse_comp[best_model_ind,5])
    plt.title(" Classification over Test Data ",fontsize=18)
    plt.scatter(test_X[:,0],test_X[:,1],c = color_label)
    
    plt.show()
    
    
    
    
    
    
    
    
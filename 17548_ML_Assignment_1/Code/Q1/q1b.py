'''
Created on 27-May-2020

@author: Neeraj Badal
'''
import dill
import numpy as np

def createGramMatrix(m_size,kernel_func_map,sampler_=None):
    functionDomain = [-5.0,5.0]
    ''' sample m_size samples '''
    
    if sampler_ != None:
        train_X = [sampler_() for i_ in range(0,m_size)]
    else:
        train_X = [np.random.uniform(functionDomain[0],functionDomain[1],3).reshape(-1,1) for i_ in range(0,m_size)] 
    
    
    train_X = np.array(train_X)
    
    ''' Structure and Fill the gram matrix '''
    K_gram = np.zeros((m_size,m_size),dtype=np.float)
    for i_  in range(0,len(train_X)):
        for j_ in range(0,len(train_X)):
            K_gram[i_,j_] = kernel_func_map(train_X[i_],train_X[j_])
    
    symm_true = np.array_equal(K_gram, K_gram.T)
#     print(K_gram)
    print("is symmetric ",symm_true)
    eig_vals_ = np.linalg.eigvals(K_gram)
    eig_vals_ = np.around(eig_vals_,decimals=3)
    eig_vals_ += 0.0
    all_eig_vals_non_neg = np.all(eig_vals_ >= 0)
    
    isValidKernel = symm_true and all_eig_vals_non_neg
    print("is positive semifinite ",all_eig_vals_non_neg)
    print("is a valid kernel ? ",isValidKernel)
    print("------------------------------------")

if __name__ == "__main__":
    functionFiles = ["function1.pkl","function2.pkl","function3.pkl","function4.pkl",
                     "function5.pkl"]
    pklDir = "./"
    
    no_of_samples = 900

    for f_i in range(0,len(functionFiles)-1):
        ''' load function pkl'''
        fileId = open(pklDir+functionFiles[f_i],'rb')
        f_l = dill.loads(dill.load(fileId))
        print("function probed k",(f_i+1))
        createGramMatrix(no_of_samples,f_l)
        
    
    
    print("checking for function 5 given with sampler")
    
    ''' load function pkl for given sampler as well'''
    fileId = open(pklDir+functionFiles[4],'rb')
    ''' load sampler pkl'''
    samplerFileId = open(pklDir+"k5sampler.pkl",'rb')
        
    f_l = dill.loads(dill.load(fileId))
    s_l = dill.loads(dill.load(samplerFileId))
    
    createGramMatrix(900,f_l,s_l)
    
        
        
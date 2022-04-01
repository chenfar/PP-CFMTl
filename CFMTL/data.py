import numpy as np
from torchvision import datasets, transforms
import copy

def mnist_iid(train_dataset, test_dataset, num_clients, num_class):
    ids = np.arange(len(train_dataset))
    labels = train_dataset.train_labels.numpy()
    ids_labels = np.vstack((ids, labels))
    ids_labels = ids_labels[:,ids_labels[1,:].argsort()]
    ids = ids_labels[0,:]
    train_ids = list(ids)
    
    ids = np.arange(len(test_dataset))
    labels = test_dataset.test_labels.numpy()
    ids_labels = np.vstack((ids, labels))
    ids_labels = ids_labels[:,ids_labels[1,:].argsort()]
    ids = ids_labels[0,:]
    test_ids = list(ids)

    train_class_items = int(len(train_dataset)/num_class)
    test_class_items = int(len(test_dataset)/num_class)
    train_num_items = int(train_class_items/num_clients)
    test_num_items = int(test_class_items/num_clients)
    train_dict_clients = [[] for i in range(num_clients)]
    test_dict_clients = [[] for i in range(num_clients)]
    
    new_train_ids = []
    for i in range(num_class):
        new_train_ids += list(np.random.permutation(train_ids[i*train_class_items : (i+1)*train_class_items]))
    train_ids = new_train_ids
    new_test_ids = []
    for i in range(num_class):
        new_test_ids += list(np.random.permutation(test_ids[i*test_class_items : (i+1)*test_class_items]))
    test_ids = new_test_ids
    
    
    for i in range(num_clients):
        for j in range(num_class):
            train_dict = train_ids[j*train_class_items+i*train_num_items : j*train_class_items+(i+1)*train_num_items]
            test_dict = test_ids[j*test_class_items+i*test_num_items : j*test_class_items+(i+1)*test_num_items]
            train_dict_clients[i] += train_dict
            test_dict_clients[i] += test_dict
    return train_dict_clients, test_dict_clients
    
def mnist_non_iid(train_dataset, test_dataset, num_clients, num_class, ratio):
    
    ids = np.arange(len(train_dataset))
    labels = train_dataset.train_labels.numpy()
    ids_labels = np.vstack((ids, labels))
    ids_labels = ids_labels[:,ids_labels[1,:].argsort()]
    ids = ids_labels[0,:]
    train_ids = list(ids)
    
    ids = np.arange(len(test_dataset))
    labels = test_dataset.test_labels.numpy()
    ids_labels = np.vstack((ids, labels))
    ids_labels = ids_labels[:,ids_labels[1,:].argsort()]
    ids = ids_labels[0,:]
    test_ids = list(ids)

    train_class_items = int(len(train_dataset)/num_class)
    test_class_items = int(len(test_dataset)/num_class)
    train_class_items_1 = int(len(train_dataset)/num_class*ratio)
    test_class_items_1 = int(len(test_dataset)/num_class*ratio)
    train_class_items_2 = int(len(train_dataset)/num_class*(1-ratio))
    test_class_items_2 = int(len(test_dataset)/num_class*(1-ratio))
    train_num_items_1 =  int(len(train_dataset)/num_clients*ratio)
    test_num_items_1 = int(len(test_dataset)/num_clients*ratio)
    train_num_items_2 = int(train_class_items_2/num_clients)
    test_num_items_2 = int(test_class_items_2/num_clients)
    train_dict_clients = [[] for i in range(num_clients)]
    test_dict_clients = [[] for i in range(num_clients)]
    
    new_train_ids = []
    for i in range(num_class):
        new_train_ids += list(np.random.permutation(train_ids[i*train_class_items : (i+1)*train_class_items]))
    train_ids = new_train_ids
    new_test_ids = []
    for i in range(num_class):
        new_test_ids += list(np.random.permutation(test_ids[i*test_class_items : (i+1)*test_class_items]))
    test_ids = new_test_ids 
    
    train_ids_1 = []
    test_ids_1 = []
    train_ids_2 = []
    test_ids_2 = []
    for i in range(num_class):
        train_ids_1 += train_ids[i*train_class_items : i*train_class_items+train_class_items_1]
        test_ids_1 += test_ids[i*test_class_items : i*test_class_items+test_class_items_1]
        train_ids_2 += train_ids[i*train_class_items+train_class_items_1 : (i+1)*train_class_items]
        test_ids_2 += test_ids[i*test_class_items+test_class_items_1 : (i+1)*train_class_items]
    
    for i in range(num_clients):
        train_dict_clients[i] += train_ids_1[i*train_num_items_1 : (i+1)*train_num_items_1]
        test_dict_clients[i] += test_ids_1[i*test_num_items_1 : (i+1)*test_num_items_1]
        for j in range(num_class):
            train_dict_clients[i] += train_ids_2[j*train_class_items_2+i*train_num_items_2 : j*train_class_items_2+(i+1)*train_num_items_2]
            test_dict_clients[i] += test_ids_2[j*test_class_items_2+i*test_num_items_2 : j*test_class_items_2+(i+1)*test_num_items_2]            
    return train_dict_clients, test_dict_clients
    
def mnist_non_iid_single_class(train_dataset, test_dataset, num_clients, num_class):
    ids = np.arange(len(train_dataset))
    labels = train_dataset.train_labels.numpy()
    ids_labels = np.vstack((ids, labels))
    ids_labels = ids_labels[:,ids_labels[1,:].argsort()]
    ids = ids_labels[0,:]
    train_ids = list(ids)
    
    ids = np.arange(len(test_dataset))
    labels = test_dataset.test_labels.numpy()
    ids_labels = np.vstack((ids, labels))
    ids_labels = ids_labels[:,ids_labels[1,:].argsort()]
    ids = ids_labels[0,:]
    test_ids = list(ids)

    train_class_items = int(len(train_dataset)/num_class)
    test_class_items = int(len(test_dataset)/num_class)
    train_num_items = int(len(train_dataset)/num_clients)
    test_num_items = int(len(test_dataset)/num_clients)
    train_dict_clients = [[] for i in range(num_clients)]
    test_dict_clients = [[] for i in range(num_clients)]
    
    new_train_ids = []
    for i in range(num_class):
        new_train_ids += list(np.random.permutation(train_ids[i*train_class_items : (i+1)*train_class_items]))
    train_ids = new_train_ids
    new_test_ids = []
    for i in range(num_class):
        new_test_ids += list(np.random.permutation(test_ids[i*test_class_items : (i+1)*test_class_items]))
    test_ids = new_test_ids
    
    for i in range(num_clients):
        train_dict = train_ids[i*train_num_items : (i+1)*train_num_items]
        test_dict = test_ids[i*test_num_items : (i+1)*test_num_items]
        train_dict_clients[i] += train_dict
        test_dict_clients[i] += test_dict
    return train_dict_clients, test_dict_clients
    
def cifar_iid(train_dataset, test_dataset, num_clients, num_class):
    train_class_items = int(len(train_dataset)/num_class)
    test_class_items = int(len(test_dataset)/num_class)
    train_num_items = int(train_class_items/num_clients)
    test_num_items = int(test_class_items/num_clients)
    train_dict_clients = [[] for i in range(num_clients)]
    test_dict_clients = [[] for i in range(num_clients)]
    
    train_ids_class = [[] for i in range(num_class)]
    for i in range(len(train_dataset)):
        image, label = train_dataset[i]
        train_ids_class[label].append(i)
        
    train_ids = []
    for i in range(len(train_ids_class)):
        train_ids += train_ids_class[i]
        
    test_ids_class = [[] for i in range(num_class)]
    for i in range(len(test_dataset)):
        image, label = test_dataset[i]
        test_ids_class[label].append(i)
        
    test_ids = []
    for i in range(len(test_ids_class)):
        test_ids += test_ids_class[i]
    
    new_train_ids = []
    for i in range(num_class):
        new_train_ids += list(np.random.permutation(train_ids[i*train_class_items : (i+1)*train_class_items]))
    train_ids = new_train_ids
    new_test_ids = []
    for i in range(num_class):
        new_test_ids += list(np.random.permutation(test_ids[i*test_class_items : (i+1)*test_class_items]))
    test_ids = new_test_ids
    
    for i in range(num_clients):
        for j in range(num_class):
            train_dict = train_ids[j*train_class_items+i*train_num_items : j*train_class_items+(i+1)*train_num_items]
            test_dict = test_ids[j*test_class_items+i*test_num_items : j*test_class_items+(i+1)*test_num_items]
            train_dict_clients[i] += train_dict
            test_dict_clients[i] += test_dict
    return train_dict_clients, test_dict_clients

def cifar_non_iid(train_dataset, test_dataset, num_clients, num_class, ratio):

    train_ids_class = [[] for i in range(num_class)]
    for i in range(len(train_dataset)):
        image, label = train_dataset[i]
        train_ids_class[label].append(i)
        
    train_ids = []
    for i in range(len(train_ids_class)):
        train_ids += train_ids_class[i]
    
    test_ids_class = [[] for i in range(num_class)]
    for i in range(len(test_dataset)):
        image, label = test_dataset[i]
        test_ids_class[label].append(i)
        
    test_ids = []
    for i in range(len(test_ids_class)):
        test_ids += test_ids_class[i]

    train_class_items = int(len(train_dataset)/num_class)
    test_class_items = int(len(test_dataset)/num_class)
    train_class_items_1 = int(len(train_dataset)/num_class*ratio)
    test_class_items_1 = int(len(test_dataset)/num_class*ratio)
    train_class_items_2 = int(len(train_dataset)/num_class*(1-ratio))
    test_class_items_2 = int(len(test_dataset)/num_class*(1-ratio))
    train_num_items_1 =  int(len(train_dataset)/num_clients*ratio)
    test_num_items_1 = int(len(test_dataset)/num_clients*ratio)
    train_num_items_2 = int(train_class_items_2/num_clients)
    test_num_items_2 = int(test_class_items_2/num_clients)
    train_dict_clients = [[] for i in range(num_clients)]
    test_dict_clients = [[] for i in range(num_clients)]
    
    new_train_ids = []
    for i in range(num_class):
        new_train_ids += list(np.random.permutation(train_ids[i*train_class_items : (i+1)*train_class_items]))
    train_ids = new_train_ids
    new_test_ids = []
    for i in range(num_class):
        new_test_ids += list(np.random.permutation(test_ids[i*test_class_items : (i+1)*test_class_items]))
    test_ids = new_test_ids 
    
    train_ids_1 = []
    test_ids_1 = []
    train_ids_2 = []
    test_ids_2 = []
    for i in range(num_class):
        train_ids_1 += train_ids[i*train_class_items : i*train_class_items+train_class_items_1]
        test_ids_1 += test_ids[i*test_class_items : i*test_class_items+test_class_items_1]
        train_ids_2 += train_ids[i*train_class_items+train_class_items_1 : (i+1)*train_class_items]
        test_ids_2 += test_ids[i*test_class_items+test_class_items_1 : (i+1)*train_class_items]
    
    for i in range(num_clients):
        train_dict_clients[i] += train_ids_1[i*train_num_items_1 : (i+1)*train_num_items_1]
        test_dict_clients[i] += test_ids_1[i*test_num_items_1 : (i+1)*test_num_items_1]
        for j in range(num_class):
            train_dict_clients[i] += train_ids_2[j*train_class_items_2+i*train_num_items_2 : j*train_class_items_2+(i+1)*train_num_items_2]
            test_dict_clients[i] += test_ids_2[j*test_class_items_2+i*test_num_items_2 : j*test_class_items_2+(i+1)*test_num_items_2]            
    return train_dict_clients, test_dict_clients
    
def cifar_non_iid_single_class(train_dataset, test_dataset, num_clients, num_class):
    train_ids_class = [[] for i in range(num_class)]
    for i in range(len(train_dataset)):
        image, label = train_dataset[i]
        train_ids_class[label].append(i)
        
    train_ids = []
    for i in range(len(train_ids_class)):
        train_ids += train_ids_class[i]
        
    test_ids_class = [[] for i in range(num_class)]
    for i in range(len(test_dataset)):
        image, label = test_dataset[i]
        test_ids_class[label].append(i)
        
    test_ids = []
    for i in range(len(test_ids_class)):
        test_ids += test_ids_class[i]

    train_class_items = int(len(train_dataset)/num_class)
    test_class_items = int(len(test_dataset)/num_class)
    train_num_items = int(len(train_dataset)/num_clients)
    test_num_items = int(len(test_dataset)/num_clients)
    train_dict_clients = [[] for i in range(num_clients)]
    test_dict_clients = [[] for i in range(num_clients)]
    
    new_train_ids = []
    for i in range(num_class):
        new_train_ids += list(np.random.permutation(train_ids[i*train_class_items : (i+1)*train_class_items]))
    train_ids = new_train_ids
    new_test_ids = []
    for i in range(num_class):
        new_test_ids += list(np.random.permutation(test_ids[i*test_class_items : (i+1)*test_class_items]))
    test_ids = new_test_ids
    
    for i in range(num_clients):
        train_dict = train_ids[i*train_num_items : (i+1)*train_num_items]
        test_dict = test_ids[i*test_num_items : (i+1)*test_num_items]
        train_dict_clients[i] += train_dict
        test_dict_clients[i] += test_dict
    return train_dict_clients, test_dict_clients
    
'''def cifar_non_iid(train_dataset, test_dataset, num_clients, num_class, ratio):

    train_ids_class = [[] for i in range(num_class)]
    for i in range(len(train_dataset)):
        image, label = train_dataset[i]
        train_ids_class[label].append(i)
        
    train_ids = []
    for i in range(len(train_ids_class)):
        train_ids += train_ids_class[i]
    
    test_ids_class = [[] for i in range(num_class)]
    for i in range(len(test_dataset)):
        image, label = test_dataset[i]
        test_ids_class[label].append(i)
        
    test_ids = []
    for i in range(len(test_ids_class)):
        test_ids += test_ids_class[i]

    train_class_items = int(len(train_dataset)/num_class)
    test_class_items = int(len(test_dataset)/num_class)
    train_class_items_1 = int(len(train_dataset)/num_class*ratio)
    test_class_items_1 = int(len(test_dataset)/num_class*ratio)
    train_class_items_2 = int(len(train_dataset)/num_class*(1-ratio))
    test_class_items_2 = int(len(test_dataset)/num_class*(1-ratio))
    train_num_items_1 =  int(len(train_dataset)/num_clients*ratio/(num_class/2))
    test_num_items_1 = int(len(test_dataset)/num_clients*ratio/(num_class/2))
    train_num_items_2 = int(train_class_items_2/num_clients)
    test_num_items_2 = int(test_class_items_2/num_clients)
    train_dict_clients = [[] for i in range(num_clients)]
    test_dict_clients = [[] for i in range(num_clients)]
    
    new_train_ids = []
    for i in range(num_class):
        new_train_ids += list(np.random.permutation(train_ids[i*train_class_items : (i+1)*train_class_items]))
    train_ids = new_train_ids
    new_test_ids = []
    for i in range(num_class):
        new_test_ids += list(np.random.permutation(test_ids[i*test_class_items : (i+1)*test_class_items]))
    test_ids = new_test_ids 
    
    train_ids_1 = []
    test_ids_1 = []
    train_ids_2 = []
    test_ids_2 = []
    for i in range(num_class):
        train_ids_1 += train_ids[i*train_class_items : i*train_class_items+train_class_items_1]
        test_ids_1 += test_ids[i*test_class_items : i*test_class_items+test_class_items_1]
        train_ids_2 += train_ids[i*train_class_items+train_class_items_1 : (i+1)*train_class_items]
        test_ids_2 += test_ids[i*test_class_items+test_class_items_1 : (i+1)*train_class_items]
    
    for i in range(num_clients):
        if i < int(num_clients/2):
            for j in range(int(num_class/2)):
                train_dict_clients[i] += train_ids_1[j*train_class_items_1+i*train_num_items_1 : j*train_class_items_1+(i+1)*train_num_items_1]
                test_dict_clients[i] += test_ids_1[j*test_class_items_1+i*test_num_items_1 : j*test_class_items_1+(i+1)*test_num_items_1]
        else:
            for j in range(int(num_class/2),num_class):
                train_dict_clients[i] += train_ids_1[j*train_class_items_1+(i-int(num_clients/2))*train_num_items_1 : j*train_class_items_1+(i-int(num_clients/2)+1)*train_num_items_1]
                test_dict_clients[i] += test_ids_1[j*test_class_items_1+(i-int(num_clients/2))*test_num_items_1 : j*test_class_items_1+(i-int(num_clients/2)+1)*test_num_items_1]                
        for j in range(num_class):
            train_dict_clients[i] += train_ids_2[j*train_class_items_2+i*train_num_items_2 : j*train_class_items_2+(i+1)*train_num_items_2]
            test_dict_clients[i] += test_ids_2[j*test_class_items_2+i*test_num_items_2 : j*test_class_items_2+(i+1)*test_num_items_2]            
    return train_dict_clients, test_dict_clients
    
def cifar_non_iid_single_class(train_dataset, test_dataset, num_clients, num_class):
    train_ids_class = [[] for i in range(num_class)]
    for i in range(len(train_dataset)):
        image, label = train_dataset[i]
        train_ids_class[label].append(i)
        
    train_ids = []
    for i in range(len(train_ids_class)):
        train_ids += train_ids_class[i]
        
    test_ids_class = [[] for i in range(num_class)]
    for i in range(len(test_dataset)):
        image, label = test_dataset[i]
        test_ids_class[label].append(i)
        
    test_ids = []
    for i in range(len(test_ids_class)):
        test_ids += test_ids_class[i]

    train_class_items = int(len(train_dataset)/num_class)
    test_class_items = int(len(test_dataset)/num_class)
    train_num_items = int(len(train_dataset)/num_clients/(num_class/2))
    test_num_items = int(len(test_dataset)/num_clients/(num_class/2))
    train_dict_clients = [[] for i in range(num_clients)]
    test_dict_clients = [[] for i in range(num_clients)]
    
    new_train_ids = []
    for i in range(num_class):
        new_train_ids += list(np.random.permutation(train_ids[i*train_class_items : (i+1)*train_class_items]))
    train_ids = new_train_ids
    new_test_ids = []
    for i in range(num_class):
        new_test_ids += list(np.random.permutation(test_ids[i*test_class_items : (i+1)*test_class_items]))
    test_ids = new_test_ids
    
    for i in range(num_clients):
        if i < int(num_clients/2):
            for j in range(int(num_class/2)):
                train_dict = train_ids[j*train_class_items+i*train_num_items : j*train_class_items+(i+1)*train_num_items]
                test_dict = test_ids[j*test_class_items+i*test_num_items : j*test_class_items+(i+1)*test_num_items]
                train_dict_clients[i] += train_dict
                test_dict_clients[i] += test_dict
        else:
            for j in range(int(num_class/2),num_class):
                train_dict = train_ids[j*train_class_items+(i-int(num_clients/2))*train_num_items : j*train_class_items+(i-int(num_clients/2)+1)*train_num_items]
                test_dict = test_ids[j*test_class_items+(i-int(num_clients/2))*test_num_items : j*test_class_items+(i-int(num_clients/2)+1)*test_num_items]
                train_dict_clients[i] += train_dict
                test_dict_clients[i] += test_dict            
    return train_dict_clients, test_dict_clients'''
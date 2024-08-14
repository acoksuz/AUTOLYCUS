import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import random
import seaborn as sns
import shap
import sklearn
import warnings
import pickle

from matplotlib import pyplot as plt
import sklearn.tree as tree
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.linear_model import LogisticRegression as lr
from sklearn.naive_bayes import MultinomialNB as mnb
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
np.set_printoptions(suppress=True)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
        
def takeFourth(elem):
    return abs(elem[3])

def explanation_parser(expMap,expList,key,features):
    result = []
    for i in features:
        result = result + [[i,-1,-1,0]]
    #result = [[0,-1,-1,0],[1,-1,-1,0],[2,-1,-1,0],[3,-1,-1,0]] 
    # Use an impossible value like -1 for no-information parts. 
    indices = []
    tmp = 0
    for i in expMap[key]:
        indices = indices + [i[0]]
    for i in expList:
        txt = i[0].split(' ')
        #print(txt)
        if len(txt) < 4: #length of lime explanation is 7 for this particular dataset's normal samples
            if (txt[-2] == '<=') or (txt[-2] == '<'):
                result[indices[tmp]][2] = float(txt[-1])
            else:
                result[indices[tmp]][1] = float(txt[-1])
        else:
            result[indices[tmp]][1] = float(txt[0])
            result[indices[tmp]][2] = float(txt[-1])
        result[indices[tmp]][3] = round(float(i[1]),2)
        tmp = tmp + 1
    result.sort(key=takeFourth, reverse=True)    
    return result

def extract_explanation_boundaries(model, explainer, n_ft):
    boundaries = np.zeros((n_ft,3))
    sample = np.zeros(n_ft)
    for i in range(3):
        exp = explainer.explain_instance(sample, model.predict_proba, top_labels=1)
        exp_map = exp.as_map()
        key = list(exp_map.keys())[0]
        exp_list = exp.as_list(key)
        exp_parsed = explanation_parser(exp_map,exp_list,key,features)
        for j in range(n_ft):
            tmp_exp = exp_parsed[j]
            feature_index = [index for index, content in enumerate(features) if tmp_exp[0] in content][0]
            boundaries[feature_index,i] = tmp_exp[2]
            sample[feature_index] = tmp_exp[2] + 1 #0.01        
    return boundaries

def sample_set_generation(dataset, n_classes, n_samples_per_class): # Make sure that the dataset is sorted&balanced
    sample_set = []
    if isinstance(n_samples_per_class, list):
        lister = n_samples_per_class
    else:
        lister = np.ones((n_classes,), dtype=int)*n_samples_per_class
    for i in range(len(lister)):
        tmp = np.where(dataset[:,-1] == i)
        indices = random.sample(tmp[0].tolist(), lister[i])
        for j in indices:
            sample_set += [dataset[j][:-1]]
    return sample_set

def traverse_explanations_LIME(sample_set, explainer, model, n_visits_lb, n_visits_ub, upper_limit, n_f_e, args2):
    classes, features, n_classes, n_features, isCat, epsilon_set, canNegative, classPossibilities, dataset_name = args2
    if isinstance(n_visits_lb, int):
        n_visits_lb = np.ones(len(classes))*n_visits_lb
        n_visits_ub = np.ones(len(classes))*n_visits_ub
    n_visits = np.zeros(len(classes))
    samples = sample_set.copy()
    init_preds = model.predict_proba(samples)
    preds = []
    visited_samples = []
    k = n_f_e # number of features to explore
    for i in init_preds:
        preds.append(np.argmax(i))
    for i in samples:
        visited_samples += [i] #.tolist()]
    query = 1
    epsilon = 1
    isPassed = [n_visits[i] >= n_visits_lb[i] for i in range(len(n_visits_lb))]
    while len(samples) != 0 and not all(isPassed) and not query > upper_limit:
        # 1. Print the information about the current sample
        curr = samples.pop(0)
        pred = model.predict_proba([curr])#.astype(int)[0]
        class_index = np.argmax(pred)
        #query += 1
        # No need to further visit overly explored sample classes
        if n_visits[class_index] < n_visits_ub[class_index]:
            query += 1
            n_visits[class_index] += 1
            preds += [classes[class_index]]
            
            visited_samples += [curr]      
            # 2. Get the explanation about the current sample
            exp = explainer.explain_instance(curr, model.predict_proba)#, top_labels=1)
            exp_map = exp.as_map()
            key = list(exp_map.keys())[0]
            exp_list = exp.as_list(key)
            exp_parsed = explanation_parser(exp_map,exp_list,key,features)

            # 3. Generate new samples and check if they were visited before
            tmp_exps, indices, cpys = [], [], []
            for i in range(k):
                tmp_exps += [exp_parsed[i]]
            for i in range(k):
                #print(tmp_exps[i][0])
                indices += [index for index, content in enumerate(features) if tmp_exps[i][0] in content]#[0]
            for i in range(2*k):
                cpys += [np.copy(curr)]
            for i in range(k):
                cpys[2*i][indices[i]] = tmp_exps[i][2] 
                cpys[2*i+1][indices[i]] = tmp_exps[i][1]
            for i in range(2*k):
                ind_i = int(i/2)
                #tmp = (any((cpys[i]==x).all() for x in visited_samples) or any((cpys[i]==x).all() for x in samples))
                #if (tmp and (cpys[i][indices[ind_i]] >= 0)):
                if (cpys[i][indices[ind_i]] >= 0):
                    if i%2 == 0:   
                        cpys[i][indices[ind_i]] += epsilon
                    else:
                        cpys[i][indices[ind_i]] -= epsilon
                tmp = (any((cpys[i]==x).all() for x in visited_samples) or 
                      (any((cpys[i]==x).all() for x in samples)) or
                      (cpys[i][indices[ind_i]] < 0) or #and isCat[indices[ind_i]] or
                      (cpys[i][indices[ind_i]] >= classPossibilities[indices[ind_i]]))
                if not tmp:
                    samples += [cpys[i]]#.tolist()]
    return visited_samples, preds, query

def traverse_explanations_SHAP(sample_set, explainer, model, n_visits_lb, n_visits_ub, upper_limit, n_f_e, args2, model_name):
    classes, features, n_classes, n_features, isCat, epsilon_set, canNegative, classPossibilities, dataset_name = args2
    if isinstance(n_visits_lb, int):
        n_v_lb = np.ones(len(classes))*n_visits_lb
        n_v_ub = np.ones(len(classes))*n_visits_ub
    n_visits = np.zeros(len(classes))
    samples = sample_set.copy()
    init_preds = model.predict_proba(samples)
    preds = []
    visited_samples = []
    for i in init_preds:
        preds.append(np.argmax(i))
    for i in samples:
        visited_samples += [i]
    query = 1
    isPassed = [n_visits[i] >= n_v_lb[i] for i in range(len(n_v_lb))]
    while len(samples) != 0 and not all(isPassed) and not query > upper_limit:
        # 1. Print the information about the current sample
        query += 1
        curr = samples.pop(0)
        pred = model.predict_proba([curr])#.astype(int)[0]
        class_index = np.argmax(pred)
        if query % 100 == 0:
            print(int(query/100), end =" ")
        # No need to further visit overly explored sample classes
        if n_visits[class_index] < n_v_ub[class_index]:
            #query += 1
            n_visits[class_index] += 1
            preds += [classes[class_index]]
            visited_samples += [curr]
            # 2. Get the SHAP explanation
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                if model_name == 'dt' or model_name == 'rdf':
                    exp = explainer.shap_values(curr)[:,class_index]
                else:
                    exp = explainer.shap_values(curr)
            k = min(np.count_nonzero(exp),n_f_e) #k = n_f_e
            sort_index = np.flip(np.argsort(abs(exp)))[:k]
            cpys = []
            oldOption = False
            if oldOption:
                # 2.1. Old version with single feature changing
                for i in range(2*k):
                    cpys += [np.copy(curr)]
                for i in range(k):
                    cpys[2*i][sort_index[i]]   += epsilon_set[sort_index[i]] 
                    cpys[2*i+1][sort_index[i]] -= epsilon_set[sort_index[i]]
                for i in range(2*k):
                    ind_i = int(i/2)
                    tmp = (any((cpys[i]==x).all() for x in visited_samples) or 
                          (any((cpys[i]==x).all() for x in samples)) or
                          (cpys[i][sort_index[ind_i]] < 0) or
                          (cpys[i][sort_index[ind_i]] >= classPossibilities[sort_index[ind_i]]))
                    if not tmp:
                        samples += [cpys[i]]
            else:
                # 2.2 New version with k features changing at the same time
                for i in range(2):
                    cpys += [np.copy(curr)]
                for i in range(k):
                    num = random.random()
                    tmp0 = cpys[0][sort_index[i]] + epsilon_set[sort_index[i]]
                    tmp1 = cpys[0][sort_index[i]] - epsilon_set[sort_index[i]]
                    if not isCat[sort_index[i]]: 
                        cond1 = True
                    else:
                        cond1 = (tmp0 < classPossibilities[sort_index[i]])
                    cond2 = (tmp1 >= 0)
                    if num < 0.8:
                        if cond1:
                            cpys[0][sort_index[i]] += epsilon_set[sort_index[i]]
                            if cond2:
                                cpys[1][sort_index[i]] -= epsilon_set[sort_index[i]]
                        else:
                            cpys[0][sort_index[i]] -= epsilon_set[sort_index[i]]
                            if cond1:
                                cpys[1][sort_index[i]] += epsilon_set[sort_index[i]]
                    else:
                        if cond1:
                            cpys[1][sort_index[i]] += epsilon_set[sort_index[i]]
                            if cond2:
                                cpys[0][sort_index[i]] -= epsilon_set[sort_index[i]]
                        else:
                            cpys[1][sort_index[i]] -= epsilon_set[sort_index[i]]
                            if cond1:
                                cpys[0][sort_index[i]] += epsilon_set[sort_index[i]]
                for i in range(2):
                    tmp = (any((cpys[i]==x).all() for x in visited_samples) or 
                          (any((cpys[i]==x).all() for x in samples)))
                    if not tmp:
                        samples += [cpys[i]]

    return visited_samples, preds, query

def decode_pred(target, v_preds):
    v_pred_dec = np.zeros(len(v_preds))
    for i in range(len(v_preds)):
        for j in range(len(target)):
            if v_preds[i] == target[j]:#+1]:
                v_pred_dec[i] = j#+1
    return v_pred_dec

def mega_sample_generation(testx, testy, n, sizes, n_set):
    test_cpy = []
    for i in range(len(testy)):                        
        test_cpy += [np.append(testx[i],testy[i])]
    test_cpy = np.array(test_cpy)
    samples_mega = []
    for i in range(n_set):
        sample_sets = []
        sample_set_sui = []
        for j in range(len(sizes)):
            sample_set_sui = sample_set_generation(test_cpy, n, sizes[j])
            sample_sets += [sample_set_sui.copy()]
        samples_mega += [sample_sets]
    return samples_mega

def rtest_sim(shadow, target, test_data):
    count = 0
    shadow_result = shadow.predict_proba(test_data)
    target_result = target.predict_proba(test_data)
    for i in range(len(shadow_result)):
        #if np.argmax(shadow.predict_proba([i])) == np.argmax(target.predict_proba([i])):
        if np.argmax(shadow_result[i]) == np.argmax(target_result[i]):
            count += 1
    return count/len(test_data)

# Change the addresses to prevent overriding
def pickling(dataset, modelName, accuracies, rtest_sims, samples_mega):
    if explanation_tool == 1:
        tool = "SHAP"
    else:
        tool = "LIME"
        
    address_acc   = tool + "/_models/" + modelName + "/" + modelName + "_accuracies_" + dataset
    address_sim   = tool + "/_models/" + modelName + "/" + modelName + "_similarities_" + dataset
    address_smega = tool + "/_models/" + modelName + "/" + modelName + "_samples_mega_" + dataset
    
    acc_file = open(address_acc, 'wb')
    pickle.dump(accuracies, acc_file)
    acc_file.close()
    sim_file = open(address_sim, 'wb')
    pickle.dump(rtest_sims, sim_file)
    acc_file.close()
    smega_file = open(address_smega, 'wb')
    pickle.dump(samples_mega, smega_file)
    smega_file.close()
    
def unpickling(dataset, modelName):
    if explanation_tool == 1:
        tool = "SHAP"
    else:
        tool = "LIME"
        
    address_acc   = tool + "/_models/" + modelName + "/" + modelName + "_accuracies_" + dataset
    address_sim   = tool + "/_models/" + modelName + "/" + modelName + "_similarities_" + dataset
    address_smega = tool + "/_models/" + modelName + "/" + modelName + "_samples_mega_" + dataset
        
    acc_file = open(address_acc, 'rb')
    accs = pickle.load(acc_file)
    acc_file.close()
    sim_file = open(address_sim, 'rb')
    sims = pickle.load(sim_file)
    acc_file.close()
    smega_file = open(address_smega, 'rb')
    samples_mega = pickle.load(smega_file)
    smega_file.close()
    return accs, sims, samples_mega

def argmaxing(accs, rss, args4): # Select the most similar model up until given query limit
    how_many_sets, sample_set_sizes, nfe, query_limit = args4
    ql = len(query_limit)
    argmax_acc = accs.copy()
    argmax_sim = rss.copy()
    for i in range(1,ql):
        for j in range(max(len(nfe),len(sample_set_sizes))):
            idx = (ql * j) + i
            for k in range(how_many_sets):
                if argmax_sim[idx][k] < argmax_sim[idx-1][k]:
                    argmax_acc[idx][k] = argmax_acc[idx-1][k]
                    argmax_sim[idx][k] = argmax_sim[idx-1][k]
    return argmax_acc, argmax_sim

def load_dataset(which_dataset):
    if which_dataset == 0:
        #iris = sklearn.datasets.load_iris()
        #X = iris.data
        #y = iris.target
        X, y = shap.datasets.iris()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)#, random_state=42)
        X_test_t, X_test_s, y_test_t, y_test_s = train_test_split(X_test, y_test, train_size=0.60, random_state=21)
        #features = iris.feature_names
        features = list(X.columns)
        classes = [0,1,2]#iris.target_names
        n_features = len(features)
        n_classes = len(classes)
        targets = dict({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        dataset_name = 'iris'
        isCategorical = [False]*n_features
        canNegative = [False]*n_features
        epsilon_set = [0.828, 0.436, 1.765, 0.762]
        epsilon_set = [x//4 for x in epsilon_set]
        #epsilon_set = [1]*n_features
        
    elif which_dataset == 1:
        n_crops = 17
        crop = pd.read_csv('data/crop/Crop_recommendation.csv') # Dataset 2
        #crop = crop[0:(n_crops*100)]
        crop.drop(crop.index[1800:1900], inplace=True)
        crop.drop(crop.index[1400:1500], inplace=True)
        crop.drop(crop.index[1000:1100], inplace=True)
        crop.drop(crop.index[800:900], inplace=True)
        crop.drop(crop.index[200:300], inplace=True)
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
        label_encoder = LabelEncoder()
        for col in features:
            label_encoder.fit(crop[col])
            crop[col] = label_encoder.transform(crop[col])
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']    
        X = crop[features]
        y = crop['label'].to_numpy()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        X_test_t, X_test_s, y_test_t, y_test_s = train_test_split(X_test, y_test, train_size=0.60, stratify=y_test, random_state=21)
        
        n_features = len(features)
        classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        n_classes = len(classes)
        dataset_name = 'crop'
        isCategorical = [False]*n_features
        canNegative = [False]*n_features
        epsilon_set = [36.26, 34.17, 56.48, 5.34, 19.98, 0.79, 54.04]
        epsilon_set = [x//4 for x in epsilon_set]
        
    if which_dataset == 2:
        X, y = shap.datasets.adult()
        #Preprocessing
        X.iloc[:,2] -= 1
        X['Country'] = np.where(X['Country'] == 39, 1, 0)
        X['Capital Gain'] = X['Capital Gain'] - X['Capital Loss'] + 4356
        X = X.drop(['Capital Loss'], axis=1)
        X.rename(columns={'Capital Gain': 'Net Capital'}, inplace=True)
        y = y.astype(int)
        
        X_display, y_display = shap.datasets.adult(display=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        X_test_t, X_test_s, y_test_t, y_test_s = train_test_split(X_test, y_test, train_size=0.60, random_state=21)
        
        classes = [0,1]
        features = list(X.columns)
        n_features = len(features)
        n_classes = len(classes)
        epsilon_set = [20,2,4,2,4,2,2,1,3000,5,1]
        isCategorical = [False, True, True, True, True, True, True, True, False, False, True]
        canNegative = [False,False,False,False,False,False,False,False,True,False,False]
        dataset_name = 'adult'
    
    elif which_dataset == 3:
        from sklearn.datasets import load_breast_cancer
        bc = load_breast_cancer()
        X = pd.DataFrame(bc.data)
        y = bc.target
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.25, stratify=bc.target)
        X_test_t, X_test_s, y_test_t, y_test_s = sklearn.model_selection.train_test_split(X_test, y_test, train_size=0.60, stratify=y_test)
        features = bc.feature_names
        classes = [0,1]
        n_features = len(features)
        n_classes = len(classes)
        targets = dict(enumerate(bc.target_names))
        isCategorical = [False]*n_features
        canNegative = [False]*n_features
        epsilon_set = list(X.std())
        #epsilon_set = [x//4 for x in epsilon_set]
        #epsilon_set = [1]*n_features
        dataset_name = 'breast'
        
    elif which_dataset == 4:
        nursery = pd.read_csv('data/nursery/nursery.csv')
        nursery[nursery == '?'] = np.nan
        #nursery[nursery['final evaluation']>=2]
    
        features = list(nursery.columns) #['parents','has_nurs','form','children','housing','finance','social','health','final evaluation']
        label_encoder = LabelEncoder()
        for col in features:
            label_encoder.fit(nursery[col])
            nursery[col] = label_encoder.transform(nursery[col])
    
        nursery.loc[nursery['final evaluation'] >= 2, 'final evaluation'] = 2
        features = nursery.columns[:-1]
        X = nursery[features]
        y = nursery['final evaluation'].to_numpy()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        X_test_t, X_test_s, y_test_t, y_test_s = train_test_split(X_test, y_test, train_size=0.60, random_state=21)
        targets = dict({0: 'not_recom', 1: 'recommend', 2: 'very_recom'})
        classes = [0,1,2]
        n_features = len(features)
        n_classes = len(classes)
        isCategorical = [True]*n_features
        epsilon_set = [1]*n_features
        canNegative = [False]*n_features
        dataset_name = 'nursery'
        
    elif which_dataset == 5:
        mushroom = pd.read_csv('data/mushroom/mushroom_data.csv')
        mushroom[mushroom == '?'] = np.nan
        mushroom = mushroom.drop(mushroom.columns[16],axis=1)
    
        features = list(mushroom.columns)
        label_encoder = LabelEncoder()
        for col in features:
            label_encoder.fit(mushroom[col])
            mushroom[col] = label_encoder.transform(mushroom[col])
    
        features = mushroom.columns[1::]
        X = mushroom[features]
        y = mushroom['p'].to_numpy()
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        X_test_t, X_test_s, y_test_t, y_test_s = train_test_split(X_test, y_test, train_size=0.60, random_state=21)
        targets = dict({0: 'not_poisonous', 1: 'poisonous'})
        classes = [0,1]
        n_features = len(features)
        n_classes = len(classes)
        isCategorical = [True]*n_features
        epsilon_set = [1]*n_features
        canNegative = [False]*n_features
        dataset_name = 'mushroom'

    classPossibilities = []
    for i in range(n_features):
        uniques, counts = np.unique(X_train.iloc[:,i], return_counts=True)
        classPossibilities.append(len(uniques))

    args1 = [X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s]
    args2 = [classes, features, n_classes, n_features, isCategorical, epsilon_set, canNegative, classPossibilities, dataset_name]
    return args1, args2

def load_model(which_model, X_train, y_train):
    if which_model == 0:
        depth = 15
        t_model = dt(max_depth=depth, random_state=101).fit(X_train.values, y_train)    
        model_name = 'dt'
    elif which_model == 1:
        #t_model = lr(solver='sag', random_state=101, max_iter=10000).fit(X_train.values, y_train)
        t_model = lr(random_state=101, max_iter=10000).fit(X_train.values, y_train)
        model_name = 'lr'
    elif which_model == 2:
        t_model = mnb().fit(X_train.values, y_train)
        model_name = 'nb'
    elif which_model == 3:
        n_classes = len(np.unique(y_train, return_counts=True)[0])
        t_model = knn(n_neighbors=n_classes).fit(X_train.values, y_train)
        model_name = 'knn'
    elif which_model == 4:
        depth = 15
        t_model = rf(max_depth=depth, random_state=101).fit(X_train.values, y_train)
        model_name = 'rdf'
    elif which_model == 5:
        t_model = mlp(hidden_layer_sizes = (20,), activation='relu', solver='adam', max_iter=10000, random_state=101).fit(X_train.values, y_train)
        t_model = mlp(activation='relu', solver='adam', max_iter=10000, random_state=101).fit(X_train.values, y_train)
        model_name = 'mlp'
    else:
        print('No such model exists!')

    return t_model, model_name

def load_explainer(explanation_tool, t_model, model_name, X_train):
    if explanation_tool == 1:
        shap.initjs()
        if model_name == 'dt' or model_name == 'rdf':
            t_explainer = shap.Explainer(t_model)
        else:
            f = lambda x: t_model.predict_proba(x)[:,1]
            med = X_train.median().values.reshape((1,X_train.shape[1]))
            t_explainer = shap.KernelExplainer(f, med, normalize=False)
            
            # -- Slower but more precise version --
            # f = lambda x: t_model.predict_proba(x)
            # t_explainer = shap.KernelExplainer(f, X_train)
    else:
        t_explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, discretize_continuous=True)
    return t_explainer

def getModelInfo(t_model, X_train, y_train, X_test_t, y_test_t):
    predict_train = t_model.predict(X_train.values)
    predict_test = t_model.predict(X_test_t.values)
    print('Train results')
    print(confusion_matrix(y_train,predict_train))
    print(classification_report(y_train,predict_train))
    print('Test results')
    print(confusion_matrix(y_test_t,predict_test))
    print(classification_report(y_test_t,predict_test))
    #if model_name == 'dt':
    #    print(features)
    #    text_representation_t = tree.export_text(t_model)
    #    print("Target\n",text_representation_t)
    predictions = t_model.predict(X_test_t.values)
    t_accuracy = round(accuracy_score(y_test_t, t_model.predict(X_test_t.values)),4)
    print('Model test accuracy: ',t_accuracy,'\n')
    return t_accuracy

def load_experiment_dicts():
    dataset_dict = {0: 'Iris',  1: 'Crop', 2: 'Adult Income', 3: 'Breast Cancer', 4: 'Nursery', 5: 'Mushroom'}
    model_dict = {0: 'Decision Tree',1: 'Logistic Regression',2: 'Multinomial Naive Bayes', 3: 'K Nearest Neighbor', 4: 'Random Forest', 5: 'Multilayer Perceptron'}
    exp_dict = {0: 'LIME', 1: 'SHAP'}
    return dataset_dict, model_dict, exp_dict

def run_attack_auto(wd, wm, et, hms, sss, nfe, ql, so): # make sure the types are correct
    which_dataset    = wd  if isinstance(wd,   int) else (lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    which_model      = wm  if isinstance(wm,   int) else (lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    explanation_tool = et  if isinstance(et,   int) else (lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    how_many_sets    = hms if isinstance(hms,  int) else (lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    sample_set_sizes = sss if isinstance(sss, list) else (lambda: (_ for _ in ()).throw(TypeError("Only lists are allowed")))()
    nfe              = nfe if isinstance(nfe, list) else (lambda: (_ for _ in ()).throw(TypeError("Only lists are allowed")))()
    query_limit      = ql  if isinstance(ql,  list) else (lambda: (_ for _ in ()).throw(TypeError("Only lists are allowed")))()
    save_option      = so  if isinstance(so,  bool) else (lambda: (_ for _ in ()).throw(TypeError("Only booleans are allowed")))()

    ## Unpack args
    args1, args2 = load_dataset(which_dataset)
    X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s = args1
    classes, features, n_classes, n_features, isCategorical, epsilon_set, canNegative, classPossibilities, dataset_name = args2
    t_model, model_name = load_model(which_model, X_train, y_train)
    t_accuracy = getModelInfo(t_model, X_train, y_train, X_test_t, y_test_t)
    t_explainer = load_explainer(explanation_tool, t_model, model_name, X_train)
    dataset_dict, model_dict, exp_dict = load_experiment_dicts()
    print('Dataset:  ', dataset_dict.get(which_dataset))
    print('ML Model: ', model_dict.get(which_model))
    print(exp_dict.get(explanation_tool),'is the explanation tool currently in use\n')
    
    ## Here goes the attack!!! 
    ## (You can further configure the parameters like lower and upper bounds, relax_factor etc. at your own risk :/) 

    samples_mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, n_classes, sample_set_sizes, how_many_sets)
    #samples_mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s.to_numpy(), n_classes, sample_set_sizes, how_many_sets)
    
    accuracies = []
    rtest_sims = []
    prioritizeSim = True
    
    if model_name == 'nb' or model_name == 'mlp' or model_name == 'lr' or model_name == 'knn':
        repetition = 1
    elif model_name == 'dt':
        repetition = 100
    else:
        repetition = 10
    
    relax_factor = 0.5
    lb_set = list(map(lambda x: int((x//n_classes)*(1-relax_factor)+1), query_limit))
    ub_set = list(map(lambda x: int((x//n_classes)*(n_classes+relax_factor)+1), query_limit))
    depth = 15
    print('Lower bounds: ', lb_set, ' Upper bounds: ', ub_set, '(per class for both)\n')
    print('-----The attack starts here!-----\n')
    
    # 1. Generate sample sets
    for f in nfe:
        print('Number of top features allowed to be explored (k):', f)
        for g in range(len(sample_set_sizes)):
            print('\nNumber of samples per class (n):', sample_set_sizes[g])
            max_sim = [False]*how_many_sets
            for h in range(len(lb_set)):                #Queries
                lb, ub = lb_set[h], ub_set[h]
                real_accuracy = []
                sims = []
                for i in range(how_many_sets):                     #Sample Sets
                    if max_sim[i]:
                        print('Sample set',i," Max similarity reached, no need for traversal!")
                        sims += [1]
                        real_accuracy += [t_accuracy]
                    else:
                        if explanation_tool == 0:
                            v_samples_np, v_pred_dec, n_query = traverse_explanations_LIME(samples_mega[i][g], t_explainer, t_model, lb, ub, query_limit[h], f, args2)
                        elif explanation_tool == 1:
                            v_samples_np, v_pred_dec, n_query = traverse_explanations_SHAP(samples_mega[i][g], t_explainer, t_model, lb, ub, query_limit[h], f, args2, model_name)
                        else:
                            print('No valid explanation tool selected')
                            break
                        s_accuracy = []
                        sim = []
                        for k in range(repetition):             #Model building
                            if model_name == 'dt':
                                s_model = dt(random_state=k, max_depth=depth)
                            elif model_name == 'lr':
                                #s_model = lr(solver='sag', max_iter=10000,random_state=k)
                                s_model = lr(max_iter=1000, random_state=k)
                            elif model_name == 'nb':
                                s_model = mnb()
                            elif model_name == 'rdf':
                                s_model = rf(max_depth=depth,random_state=k)
                            elif model_name == 'knn':
                                s_model = knn(n_neighbors=n_classes)
                            elif model_name == 'mlp':
                                #This part is added later
                                models = []
                                for layer in range(10):
                                    l = layer + 1
                                    models += [mlp(activation='tanh', hidden_layer_sizes=(10*l), solver='adam',  max_iter=10000)]
                                    models += [mlp(activation='relu', hidden_layer_sizes=(10*l), solver='adam',  max_iter=10000)]
                                    #models += [mlp(activation='tanh', hidden_layer_sizes=(10*l,10*l), solver='adam', max_iter=10000)]
                                    #models += [mlp(activation='relu', hidden_layer_sizes=(10*l,10*l), solver='adam', max_iter=10000)]
                                #models = [s_model, s_model1]
                                #models = [s_model, s_model1, s_model2, s_model3]
                            else:
                                print('No such model!')
                            if model_name == 'mlp':
                                for m in models:
                                    m.fit(v_samples_np, v_pred_dec)
                                    sim += [rtest_sim(m, t_model, X_test_t.values)]
                                    s_accuracy += [accuracy_score(y_test_t, m.predict(X_test_t.values))]
                            else:
                                s_model.fit(v_samples_np, v_pred_dec)
                                sim += [rtest_sim(s_model, t_model, X_test_t.values)]
                                s_accuracy += [accuracy_score(y_test_t, s_model.predict(X_test_t.values))]
                        if prioritizeSim:
                            tmp = np.argmax(sim)
                        else:
                            tmp = np.argmax(s_accuracy)
                        m_sim = round(sim[tmp],4)
                        sims += [m_sim]
                        real_accuracy += [round(s_accuracy[tmp],4)]
                        if m_sim == 1:
                            max_sim[i] = True
                        print('Sample set',i,', n_queries =',len(v_pred_dec), ', Top similarity =', m_sim)
                accuracies += [real_accuracy]
                rtest_sims += [sims]
                print("Accuracy: ",real_accuracy,"\nSimilarity: ",sims,"\n")

    # Pack up all remaining variables for checking
    args0 = [which_dataset, which_model, explanation_tool]
    args3 = [t_model, model_name, t_accuracy, t_explainer]
    args4 = [how_many_sets, sample_set_sizes, nfe, query_limit]
    other_args = [args0, args1, args2, args3, args4]
    
    accuracies, rtest_sims = argmaxing(accuracies, rtest_sims, args4) # Max similarity surrogate model is preserved
    if save_option:
        save_results(dataset_name, model_name, accuracies, rtest_sims, samples_mega)
    
    return accuracies, rtest_sims, samples_mega, other_args

def run_attack_auto_v2(wd, wm, et, hms, sss, nfe, ql, so): # make sure the types are correct
    which_dataset    = wd  if isinstance(wd,   int) else (lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    which_model      = wm  if isinstance(wm,   int) else (lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    explanation_tool = et  if isinstance(et,   int) else (lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    how_many_sets    = hms if isinstance(hms,  int) else (lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    sample_set_sizes = sss if isinstance(sss, list) else (lambda: (_ for _ in ()).throw(TypeError("Only lists are allowed")))()
    nfe              = nfe if isinstance(nfe, list) else (lambda: (_ for _ in ()).throw(TypeError("Only lists are allowed")))()
    query_limit      = ql  if isinstance(ql,  list) else (lambda: (_ for _ in ()).throw(TypeError("Only lists are allowed")))()
    save_option      = so  if isinstance(so,  bool) else (lambda: (_ for _ in ()).throw(TypeError("Only booleans are allowed")))()

    ## Unpack args
    args1, args2 = load_dataset(which_dataset)
    X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s = args1
    classes, features, n_classes, n_features, isCategorical, epsilon_set, canNegative, classPossibilities, dataset_name = args2
    t_model, model_name = load_model(which_model, X_train, y_train)
    t_accuracy = getModelInfo(t_model, X_train, y_train, X_test_t, y_test_t)
    t_explainer = load_explainer(explanation_tool, t_model, model_name, X_train)
    dataset_dict, model_dict, exp_dict = load_experiment_dicts()
    print('Dataset:  ', dataset_dict.get(which_dataset))
    print('ML Model: ', model_dict.get(which_model))
    print(exp_dict.get(explanation_tool),'is the explanation tool currently in use\n')
    
    ## Here goes the attack!!! 
    ## (You can further configure the parameters like lower and upper bounds, relax_factor etc. at your own risk :/) 
    samples_mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, n_classes, sample_set_sizes, how_many_sets)
    #samples_mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s.to_numpy(), n_classes, sample_set_sizes, how_many_sets)
    
    rows, cols, dims = len(query_limit), how_many_sets, len(nfe)
    # Create a 3D matrix with dimensions (dims, rows, cols) initialized to 0
    accuracies = [[[0 for _ in range(cols)] for _ in range(rows)] for _ in range(dims)]
    rtest_sims = [[[0 for _ in range(cols)] for _ in range(rows)] for _ in range(dims)]
    prioritizeSim = True
    
    if model_name == 'nb' or model_name == 'mlp' or model_name == 'lr' or model_name == 'knn':
        repetition = 1
    elif model_name == 'dt':
        repetition = 100
    else:
        repetition = 10
    
    relax_factor = 0.8
    lb_set = list(map(lambda x: int((x//n_classes)*(1-relax_factor)+1), query_limit))
    ub_set = list(map(lambda x: int((x//n_classes)*(n_classes+relax_factor)+1), query_limit))
    depth = 15
    print('Lower bounds: ', lb_set, ' Upper bounds: ', ub_set, '\n')
    print('-----The attack starts here!-----\n')
    
    # 1. Generate sample sets
    for f in nfe:
        f_idx = nfe.index(f)
        print(f_idx)
        print('Number of top features allowed to be explored (k):', f)
        for g in range(len(sample_set_sizes)):
            print('\nNumber of samples per class (n):', sample_set_sizes[g])
            for i in range(how_many_sets):                     #Sample Sets
                if explanation_tool == 0:
                    v_samples_np, v_pred_dec, n_query = traverse_explanations_LIME(samples_mega[i][g], t_explainer, t_model, lb_set[-1], ub_set[-1], query_limit[-1], f, args2)
                elif explanation_tool == 1:
                    v_samples_np, v_pred_dec, n_query = traverse_explanations_SHAP(samples_mega[i][g], t_explainer, t_model, lb_set[-1], ub_set[-1], query_limit[-1], f, args2, model_name)
                else:
                    print('No valid explanation tool selected')
                    break    
                for h in range(len(query_limit)):    
                    data_x = v_samples_np[:query_limit[h]]
                    data_y = v_pred_dec[:query_limit[h]]
                    s_accuracy = []
                    sim = []
                    for k in range(repetition):             #Model building
                        if model_name == 'dt':
                            s_model = dt(random_state=k, max_depth=depth)
                        elif model_name == 'lr':
                            #s_model = lr(solver='sag', max_iter=10000,random_state=k)
                            s_model = lr(max_iter=1000, random_state=k)
                        elif model_name == 'nb':
                            s_model = mnb()
                        elif model_name == 'rdf':
                            s_model = rf(max_depth=depth,random_state=k)
                        elif model_name == 'knn':
                            s_model = knn(n_neighbors=n_classes)
                        elif model_name == 'mlp':
                            #This part is added later
                            models = []
                            for layer in range(10):
                                l = layer + 1
                                models += [mlp(activation='tanh', hidden_layer_sizes=(10*l), solver='adam',  max_iter=10000)]
                                models += [mlp(activation='relu', hidden_layer_sizes=(10*l), solver='adam',  max_iter=10000)]
                        else:
                            print('No such model!')
                        if model_name == 'mlp':
                            for m in models:
                                m.fit(data_x, data_y)
                                sim += [rtest_sim(m, t_model, X_test_t.values)]
                                s_accuracy += [accuracy_score(y_test_t, m.predict(X_test_t.values))]
                        else:
                            s_model.fit(data_x, data_y)
                            sim += [rtest_sim(s_model, t_model, X_test_t.values)]
                            s_accuracy += [accuracy_score(y_test_t, s_model.predict(X_test_t.values))]
                    if prioritizeSim:
                        tmp = np.argmax(sim)
                    else:
                        tmp = np.argmax(s_accuracy)
                    m_sim = round(sim[tmp],4)
                    if m_sim == 1:
                        max_sim[i] = True
                    accuracies[f_idx][h][i] = round(s_accuracy[tmp],4)
                    rtest_sims[f_idx][h][i] = round(sim[tmp],4)
                print('Sample set',i,', n_queries =',len(v_pred_dec), ', Top similarity =', m_sim)
        print("Accuracy: ",accuracies,"\nSimilarity: ",rtest_sims,"\n")

    # Pack up all remaining variables for checking
    args0 = [which_dataset, which_model, explanation_tool]
    args3 = [t_model, model_name, t_accuracy, t_explainer]
    args4 = [how_many_sets, sample_set_sizes, nfe, query_limit]
    other_args = [args0, args1, args2, args3, args4]
    
    #accuracies, rtest_sims = argmaxing(accuracies, rtest_sims, args4) # Max similarity surrogate model is preserved
    if save_option:
        save_results(dataset_name, model_name, accuracies, rtest_sims, samples_mega)
    
    return accuracies, rtest_sims, samples_mega, other_args

def run_attack_prepared(isFast):
    if isFast:
        which_dataset    = 0 
        which_model      = 1 
        explanation_tool = 0
        how_many_sets    = 10
        sample_set_sizes = [1]
        nfe              = [3]
        query_limit      = [0,10,25,50,100]
    else:
        which_dataset    = 2 
        which_model      = 4 
        explanation_tool = 1
        how_many_sets    = 10
        sample_set_sizes = [5]
        nfe              = [3,5,7]
        query_limit      = [0,100,250,500,1000]

    return run_attack_auto(which_dataset, which_model, explanation_tool, how_many_sets, sample_set_sizes, nfe, query_limit, False)

# Save Results
def save_results(dataset_name, model_name, acs, rsims, smegas):
    try:
      pickling(dataset_name, model_name, acs, rsims, smegas)
      print('Save operation successful!')
    except:
      print('Save operation failed!')

def load_results(dataset_name, model_name):      # accuracies, rtest_sims, samples_mega = unpickling(dataset_name, model_name)
    return unpickling(dataset_name, model_name)

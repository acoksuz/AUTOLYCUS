from utils import *
from matplotlib import pyplot as plt

def main():
    which_dataset = 2    # 'iris = 0', 'crop = 1', 'adult = 2', 'breast = 3', 'nursery = 4', 'mushroom = 5'
    which_model = 1      # 'Decision_Tree = 0', 'Logistic_Regression = 1', 'Multinomial_Naive_Bayes = 2', 'K_Nearest_Neeighbor = 3', 'Random_Forest = 4', 'Multilayer_Perceptron = 5'
    explanation_tool = 0 # 'Lime = 0', 'Shap = 1'
    how_many_sets = 10 # This is the number of auxiliary datasets to experiment on. Lower the number for lowering the runtime.
    sample_set_sizes = [3] # (list of n values from the paper) Size of auxiliary dataset (per class) in a list format
    nfe = [1,3] #[1,3,5,7,9,11] # (list of k values from the paper) Number of features explored in a list format
    query_limit = [100,250,500,1000] # (list of Q values from the paper) Number of queries allowed for traversal

    print('This might take a bit of time!')
    accuracies, rtest_sims, samples_mega, other_args = run_attack_auto(which_dataset, which_model, explanation_tool, how_many_sets, sample_set_sizes, nfe, query_limit, False)
    print('Here starts the plotting')
    
    ## Plotting is pain \_(-_-)_/
    isSim = True
    data_s0, data_s1, data_s2, data_s3 = [], [], [], []
    dataset_dict, model_dict, exp_dict = load_experiment_dicts()
    k = len(nfe)
    q = len(query_limit)
    
    for i in range(k):
        data_s0 += [rtest_sims[q*i]]
        data_s1 += [rtest_sims[q*i+1]]
        data_s2 += [rtest_sims[q*i+2]]
        data_s3 += [rtest_sims[q*i+3]]
    
    data_s0 = np.mean(data_s0,axis=1)
    data_s1 = np.mean(data_s1,axis=1)
    data_s2 = np.mean(data_s2,axis=1)
    data_s3 = np.mean(data_s3,axis=1)
    
    x_axis = 3 #nfe.copy()
    y_axis = [0,0.2,0.4,0.6,0.8,1.0]
    
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    plt.title('Dataset: '+dataset_dict.get(which_dataset)+
              ',   Model Type: '+model_dict.get(which_model)+
              ',   Explainer: '+exp_dict.get(explanation_tool), color='black')
    ax = plt.subplot(111)
    ax.bar(x_axis-0.3, data_s0, width=0.2, color='b', align='center')
    ax.bar(x_axis-0.1, data_s1, width=0.2, color='g', align='center')
    ax.bar(x_axis+0.1, data_s2, width=0.2, color='r', align='center')
    ax.bar(x_axis+0.3, data_s3, width=0.2, color='y', align='center')

    x_axis = nfe.copy()
    ax.yaxis.grid(True)
    ax.set_xticklabels(x_axis,fontsize=12)
    ax.set_yticklabels(y_axis,fontsize=12)
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('Number of Features Explored (k)', fontsize=12)
    ax.set_ylabel('Similarity', fontsize=12)
    ax.autoscale(tight=True)
    plt.show()

if __name__ == "__main__":
    main()
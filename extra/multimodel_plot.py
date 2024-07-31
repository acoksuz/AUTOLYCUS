## This part has been changed for producing your own results

## ---IMPORTANT---
## 1. Check ALL the comments before running!!!

## 2. DO NOT RUN before you run all the model experiments listed below for a particular dataset!
##    Example: Run all the model experiments (except mlp) for Mushroom dataset on a fixed n first.

## 3. Make sure the parameters like explainer, n, k, query_limit are fixed across experiments for consistent results!!!

## To start, add this as a cell to the end of experiments.ipynb

#dataset_name = "select_your_dataset" 

model_names = ['Decision Tree', 'Logistic Regression', 'Naive Bayes', 'K-Nearest Neighbor', 'Random Forest']

accuracies_dt,  rtest_sims_dt,  samples_mega_dt  = unpickling(dataset_name, 'dt')
accuracies_lr,  rtest_sims_lr,  samples_mega_lr  = unpickling(dataset_name, 'lr')
accuracies_nb,  rtest_sims_nb,  samples_mega_nb  = unpickling(dataset_name, 'nb')
accuracies_knn, rtest_sims_knn, samples_mega_knn = unpickling(dataset_name, 'knn')
accuracies_rdf, rtest_sims_rdf, samples_mega_rdf = unpickling(dataset_name, 'rdf')

data_plot = rtest_sims_dt + rtest_sims_lr + rtest_sims_nb + rtest_sims_knn + rtest_sims_rdf

x_axis = [0,100,250,500,1000,0,100,250,500,1000,0,100,250,500,1000,0,100,250,500,1000,0,100,250,500,1000]
#x_axis = [0,10,25,50,100,0,10,25,50,100,0,10,25,50,100,0,10,25,50,100,0,10,25,50,100] # For Iris and smaller datasets
y_axis = [0,0.2,0.4,0.6,0.8,1.0]

plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
fig, ax = plt.subplots(figsize=(16, 3))
medianprops = dict(linewidth=1.5, color='blue')
ax.boxplot(data_plot,0,'', medianprops=medianprops)
ax.yaxis.grid(True)
ax.set_xticklabels(x_axis)

plt.axvline(x = 5.5, color = 'black')
plt.axvline(x = 10.5, color = 'black')
plt.axvline(x = 15.5, color = 'black')
plt.axvline(x = 20.5, color = 'black')

## IMPORTANT: Update n (number of initial surrogate samples per class), depending on your experiments!!
label = 'n = 5'

plt.annotate(label, 
             (13,0.26), 
             textcoords="offset points", # how to position the text
             xytext=(0,0), # distance from text to points (x,y)
             ha='center', fontsize=12, color='red')

for x in range(5):
    label = model_names[x]
    plt.annotate(label, # this is the text
                 (5*x+3,0.05), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(1,1.3), # distance from text to points (x,y)
                 ha='center', fontsize=12, color='red')
    
ax.set_ylim(0,1.05)
ax.set_xticklabels(x_axis,fontsize=12)
ax.set_yticklabels(y_axis,fontsize=12)
ax.set_xlabel('Number of Queries', fontsize=12)
ax.set_ylabel('Similarity', fontsize=12)
plt.show()

#saveAddress = "LIME/plots/_similarity_" + dataset_name + ".pdf"
#fig.savefig(saveAddress, bbox_inches='tight')

#saveAddress = "SHAP/plots/_similarity_" + dataset_name + ".pdf"
#fig.savefig(saveAddress, bbox_inches='tight')

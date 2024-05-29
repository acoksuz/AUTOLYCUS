## This part has been changed for producing your own results

## ---IMPORTANT---
## 1. Check ALL the comments before running!!!

## 2. DO NOT RUN before you run the multilayer perceptron experiments on all the datasets listed below!

## 3. Make sure the parameters like explainer, n, k, query_limit are fixed across experiments for consistent results!!!

## To start, add this as a cell to the end of experiments.ipynb


dataset_names = {'Breast Cancer Dataset':'breast', 
                 'Adult Income Dataset':'adult', 
                 'Nursery Dataset':'nursery', 
                 'Mushroom Dataset':'mushroom'}

for i in dataset_names.values():
    if dataset_name == 'breast':
        accuracies_breast, rtest_sims_breast, samples_mega_breast = unpickling(dataset_name, 'mlp')
    elif dataset_name == 'adult':
        accuracies_adult, rtest_sims_adult, samples_mega_adult = unpickling(dataset_name, 'mlp')
    elif dataset_name == 'nursery':
        accuracies_nursery, rtest_sims_nursery, samples_mega_nursery = unpickling(dataset_name, 'mlp')
    elif dataset_name == 'mushroom':
        accuracies_mushroom, rtest_sims_mushroom, samples_mega_mushroom = unpickling(dataset_name, 'mlp')

data_plot = rtest_sims_breast + rtest_sims_adult + rtest_sims_nursery + rtest_sims_mushroom

x_axis = [0,100,250,500,1000,0,100,250,500,1000,0,100,250,500,1000,0,100,250,500,1000]
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

i = 0
for x in dataset_names.keys():
    plt.annotate(x, # this is the text
                 (5*i+3,0.05), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(1,1.3), # distance from text to points (x,y)
                 ha='center', fontsize=12, color='red')
    i += 1
    
ax.set_ylim(0,1.05)
ax.set_xticklabels(x_axis,fontsize=12)
ax.set_yticklabels(y_axis,fontsize=12)
ax.set_xlabel('Number of Queries', fontsize=12)
ax.set_ylabel('Similarity', fontsize=12)
plt.show()


## Use one of the two lines depending on the explainer you used.

#saveAddress = "LIME/plots/_similarity_mlp" + dataset_name + ".pdf"
#fig.savefig(saveAddress, bbox_inches='tight')

#saveAddress = "SHAP/plots/_similarity_mlp" + dataset_name + ".pdf"
#fig.savefig(saveAddress, bbox_inches='tight')
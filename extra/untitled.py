data_s0, data_s1, data_s2, data_s3 = [], [], [], []
data_a0, data_a1, data_a2, data_a3 = [], [], [], []

for i in range(len(nfe)):
    data_s0 += [rtest_sims[len(query_limit)*i]]
    data_s1 += [rtest_sims[len(query_limit)*i+1]]
    data_s2 += [rtest_sims[len(query_limit)*i+2]]
    data_s3 += [rtest_sims[len(query_limit)*i+3]]
    data_a0 += [accuracies[len(query_limit)*i]]
    data_a1 += [accuracies[len(query_limit)*i+1]]
    data_a2 += [accuracies[len(query_limit)*i+2]]
    data_a3 += [accuracies[len(query_limit)*i+3]]
    
data_s0 = np.mean(data_s0,axis=1)
data_s1 = np.mean(data_s1,axis=1)
data_s2 = np.mean(data_s2,axis=1)
data_s3 = np.mean(data_s3,axis=1)
data_a0 = np.mean(data_a0,axis=1)
data_a1 = np.mean(data_a1,axis=1)
data_a2 = np.mean(data_a2,axis=1)
data_a3 = np.mean(data_a3,axis=1)
    
isSim = True

x_axis = 3#[1,3,5,7,9]
#y_axis = [0,0.2,0.4,0.6,0.8,1.0]

plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
ax = plt.subplot(111)
ax.bar(x_axis-0.3, data_s0, width=0.2, color='b', align='center')
ax.bar(x_axis-0.1, data_s1, width=0.2, color='g', align='center')
ax.bar(x_axis+0.1, data_s2, width=0.2, color='r', align='center')
ax.bar(x_axis+0.3, data_s3, width=0.2, color='y', align='center')

#ax.boxplot(data0,0,'',medianprops=medianprops)
#ax.xaxis.grid(True)
ax.yaxis.grid(True)
ax.set_xticklabels(x_axis)
if not isSim:
    plt.axhline(y = t_accuracy, color = 'green')
ax.set_xticklabels(x_axis,fontsize=12)
ax.set_yticklabels(y_axis,fontsize=12)
ax.set_ylim([0, 1.05])
ax.set_xlabel('Number of Features Explored (k)', fontsize=12)
if isSim:
    ax.set_ylabel('Similarity', fontsize=12)
else:
    ax.set_ylabel('Accuracy', fontsize=12)
ax.autoscale(tight=True)
plt.show()

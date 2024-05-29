# Artifact Appendix

Paper title: **AUTOLYCUS: Exploiting Explainable Artificial Intelligence (XAI) for Model Extraction Attacks against Interpretable Models**

Artifacts HotCRP Id: **#17**

Requested Badge: **Reproducible**

## Description
As an artifact, we provide a public GitHub repository which contains the source code, figures and datasets used in our paper. Its link is https://github.com/acoksuz/AUTOLYCUS

### Security/Privacy Issues and Ethical Concerns
There are no ethical concerns related to this artifact. 

## Basic Requirements
8 GB ram and 40 MB disk space are recommended for running the given attack.
Python notebook IDE (e.g. Jupyter) with Python 3.3 and higher is required to run the code.
Explanation calculation, sample set size and query limits determine time complexity of the code. For example, 1000 queries for 5 sample set sizes on SHAP can take a minute.

### Hardware Requirements
This artifact does not contain specific hardware requirements.

### Software Requirements
The experiments are conducted on Apple M1 Macbook Air (A2337) with macOS 14 Sonoma and 8 GB ram.
Python notebook IDE (e.g. Jupyter) with Python 3.3 and higher is required to run the code.
Dependent libraries in the code are lime, matplotlib, numpy, pickle, sklearn, scipy, seaborn, shap. 
The version numbers of these libraries are further specified in the requirements.txt of GitHub repository.
All used datasets are provided in the repository.

### Estimated Time and Storage Consumption
The repository is light-weight and requires only 32 MB of space. Depending on the query limits and the explainers, time constraints can change drastically. 
In most of our experiments, each dataset and sample set size list combination ran for 40 minutes on average. 
However, it can be drastically reduced by minimizing the number of queries and setting a single sample set size.  
Our observation is that LIME and SHAP explanations are returned faster on macOS or Linux.

## Environment
Python notebook IDE (e.g. Jupyter) with Python 3.3 and higher is required to run the code.
In a regular Python3 environment install the following libraries using pip or conda commands: lime, matplotlib, numpy, pickle, sklearn, scipy, seaborn, shap
We provided a commented script in the beginning of 'Experiments.ipynb' notebook. If any of the libraries is missing, the code can be uncommented and run for installation.

### Accessibility
This artifact and any of its future updates can be accessed on https://github.com/acoksuz/AUTOLYCUS/
The current commit-id of the repository is b329574.

### Set up the environment

1. Run the following commands on terminal to install up-to-date Python, Pip and Jupyter Notebook libraries. 
sudo apt update
sudo apt install python3
sudo apt install python3-pip
pip3 install notebook
jupyter notebook

3. Download the repository using the command
gh repo clone acoksuz/AUTOLYCUS

4. Run the following command to install required libraries used in the proposed attack
pip3 install lime matplotlib numpy pickle seaborn shap scipy sklearn

5. Run all the scripts in "Experiments.ipynb" from top to bottom.

### Testing the Environment

"Experiments.ipynb" notebook file is set to an example case to test the functionality of the attack. 
If the code does not give an error, it should print and plot the attack results and be good to go for further configuration. 
The cells available for user editing are highlighted with comments and in-comment !!! signs.

## Artifact Evaluation
In order to reproduce results displayed in the paper, following parameters are required to be set;

explanation_tool = 0 or 1 (LIME or SHAP)
n = 1, 5 (for LIME and SHAP respectively) and (sample_set_sizes = [1] and sample_set_sizes = [5] respectively) 
k = 3
query_limit = [0,100,250,500,1000] (for Crop, Adult Income, Nursery and Mushroom datasets) or
query_limit = [0,10,25,50,100] (for Iris and Breast Cancer datasets)

Higher n, k and query_limit values result with more similar extractions.

sample_set_sizes = [1,2,3,4,5] can be used for plotting (with the last cell in "Experiments.ipynb") a specific dataset-model combination.

### Main Results and Claims

The performance of AUTOLYCUS is influenced by the careful adjustment of the following attack configuration and model parameters: 
(i) ensuring a sufficiently high number of queries is sent,         
(ii) selecting an adequate number of features (𝑘) for perturbation,
guided by the distribution of feature importance;                   (less entropy is better)
(iii) incorporating sufficient number of (iv) informative auxiliary samples; 
(v) considering the complexity of the models -lower complexity models are extracted easier-, including any unknowns or parameters in neural network terminology; 
(vi) ensuring the accuracy of the model, 
(vii) optimizing the selection of 𝛿 values especially when SHAP is used as the XAI tool. 
These factors collectively contribute to the efficacy of AUTOLYCUS in adversarial settings. (Higher is better for almost all settings, except for complexity)

Thanks to additional information provided by the explainers, AUTOLYCUS requires less queries compared to SOTA model extraction attacks mentioned in the paper.
Additionally, LIME explanations provide higher utility in our particular attack model compared to SHAP explanations.

#### Main Result 1: Importance of number of features (k) on similarity
When n=3, k=[1,3,5,7,9], query_limit=[1000] and explanation_tool=0 for Adult Income Dataset, as k increases similarity (extraction capability) increases. It is in accordance with Figure 6.

#### Main Result 2: Impact of model complexity on similarity
When the attack configurations are kept the same, a Random Forest classifier trained on Adult Income dataset is extracted with less efficiency compared to a less complex Random Forest classifier trained on Nursery dataset. It can be verified with multiple results in Figure 4 and 5. 

#### Main Result 3: Impact of sample set size (n) and number of queries on similarity
When n is increased from 1 to 5 or query limits are increased from 100 to 1000, extractions get better due to more traversal. It is in accordance with all the Figures from 4 to 8. 

### Experiments
explanation_tool, n, k, query_limit are all configurable parameters. Except for explanation_tool, all of them can be initialized as a list to see their individual impacts on extractions.

#### Experiment 1: Impact of number of features (k) on similarity

Keep every parameter fixed while providing a list for k = [1,2,...,m] 
Expected result: As k increases, traversal algorithm perturbs more features. Perturbations with more features increase over-fitting to the decision boundaries of the target model, resulting with better extractions

#### Experiment 2: Impact of query limit on similarity 
Keep every parameter fixed while providing a list for query_limit = [0,100,250,500,1000]
Expected result: Intuitively, this is very similar to a regular search algorithm. As we traverse through the model more, we learn more about its decision boundaries. Learned decision boundaries is equivalent to extraction. 

#### Experiment 3: Impact of auxiliary dataset size (n) on similarity
Keep every parameter fixed while providing a list for sample_set_sizes = [1,2,3,4,5]
Expected result: Increasing the size of auxiliary data provides more and more diverse examples to explore for the traversal algorithm. Even without perturbation, these diverse examples act as a training data. Furthermore, they provide an opportunity to attack the target model from multiple angles instead of few angles which might keep certain decision boundaries unexplored. 


## Limitations
Exact traversal paths in the experiments are not reproducible due to the use of randomized perturbations.
Also, some auxiliary datasets (samples_mega in the code) can be more informative than others and yield better extractions. Since the informativeness of a dataset in ML classifiers is outside the scope of this paper, it is not reflected as a quantified metric in our paper. 

## Notes on Reusability
This artifact can be re-used to demonstrate the vulnerability of other ML models trained on other datasets. 
It can be further re-purposed to enhance the capabilities of other ML attacks like membership inference attacks and model inversion attacks.
With better tailored traversal techniques or use of auxiliary data, the query requirements can be decreased further.

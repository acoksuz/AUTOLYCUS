# Artifact Appendix

Paper title: **AUTOLYCUS: Exploiting Explainable Artificial Intelligence (XAI) for Model Extraction Attacks against Interpretable Models**

Artifacts HotCRP Id: **#17**

Requested Badge: **Reproducible**

## Description
As an artifact, we provide a public GitHub repository and a Docker image which contain the source code, figures and datasets used in our paper. 
The links are: \
https://github.com/acoksuz/AUTOLYCUS \
https://hub.docker.com/r/acoksuz/autolycus/tags

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
Dependent libraries and their version numbers are listed in "requirements.txt" of GitHub repository.
All used datasets are provided in the repository.

### Estimated Time and Storage Consumption
The repository is light-weight and requires only 32 MB of space. Depending on the query limits and the explainers, time constraints can change drastically. 
In most of our experiments, each dataset and sample set size list combination ran for 40 minutes on average. 
However, it can be drastically reduced by minimizing the number of queries and setting a single sample set size.  
Our observation is that LIME and SHAP explanations are returned faster on macOS or Linux.

## Environment
Python notebook IDE (e.g. Jupyter) with Python 3.3 and higher is required to run the code.
In a regular Python3 environment install the following libraries using pip or conda commands: lime, matplotlib, numpy, pickle, sklearn, scipy, seaborn, shap.
Use the following command for easy installation: pip3 install -r requirements.txt

### Accessibility
This artifact and any of its future updates can be accessed on https://github.com/acoksuz/AUTOLYCUS/
The current commit-id of the repository is 936611498be19c8a46cbc83fed794db8537f6527.

### Set up the environment

A. Docker Route

1. Please download and install Docker Desktop for this route. Then, run the following commands using terminal on the location where the Dockerfile is located. This will build and run a Docker image from the Dockerfile. \
docker build . --no-cache -t autolycus \
docker run -p 8888:8888 autolycus 

B. Manual Route

1. Run the following commands on terminal to install up-to-date Python, Pip and Jupyter Notebook libraries.  
sudo apt update \
sudo apt install python3 \
sudo apt install python3-pip \
pip3 install notebook \
jupyter notebook

2. Download the repository using the command
gh repo clone acoksuz/AUTOLYCUS

3. Run the following command to install required libraries used in the proposed attack
pip3 install -r requirements.txt

4. Run individual notebooks for experiments or for manual configurations run all the scripts in "Experiments.ipynb" from top to bottom.

### Testing the Environment

"Experiments.ipynb" notebook file is set to an example case to test the functionality of the attack. 
If the code does not give an error, it should print and plot the attack results and be good to go for further configuration. 
The cells are available for user editing but be way of comments and potential runtime issues. Some errors might be circumvented by executing multiple times during errors. The skippable errors occur due to random generation of datasets. Let us know of the consistent errors.  

## Artifact Evaluation
In order to reproduce results displayed in Figure 4 and 5 of the paper, following parameters are required to be set;

- explanation_tool = 0 or 1 (LIME or SHAP)
- n = 1, 5 (for LIME and SHAP respectively) and (sample_set_sizes = [1] and sample_set_sizes = [5] respectively) 
- k = 3
- query_limit = [0,100,250,500,1000] (for Crop, Adult Income, Nursery and Mushroom datasets) or
- query_limit = [0,10,25,50,100] (for Iris and Breast Cancer datasets)

Higher n, k and query_limit values result with more similar extractions.

sample_set_sizes = [1,2,3,4,5] can be used for plotting (with the last cell in "Experiments.ipynb") a specific dataset-model combination.

### Main Results and Claims

The performance of AUTOLYCUS is influenced by the careful adjustment of the following attack configuration and model parameters: (i) ensuring a sufficiently high number of queries is sent, (ii) selecting an adequate number of features (𝑘) for perturbation,guided by the distribution of feature importance; (less entropy is better) (iii) incorporating sufficient number of (iv) informative auxiliary samples; (v) considering the complexity of the models -lower complexity models are extracted easier-, including any unknowns or parameters in neural network terminology; (vi) ensuring the accuracy of the model, (vii) optimizing the selection of 𝛿 values especially when SHAP is used as the XAI tool. These factors collectively contribute to the efficacy of AUTOLYCUS in adversarial settings. (Higher is better for almost all settings, except for complexity)

Thanks to additional information provided by the explainers, AUTOLYCUS requires less queries compared to SOTA model extraction attacks mentioned in the paper.
Additionally, LIME explanations provide higher utility in our particular attack model compared to SHAP explanations.

#### Main Result 1: Importance of number of features (k) on similarity
When n=3, k=[1,3,5,7,9,11], query_limit=[1000] and explanation_tool=0 for Adult Income Dataset, as k increases similarity (extraction capability) increases. It is in accordance with Figure 6.

#### Main Result 2: Impact of model/dataset complexity on similarity
When the attack configurations are kept the same, a Random Forest classifier trained on Adult Income dataset is extracted with less efficiency compared to a less complex Random Forest classifier trained on Nursery dataset. It can be verified with multiple results in Figure 4 and 5. 

#### Main Result 3: Impact of number of queries (Q) on similarity
When query limits are increased from 100 to 1000, extractions get better due to more traversal. The result you'll get from experiment_3.ipynb is in accordance with Figure 4(e)'s Logistic Regression plot.

#### Main Result 4: Impact of auxiliary dataset size (n) on similarity
When n(s) are increased from 1 to 5, extractions get better due to more informed sample space traversal. The result you'll get from experiment_4.ipynb is in accordance with the type of plot used in Figure 7. Since Figure 4 and 5 are summarizing plots with fixed n, the impact of n is not visible there.

### Experiments
explanation_tool, n, k, query_limit are all configurable parameters. Except for explanation_tool, all of them can be initialized as a list to see their individual impacts on extractions.

#### Experiment 1: Impact of number of features (k) on similarity
Run "experiment_1.ipynb" and check the plot at the bottom. \
Keep every parameter fixed while providing a list for k = [1,2,...,m] 
Expected result: As k increases, traversal algorithm perturbs more features. Perturbations with more features increase over-fitting to the decision boundaries of the target model, resulting with better extractions.

#### Experiment 2: Impact of model/dataset complexity on similarity
Run "experiment_2.ipynb" and check the plot at the bottom. \
Expected result: Since Adult Income Dataset has more features and possible values for each feature, its solution space is larger than Nursery dataset. Therefore with the same amount of query, models trained on Nursery dataset will be extracted faster compared to their counterparts on Adult Income dataset. 

#### Experiment 3: Impact of number of queries (Q) on similarity
Run "experiment_3.ipynb" and check the plot at the bottom. \
Expected result: Increasing the number of queries allows better traversal of solution space and results with more informed and similar models to the target model.

#### Experiment 4: Impact of auxiliary dataset size (n) on similarity
Run "experiment_4.ipynb" and check the plot at the bottom.
Keep every parameter fixed while providing a list for sample_set_sizes = [1,2,3,4,5] \
Expected result: Increasing the size of auxiliary data provides more and more diverse examples to explore for the traversal algorithm. Even without perturbation, these diverse examples act as a training data. Furthermore, they provide an opportunity to attack the target model from multiple angles instead of few angles which might keep certain decision boundaries unexplored.

## Limitations
Exact traversal paths in the experiments are not reproducible due to the use of randomized perturbations.
Also, some auxiliary datasets (samples_mega in the code) can be more informative than others and yield better extractions. Since the informativeness of a dataset in ML classifiers is outside the scope of this paper, it is not reflected as a quantified metric in our paper. 

## Notes on Reusability
This artifact can be re-used to demonstrate the vulnerability of other ML models trained on other datasets. 
It can be further re-purposed to enhance the capabilities of other ML attacks like membership inference attacks and model inversion attacks.
With better tailored traversal techniques or use of auxiliary data, the query requirements can be decreased further.

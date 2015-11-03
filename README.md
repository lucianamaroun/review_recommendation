Review Recommendation
=====================
This project contains the implementation of different solutions of review recommendation task. The goal of a method of this kind is to obtain a ranking of reviews for a given user-item pair, regarding a reader user and an item of which reviews are about.

Filtering Step
--------------
The filtering step disregards several reviews considering the following criteria: empty fields, foreign text, and reviews in ranking groups (user-item pairs) with less than 10 elements. For filtering reviews, execute in the root folder of the project:

```
$ python -m script.filter_reviews
$ python -m script.filter_sparse
```

Where the first ignores invalid reviews and the second ignore reviews in too small rankings and reviews file is  data/rating.txt.

This step may be ignored if a filtered dataset is available at hand. The filtered dataset of a crawl from Ciao [1], whose format we use, is publicly available at http://homepages.dcc.ufmg.br/~lubm/review/reviews_filt.tar.gz. The original unfiltered dataset of Ciao platform is available at http://www.jiliang.xyz/Ciao.rar.

[1] Jiliang Tang, Huiji Gao, Xia Hu, and Huan Liu. 2013. Context-aware review helpfulness rating prediction. In Procs. of RecSys '13.

Modeling Step
-------------
The second step consists in modeling the collection of reviews, users and helpfulness votes using a set of features. To run the modeling step in the root folder of the project, 

```
$ python -m prep.modeling
```

After modeling all entitites, test and validation should be filtered to evaluate rankings properly. We considered a filter of at least 5 reviews per ranking, since we used nDCG@5 as metric.

```
$ python -m script.filter_val
$ python -m script.filter_test
```

Prediction Step
---------------
The prediction step uses a to predict reviews' rankings based on helpfulness scores, which is the same as recommending reviews in a top-N format. In this step, a method is fitted on training set and used for prediction of validation and test sets, whose predicted values are output to files. Several command line arguments configure this execution for each strategy; refer to source header of the respective technique. We explain here how to execute BETF and CAP methods.

BETF
----
The unBiased Extended Tensor Factorization (BETF) is a method for recommending reviews based only on latent variables of author, voter, review and product. It optimizes a least squares function using stochastic gradient descent. To run this algorithm, in root directory:

```
$ python -m algo.betf.main [-k <latent_dimensions>] [-l <learning_rate>] [-r <regularization_factor>] [-e <convergence_tolerance>] [-i <number_iterations>]
```

Where:
- \<latent_dimensions\> is an integer with the number of latent dimensions,      
- \<learning_rate\> is a float representing the update rate of gradient descent, 
- \<regularization\> is a float with the weight of regularization in objective function,
- \<tolerance\> is a float with the tolerance for convergence,
- \<iterations\> is an integer with the maximum number of iterations of gradient descent,
- \<bias_type\> is either 's' static or 'd' for dynamic, being updated in the optimization.


CAP
---
The Context-Aware review helpfulnes Prediction (CAP) is a method to recommend reviews based on latent variables. It uses a Monte Carlo Expectation Maximization (MCEM) algorithm to adjust latent variables and parameters in order to maximize the likelihood of the observed data (train set). To run this baseline,

```
$ python -m algo.cap.main [-k <latent_dimensions>] [-i <number_iterations>] [-g <gibbs_samples>] [-n <newton_iterations>] [-t <newton_tolerance>] [-l <newton_learning_rate>] [-a <eta>] [-s <scale>]
```

Where:
- \<latent_dimensions\> is an integer with the number of latent dimensions,      
- \<number_iterations\> is an integer with number of EM iterations,                     
- \<gibbs_samples\> is an integer with number of gibbs samples in each EM iteration,
- \<newton_iterations\> is an integer with number of newton-raphson iterations,      
- \<newton_tolerance\> is a float with newton-raphson convergence tolerance,          
- \<newton_learning_rate\> is a float with newton-raphson learning rate,             
- \<eta\> is a float constant used in OLS for easier computation of inverse,                  
- \<scale\> defines whether scale features, either 'y' for yes or 'n' for no. 

Evaluation Step
---------------
After fitting and applying a technique, we may evaluate considering RMSE and nDCG@p, p from 1 to 5, metrics. To evaluate an algorithm, in root directory:

```
$ python -m perf.main <set> <rep> <algorithm_configuration>
```
Where:
- \<set\> is either val, for validation, or test.
- <rep> is either 'y' or 'n' indicating the presence or absense, respectively, of repeated executions for each train-validation-test split (this should be true for non-deterministic methods, such as CAP, BETF and RLFM).
- <algorithm_configuration> is a string containing the name of the algorithm and a sequence of parameters values. For example, "cap-k:5,i:10,g:50,n:10,t:0.000100,l:0.100000,a:0.000100,s:y".

Dependencies
------------

<h4>External Methods</h4>
SVMRank (`algo/l2r/svmrank.py`), LambdaMART (`algo/l2r/lambdamart.py`) and RLFM (`algo/recsys/rlfm.py`) depend on third party modules to execute. SVMRank source code is available at https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html; LambdaMART is available inside RankLib at http://sourceforge.net/p/lemur/wiki/RankLib/; and RLFM at https://github.com/yahoo/Latent-Factor-Models. All of them shall be placed at lib folder under directories svm_rank, ranklib and rlfm, respectively.

<h4>Python Libraries</h4>
This project depends on the following libraries:
- NumPy: http://www.numpy.org/
- SciPy: http://www.scipy.org/
- Scikit-learn: http://scikit-learn.org/stable/
- NLTK: http://www.nltk.org/ and (the following corpus: maxent_treebank_pos_tagger, punkt, wordnet)
- TextBlob: https://textblob.readthedocs.org/en/dev/
- NetworkX: https://networkx.github.io/

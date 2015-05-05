Review Recommendation
=====================
This project solves the problem of review recommendation using different strategies.

Modeling Step
-------------
The first step consists in modeling the collection of reviews using a set of features. To run the modeling step in the root folder of the project, 

```
$ python -m src.modeling
```

The output for the train and test sets is configured in the header of the file src/modeling.py (variables _TRAIN_FILE  and _TEST_FILE).

Prediction Step
---------------
The prediction step uses a machine learning method to predict helpfulness ratings, i.e., recommend reviews. Several command line arguments configure this execution. To run the prediction step, 

```
$ python -m src.prediction -p <predictor> [-b <bias_code>] [-s <scaling_type>] [-i <iteration_count>]
```

In this command, <predictor> refers to the algorithm to be used (the algorithm codes are defined in the header of src/prediction.py under the variable _PREDICTORS). The <bias_code> refers to an optional bias adjustment using a subset of {'r', 'v', 'a'}, which corresponds to review, voter and author bias, respectively. The <scaling_type> configures feature scaling using either the scaling to the range to \[0,1\] ('minmax') or the mean normalization plus division by standard deviation ('standard'). The <iteration_count> is the number of times to reproduce the prediction, useful when the algorithm is not deterministic.

CAP Baseline
------------
The Context-Aware review helpfulnes Prediction (CAP) is a method to recommend reviews based on latent variables. It uses a Monte Carlo Expectation Maximization (MCEM) algorithm to adjust latent variables and parameters in order to maximize the likelihood of the observed data (train set). To run this baseline,

```
$ python -m cap.prediction
```

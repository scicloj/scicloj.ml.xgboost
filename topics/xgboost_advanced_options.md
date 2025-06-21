# Advanced Options in XGBoost

Where possible the options are frequently modeled off of the python library. For
references of using some of these functions, the tests of this library can be
helpful. Below is a list of some of the more advanced features of the library
and some usage examples.


## Custom Sample Weights

You may specify sample weights to have certain training data be considered more or less heavily.

To do this, pass in a `:sample-weights` option with a dataset that is `1 x N_TRAINING_DATA`

```clj
;; Example with random sample weights
(ml/train train-ds {:model-type :xgboost/classification
                    :sample-weights
                    (ds/->dataset
                      {:weight (repeatedly (ds/row-count train-ds) rand)})})
```

## Custom Objective Functions

You can specify a custom objective by passing a function to the `:objective`.

The function will be called with `float[][] predicts` predictions and the `DMatrix dtrain`.

It is expected to return a vector of two float arrays, corresponding to the first
and second order gradients.

Note that for multi-classification it is expected the number of floats correspond to
`N classes * N predictions`. You probably want to use a tensor and flatten it.

```clj
(ml/train
  train-ds
  {:model-type :xgboost/classification
   :objective
   (fn [predicts dtrain]
     (let [n (* n-classes (count predicts))]
       [(float-array n (repeatedly n rand))
        (float-array n (repeatedly n rand))]))})
```


## Custom Metric Functions

When using watches you can supply a custom evaluation metric for early stopping by providing
a vector of `[metric-name eval-fn]`.


The function will be called with `float[][] predicts` predictions and the `DMatrix dtrain` and is
expected to return a float


```clj
(def model
  (ml/train
    train-ds
    {:model-type :xgboost/regression
     :eval-metric
     ["CustomMetric"
      (fn [predicts dtrain]
        (rand))]}))

;; Accesss historical metric performance across rounds:

(-> model
   (get-in [:model-data :metrics]))
```

Also worth noting that you may want to set `:maximize-evaluation-metrics true`
if you want to maximize your metric instead of minimize it.

(ns scicloj.ml.xgboost-test
  (:require [clojure.test :refer [deftest is]]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.datatype :as dtype]
            [tech.v3.datatype.functional :as dfn]
            [scicloj.ml.smile.discrete-nb :as nb]
            [scicloj.ml.smile.nlp :as nlp]
            [scicloj.ml.xgboost]
            [scicloj.metamorph.ml :as ml]
            [scicloj.metamorph.ml.loss :as loss]
            [scicloj.metamorph.ml.verify :as verify]
            [tech.v3.dataset.categorical :as ds-cat]
            [scicloj.metamorph.ml.gridsearch :as ml-gs]))

            

(deftest basic
  (verify/basic-regression {:model-type :xgboost/regression} 0.22))


(deftest basic-early-stopping
  (verify/basic-regression {:model-type :xgboost/regression
                            :early-stopping-round 5
                            :round 50}
                           0.22))


(deftest watches
  (let [{train-dataset :train-ds
         test-dataset :test-ds} (ds-mod/train-test-split @verify/regression-iris*)
        options {:model-type :xgboost/regression
                 :watches {:test-ds test-dataset}
                 :round 25
                 :eval-metric "mae"}
        model (ml/train train-dataset options)
        watch-data (get-in model [:model-data :metrics])
        predictions (ml/predict test-dataset model)
        mse (loss/mse (predictions verify/target-colname)
                      (test-dataset verify/target-colname))]
    (is (= 25 (ds/row-count watch-data)))
    (is (not= 0 (dfn/reduce-+ (watch-data :test-ds))))
    (is (< mse (double 0.2)))))




(deftest classification
  (verify/basic-classification {:model-type :xgboost/classification} 0.25))

(deftest sparse-train-does-not-crash []
  (let [reviews
        (->
         (ds/->dataset "test/data/reviews.csv.gz" {:key-fn keyword})
         (ds/select-columns [:Text :Score])
         (nlp/count-vectorize :Text :bow nlp/default-text->bow)
         (nb/bow->SparseArray :bow :bow-sparse {:create-vocab-fn  #(nlp/->vocabulary-top-n % 100)})
         (ds/drop-columns [:Text :bow])
         (ds/update-column :Score
                           (fn [col]
                             (let [val-map {0 :c0
                                            1 :c1
                                            2 :c2
                                            3 :c3
                                            4 :c4
                                            5 :c5}]
                               (dtype/emap val-map :keyword col))))
         (ds/categorical->number cf/categorical)
         (ds-mod/set-inference-target :Score))
        model
        (ml/train reviews {:model-type :xgboost/classification
                           :sparse-column :bow-sparse
                           :n-sparse-columns 100})

        
        explanation (ml/explain model)
        test-ds (ds/head reviews 100)
        prediction (ml/predict test-ds model)
        train-acc
        (loss/classification-accuracy
         (->
          (ds-cat/reverse-map-categorical-xforms prediction)
          :Score)
         (-> test-ds
             (ds-cat/reverse-map-categorical-xforms)
             :Score))
        ]
       (is ( > train-acc 0.97))))


(deftest iris 
  (let [ src-ds (ds/->dataset "test/data/iris.csv")
        ds (->  src-ds
                (ds/categorical->number cf/categorical)
                (ds-mod/set-inference-target "species"))
        feature-ds (cf/feature ds)
        split-data (ds-mod/train-test-split ds {:seed 12345})
        train-ds (:train-ds split-data)
        test-ds (:test-ds split-data)
        model (ml/train train-ds {:validate-parameters "true"
                                  :seed 123
                                  :verbosity 1
                                  :model-type :xgboost/classification})
        predictions (ml/predict test-ds model)
        loss
        (loss/classification-accuracy
 (->
  predictions
  ds-cat/reverse-map-categorical-xforms
  (get "species"))
 (->
  test-ds
  ds-cat/reverse-map-categorical-xforms
  (get "species")))]

    (is (= 0.9555555555555556 loss))
    
    (is (= 
         [{:importance-type "gain", :colname "petal_width", :gain 3.0993214419727266}
          {:importance-type "gain", :colname "petal_length", :gain 2.8288314797695904}
          {:importance-type "gain", :colname "sepal_width", :gain 0.272344306208}
          {:importance-type "gain", :colname "sepal_length", :gain 0.12677490274290323}]         
         
         (ds/rows
          (ml/explain model))))))


(defn- test-options [train-ds test-ds options]
  (let [model (ml/train train-ds options)

        predictions (ml/predict test-ds model)
        accuracy
        (loss/classification-accuracy
         (->
          predictions
          ds-cat/reverse-map-categorical-xforms
          (get "Survived"))
         (->
          test-ds
          ds-cat/reverse-map-categorical-xforms
          (get "Survived")))]
    (assoc model :accuracy accuracy)))

(deftest titanic 
  (let [titanic (-> (ds/->dataset "test/data/titanic.csv")
                    (ds/drop-columns ["Name"])
                    (ds/update-column "Survived" (fn [col]
                                                   (dtype/emap #(if (== 1 (long %))
                                                                  "survived"
                                                                  "drowned")
                                                               :string col)))
                    (ds-mod/set-inference-target "Survived"))

        titanic-numbers (ds/categorical->number titanic cf/categorical)

        split-data (ds-mod/train-test-split titanic-numbers {:seed 1234})
        train-ds (:train-ds split-data)
        test-ds (:test-ds split-data)
        model (ml/train train-ds {:model-type :xgboost/classification})
        predictions (ml/predict test-ds model)

        accuracy
        (loss/classification-accuracy
         (->
          predictions
          ds-cat/reverse-map-categorical-xforms
          (get "Survived"))
         (->
          test-ds
          ds-cat/reverse-map-categorical-xforms
          (get "Survived")))

        
  

        opt-map (merge {:model-type :xgboost/classification}
                       (ml/hyperparameters :xgboost/classification))
        options-sequence (take 200  (ml-gs/sobol-gridsearch opt-map))


        models
        (->> (map #(test-options train-ds test-ds %) options-sequence)
             (sort-by :accuracy)
             reverse
             (take 10)
             (map #(select-keys % [:accuracy :options])))]
    (is (< 0.80 accuracy))
    (is (< 82
           (-> models first :accuracy (* 100) Math/round)))))


    


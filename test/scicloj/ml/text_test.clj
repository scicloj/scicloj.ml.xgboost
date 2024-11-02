(ns scicloj.ml.text-test
  (:require
   [clojure.data.csv :as csv]
   [clojure.java.io :as io]
   [clojure.set :as set]
   [clojure.string :as str]
   [clojure.test :refer [deftest is]]
   [scicloj.metamorph.ml.loss :as loss]
   [scicloj.metamorph.ml.text :as text]
   [scicloj.ml.xgboost :as xgboost]
   [scicloj.ml.xgboost.csr :as csr]
   [tablecloth.api :as tc]
   [tablecloth.column.api :as tcc]
   [tech.v3.dataset.column-filters :as cf])
  (:import
   [java.util.zip GZIPInputStream]
   [ml.dmlc.xgboost4j.java XGBoost]
   [ml.dmlc.xgboost4j.java DMatrix DMatrix$SparseType]))


(defn deterministic-shuffle
  [^java.util.Collection coll seed]
  (let [al (java.util.ArrayList. coll)
        rng (java.util.Random. seed)]
    (java.util.Collections/shuffle al rng)
    (clojure.lang.RT/vector (.toArray al))))

(deftest reviews-accuracy-sparse-matrix-classification
  (let [tidy 
        (text/->tidy-text  (io/reader (GZIPInputStream. (io/input-stream "test/data/reviews.csv.gz")))
                           line-seq
                           (fn [line]
                             (let [splitted (first
                                             (csv/read-csv line))]
                               [(first splitted)
                                (dec (Integer/parseInt (second splitted)))]))
                           #(str/split % #" ")
                           :max-lines 1000
                           :skip-lines 1)
        
        ds
        (-> tidy
         :datasets first          
         ;
         (tc/drop-rows #(= "" (:term %)))
         (tc/drop-missing))

        rnd-indexes (-> (range 1000) (deterministic-shuffle 123))
        rnd-indexes-train  (take 800 rnd-indexes)
        rnd-indexes-test (take-last 200 rnd-indexes)

        ds-train (tc/left-join (tc/dataset {:document rnd-indexes-train}) ds [:document])
        ds-test (tc/left-join (tc/dataset {:document rnd-indexes-test}) ds [:document])


        bow-train
        (-> ds-train
            
            text/->tfidf
            (tc/rename-columns {:meta :label})
            )

        bow-test
        (-> ds-test
            text/->tfidf
            (tc/rename-columns {:meta :label})
            )


        m-train (xgboost/tidy-text-bow-ds->dmatrix (cf/feature bow-train)
                                                   (tc/select-columns bow-train [:label])
                                                   :tfidf)
        m-test (xgboost/tidy-text-bow-ds->dmatrix (cf/feature bow-test)
                                                  (tc/select-columns bow-test [:label])
                                                  :tfidf)

        model
        (xgboost/train-from-dmatrix
         m-train
         ["word"]
         ["label"]
         {:num-class 5
          :validate-parameters "true"
          :seed 123
          :verbosity 0}
         {}
         "multi:softmax")

        booster
        (XGBoost/loadModel
         (java.io.ByteArrayInputStream. (:model-data model)))

        predition-train
        (->>
         (.predict booster m-train)
         (map #(int (first %))))

        predition-test
        (->>
         (.predict booster m-test)
         (map #(int (first %))))

        train-accuracy
        (loss/classification-accuracy
         (float-array predition-train)
         (.getLabel m-train))

        test-accuracy
        (loss/classification-accuracy
         (float-array predition-test)
         (.getLabel m-test))]

    (println :train-accuracy train-accuracy)
    (println :test-accuracy test-accuracy)

    (is (< 0.95 train-accuracy))
    (is (< 0.54 test-accuracy))))





(deftest small-text

  (let [tidy-result
        (text/->tidy-text (io/reader "test/data/small_text.csv")
                          line-seq
                          (fn [line]
                            (let [splitted (first
                                            (csv/read-csv line))]
                              (vector
                               (first splitted)
                               (dec (Integer/parseInt (second splitted))))))
                          #(str/split % #" ")
                          :max-lines 10000
                          :skip-lines 1)

        tidy-ds

        (->
         tidy-result
         :datasets
         first)
        
  
         id->token
        (-> tidy-result :token-lookup-table set/map-invert)
   
        bow
        (->
         tidy-ds
         text/->tfidf
         (tc/rename-columns {:meta :label}))

        sparse-features
        (-> bow
            (tc/select-columns [:document :token-idx :token-count])
            (tc/rows))

        n-rows (inc (apply tcc/max (bow :document)))
        n-col (inc (apply max  (bow :token-idx)))

        csr
        (csr/->csr sparse-features)

        dense
        (csr/->dense csr n-rows n-col)


        labels
        (->
         bow
         (tc/group-by :document)
         (tc/aggregate #(-> % :label first))
         (tc/column "summary"))

        m
        (DMatrix.
         (long-array (:row-pointers csr))
         (int-array (:column-indices csr))
         (float-array (:values csr))
         DMatrix$SparseType/CSR
         n-col)
        _ (.setLabel m (float-array labels))


        model
        (xgboost/train-from-dmatrix
         m
         ["word"]
         ["label"]
         {:num-class 2
          :verbosity 0}
         {}
         "multi:softprob")

        booster
        (XGBoost/loadModel
         (java.io.ByteArrayInputStream. (:model-data model)))

        predition
        (.predict booster m)]

    (is (= ["I", "like", "fish", "and", "you", "the", "fish", "Do", "you", "like", "me", "?"]
           (map id->token (:token-idx tidy-ds))))

    
    

    (is (= [1, 2, 3, 4, 5, 6, 3, 7, 5, 2, 8, 9] (tidy-ds :token-idx)))

    (is (= [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1] (tidy-ds :document)))

    (is (=  [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1] (tidy-ds :meta)))

    (is (= 
         [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]
         (:token-count bow)))


    (is (= 
         [[0 5 1] [0 6 1] [0 1 1] [0 3 2] [0 4 1] [0 2 1] [1 5 1] [1 9 1] [1 7 1] [1 8 1] [1 2 1]]
         sparse-features))

    (is (= 2 n-rows))
    (is (= 10 n-col))
    (is (= [[0 1.0 1.0 2.0 1.0 1.0 1.0 0 0 0]
            [0 0 1.0 0 0 1.0 0 1.0 1.0 1.0]]
           dense))
    (is (= [0 1] labels))

    (is (= [[0.5, 0.5], [0.5, 0.5]]
           (map seq predition)))))
  







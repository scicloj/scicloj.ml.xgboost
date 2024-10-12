(ns scicloj.ml.text-test
  (:require [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is]]
            [scicloj.metamorph.ml.loss :as loss]
            [scicloj.metamorph.ml.text :as text]
            [scicloj.ml.xgboost :as xgboost]
            [scicloj.ml.xgboost.csr :as csr]
            [tablecloth.api :as tc]
            [tablecloth.column.api :as tcc]
            [scicloj.metamorph.ml :as ml])
  (:import [java.util.zip GZIPInputStream]
           [ml.dmlc.xgboost4j.java XGBoost]
           [ml.dmlc.xgboost4j.java DMatrix DMatrix$SparseType]))



(deftest reviews-accuracy-sparse-matrix-classification
  (let [ds
        (->
         (text/->tidy-text  (io/reader (GZIPInputStream. (io/input-stream "test/data/reviews.csv.gz")))
                            (fn [line]
                              (let [splitted (first
                                              (csv/read-csv line))]
                                [(first splitted)
                                 (dec (Integer/parseInt (second splitted)))]))
                            #(str/split % #" ")
                            :max-lines 10000
                            :skip-lines 1)
         (tc/rename-columns {:meta :label})
         (tc/drop-rows #(= "" (:word %)))
         (tc/drop-missing))

        rnd-indexes (-> (range 1000) (shuffle))
        rnd-indexes-train  (take 800 rnd-indexes)
        rnd-indexes-test (take-last 200 rnd-indexes)

        ds-train (tc/left-join (tc/dataset {:document rnd-indexes-train}) ds [:document])
        ds-test (tc/left-join (tc/dataset {:document rnd-indexes-test}) ds [:document])

        bow-train
        (-> ds-train
            text/->term-frequency
            text/add-word-idx)
        bow-test
        (-> ds-test
            text/->term-frequency
            text/add-word-idx)

        m-train (xgboost/tidy-text-bow-ds->dmatrix bow-train)
        m-test (xgboost/tidy-text-bow-ds->dmatrix bow-test)

        model
        (xgboost/train-from-dmatrix
         m-train
         ["word"]
         ["label"]
         {:num-class 5}
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

    (is (< 0.95 train-accuracy))
    (is (< 0.54 test-accuracy))))





(deftest small-text

  (let [ds
        (->
         (text/->tidy-text (io/reader "test/data/small_text.csv")
                           (fn [line]
                             (let [splitted (first
                                             (csv/read-csv line))]
                               (vector
                                (first splitted)
                                (dec (Integer/parseInt (second splitted))))))
                           #(str/split % #" ")
                           :max-lines 10000
                           :skip-lines 1)
         (tc/rename-columns {:meta :label}))



        bow
        (-> ds
            text/->term-frequency
            text/add-word-idx)


        sparse-features
        (-> bow
            (tc/select-columns [:document :word-idx :tf])
            (tc/rows))


        n-rows (inc (apply tcc/max (bow :document)))


        n-col (inc (apply max  (bow :word-idx)))



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
         {:num-class 2}
         {}
         "multi:softprob")

        booster
        (XGBoost/loadModel
         (java.io.ByteArrayInputStream. (:model-data model)))

        predition
        (.predict booster m)]

    (is (= ["I", "like", "fish", "and", "you", "the", "fish", "Do", "you", "like", "me", "?"]
           (:word ds)))


    (is (= [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4] (ds :word-index)))

    (is (= [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1] (ds :document)))

    (is (=  [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1] (ds :label)))

    (is (= [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1]
           (:tf bow)))


    (is (= [[0 1 1] [0 2 1] [0 3 2] [0 4 1] [0 5 1] [0 6 1] [1 7 1] [1 5 1] [1 2 1] [1 8 1] [1 9 1]]
           sparse-features))

    (is (= 2 n-rows))
    (is (= 10 n-col))
    (is (= [[0 1.0 1.0 2.0 1.0 1.0 1.0 0 0 0]
            [0 0 1.0 0 0 1.0 0 1.0 1.0 1.0]]
           dense))
    (is (= [0 1] labels))

    (is (= [[0.5, 0.5], [0.5, 0.5]]
           (map seq predition)))))
  







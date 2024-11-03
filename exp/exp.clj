(ns exp
  (:require [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [scicloj.metamorph.ml.loss :as loss]
            [scicloj.metamorph.ml.text :as text]
            [scicloj.ml.xgboost :as xgboost]
            [tablecloth.api :as tc]
            [tech.v3.dataset.column-filters :as cf])
  (:import [java.util.zip GZIPInputStream]
           [ml.dmlc.xgboost4j.java XGBoost]))
(def max-lines 1000) ;  fails with 10000

(defn deterministic-shuffle
  [^java.util.Collection coll seed]
  (let [al (java.util.ArrayList. coll)
        rng (java.util.Random. seed)]
    (java.util.Collections/shuffle al rng)
    (clojure.lang.RT/vector (.toArray al))))

  (let [
        
        _ (println :slurp)
        ds
        (->
         (text/->tidy-text  (io/reader  (io/input-stream "repeatedAbstrcats.txt"))
                            line-seq
                            (fn [line]
                              [line
                              (rand-int 5)])
                            #(str/split % #" ")
                            :max-lines max-lines
                            :skip-lines 1
                            :datatype-document :int32
                            :datatype-token-idx :int32)
         :datasets
         first
         
         (tc/drop-rows #(= "" (:term %)))
         (tc/drop-missing))

        _ (def ds ds)

        ;(tc/select-rows ds #(= 603 (:document %)))
        
        rnd-indexes (-> (range max-lines) (deterministic-shuffle 123))
        rnd-indexes-train  (take (* max-lines 0.8) rnd-indexes)
        rnd-indexes-test (take-last (* max-lines 0.2) rnd-indexes)

        ds-train (tc/inner-join (tc/dataset {:document rnd-indexes-train}) ds [:document])
        ds-test (tc/inner-join (tc/dataset {:document rnd-indexes-test}) ds [:document])

        _ (def ds-train ds-train)
        _ (def ds-test ds-test)
        _ (tc/select-missing ds-train)
        _ (println :->term-frequency)

        n-sparse-columns (inc (apply max  (ds :token-idx)))
        bow-train
        (-> ds-train
            text/->tfidf
            (tc/rename-columns {:meta :label}))

        _ (tc/select-missing bow-train)
        bow-test
        (-> ds-test
            text/->tfidf
            (tc/rename-columns {:meta :label}))


        _ (println :to-matrix)

        _ (def bow-train bow-train)
        m-train (xgboost/tidy-text-bow-ds->dmatrix (cf/feature bow-train)
                                                   (tc/select-columns bow-train [:label])
                                                   :tfidf
                                                   n-sparse-columns)
        m-test (xgboost/tidy-text-bow-ds->dmatrix (cf/feature bow-test)
                                                  (tc/select-columns bow-test [:label])
                                                  :tfidf
                                                  n-sparse-columns)

        _ (println :train)
        model
        (xgboost/train-from-dmatrix
         m-train
         ["term"]
         ["label"]
         {:num-class 5
          :validate-parameters "true"
          :seed 123
          :verbosity 0
          ;:n-sparse-columns n-sparse-columns
          }
         {}
         "multi:softmax")

        booster
        (XGBoost/loadModel
         (java.io.ByteArrayInputStream. (:model-data model)))

        _ (println :predict)
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
    (println :test-accuracy test-accuracy))

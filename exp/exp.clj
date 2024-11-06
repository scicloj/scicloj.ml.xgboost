(ns exp
  (:require
   [clojure.data.csv :as csv]
   [clojure.java.io :as io]
   [clojure.string :as str]
   [scicloj.metamorph.ml.text :as text]
   [scicloj.ml.xgboost :as xgboost]
   [tablecloth.api :as tc]
   [tablecloth.column.api :as tcc]
   [scicloj.ml.xgboost.csr :as csr]
   [tech.v3.datatype :as dt]
   [tech.v3.datatype.list :as dt-list]))
(def max-lines 10000) ;  fails with 10000

(def reviews
  (text/->tidy-text  (io/reader  (io/input-stream "test/data/amazon_reviews.csv"))
                     line-seq
                     (fn [line]
                       (let [[[rating title text]]
                             (csv/read-csv line)]
                         [(str title " " text)
                          (Integer/parseInt rating)]))
                     #(str/split % #" ")
                     :max-lines max-lines
                     :skip-lines 1
                     :datatype-token-idx :int32
                     :datatype-document :int32
                     :container-type :native-heap))


(def reday-for-softmax
  (-> reviews
      :datasets
      first
      (tc/update-columns {:meta #(tcc/- % 1)})))



(comment

  (def tidy
    (text/libsvm->tidy (io/reader "test/data/amazon_reviews.libsvm")))




  (def r-c-vs
    (-> tidy
        (tc/select-columns [:instance :index :value])
        (tc/head 1000000)
        (tc/rows)))





  ;; "Elapsed time: 22430.996899 msecs"
  ;; "Elapsed time: 901 msecs"
  (time
   (def csrs
     (csr/->csr r-c-vs))))



(comment)



(comment
  (def tfidf
    (-> reday-for-softmax
        text/->tfidf))




  (text/tfidf->svmlib! tfidf (io/writer "test/data/amazon_reviews.libsvm") :tfidf)



;; https://www.kaggle.com/code/shivangamsoni/sentiment-analysis-tf-idf

  (import '[ml.dmlc.xgboost4j.java DMatrix])


  (def dmatrix (DMatrix. "test/data/amazon_reviews.libsvm?format=libsvm"))


  (def model
    (xgboost/train-from-dmatrix
     {:dmatrix dmatrix}
     ["word"]
     ["label"]
     {:num-class 2
      :validate-parameters "true"
      :seed 123
      :verbosity 0}
     {}
     "multi:softmax")))


(require '[tech.v3.datatype :as dt]
         '[tech.v3.datatype.list :as dt-list])


(def l (dt/make-list :float32))
(.add l 1.0)
;;=> true
l
;;=> [1.0]


(def l-1 (dt-list/wrap-container (dt/make-container :jvm-heap :float32 0)))
(.add l-1 1.0)
;;=> true
l-1
;;=> [1.0]


(def l-2 (dt-list/wrap-container (dt/make-container :native-heap :float32 0)))
(.ensureCapacity l-2 1)
;;=> Execution error (IndexOutOfBoundsException) at tech.v3.datatype.native_buffer.DoubleNativeBuf/writeDouble (native_buffer.clj:424).
;;   idx (0) >= n-elems (0)
;;   
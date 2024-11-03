;; re-implements https://blog.newtum.com/sparse-matrix-in-java/
;; maybe se here, nmot sure teh same: https://github.com/scipy/scipy/blob/v1.14.1/scipy/sparse/_csr.py
(ns scicloj.ml.xgboost.csr
  (:require
   [tech.v3.datatype :as dt]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defn- add-to-csr [csr ^long row ^long col ^double value]
  (if (zero? value)
    csr
    (let [new-values (conj (:values csr) (float value))
          new-column-indices (conj (:column-indices csr) col)
          new-row-pointers (if (<= (count (:row-pointers csr)) row)

                             (conj (:row-pointers csr) (dec (count new-values)))
                             (:row-pointers csr))]
      {:values new-values
       :column-indices new-column-indices
       :row-pointers new-row-pointers})))

(defn ->csr [r-c-vs]
  ;; data gets sorted by r and c
  ;; not sure, if good idea for performace ?
  
  (let [ r-c-v-maps
        (->> r-c-vs
             ( (fn [it] (println :sort) it))
             (sort-by (juxt first second))
             ( (fn [it] (println :reduce) it))
             (reduce
              (fn [csr [row col value]]
                (add-to-csr csr row col value))
              {:values (dt/make-list :float)
               :column-indices (dt/make-list :int)
               :row-pointers (dt/make-list :long [0])}))]  
    
    (assoc r-c-v-maps :row-pointers 
           (conj (:row-pointers r-c-v-maps)
                 (count (:values r-c-v-maps))))))


(defn- first-non-nil-or-0 [s]
  (or
   (first (filter some? s))
   0))


(defn ->dense [csr rows cols]
  (for [^long i (range rows)]
    (let [row-start (nth (:row-pointers csr) i 0)
          row-end   (nth (:row-pointers csr) (inc i) 0)]
      (for [j (range cols)]
        (first-non-nil-or-0
         (for [k (range row-start row-end)]
           (if (= (nth (:column-indices csr) k) j)
             (nth (:values csr) k)
             nil)))))))



(comment
  (import '[ml.dmlc.xgboost4j.java DMatrix DMatrix$SparseType])
  (DMatrix.
   (long-array (:row-pointers csr))
   (int-array (:column-indices csr))
   (float-array (:values csr))
   DMatrix$SparseType/CSR)
  )




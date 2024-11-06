; re-implements https://blog.newtum.com/sparse-matrix-in-java/
;; maybe se here, nmot sure teh same: https://github.com/scipy/scipy/blob/v1.14.1/scipy/sparse/_csr.py
(ns scicloj.ml.xgboost.csr
  (:require
   [tech.v3.datatype :as dt]
   [ham-fisted.api :as hf]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defn- add-to-csr [csr ^long row ^long col value]
  (if (zero? ^float value)
    csr
    (let [new-values (hf/conj! (:values csr) value)
          new-column-indices (hf/conj! (:column-indices csr) col)
          new-row-pointers (if (<=  (hf/constant-count (:row-pointers csr)) row)
                             (hf/conj! (:row-pointers csr) (dec (hf/constant-count new-values)))
                             (:row-pointers csr))]
      {:values new-values
       :column-indices new-column-indices
       :row-pointers new-row-pointers})))

(defn ->csr [r-c-vs]
  ;; TODO:: faster impl based on this:
  ;;https://stackoverflow.com/questions/23583975/convert-coo-to-csr-format-in-c
  
  (let [ r-c-v-maps
        (->> r-c-vs
             (sort-by (juxt first second))
             (reduce
              (fn [csr [row col value]]
                (add-to-csr csr row col value))
              {:values (hf/double-array-list)
               :column-indices (hf/int-array-list)
               :row-pointers (hf/long-array-list [0])}))]  
    
    (assoc r-c-v-maps :row-pointers 
           (hf/conj! (:row-pointers r-c-v-maps)
                 (hf/constant-count (:values r-c-v-maps))))))


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




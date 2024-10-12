(ns scicloj.ml.xgboost.csr
  (:require
            [ tech.v3.datatype :as dt]
            ))

(defn- add-to-csr [csr row col value]
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
  (->
   (reduce

    (fn [csr [row col value]]
      (add-to-csr csr row col value))
    {:values (dt/make-list :float)
     :column-indices (dt/make-list :int)
     :row-pointers (dt/make-list :long [0])}
    r-c-vs)

   (#(assoc % :row-pointers (conj (:row-pointers %)
                                  (count (:values %)))))))


(defn- first-non-nil-or-0 [s]
  (or
   (first (filter some? s))
   0))


(defn ->dense [csr rows cols]
  (for [i (range rows)]
    (let [row-start (nth (:row-pointers csr) i)
          row-end   (nth (:row-pointers csr) (inc i))]
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




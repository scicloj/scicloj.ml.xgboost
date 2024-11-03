(ns scicloj.ml.xgboost.model
  "Internal namespace of helper functions used to implement models."
  (:require [tech.v3.datatype :as dtype]
            [tech.v3.tensor :as dtt]
            [tech.v3.dataset.tensor :as ds-tens]
            [tech.v3.dataset :as ds]
            [clojure.set :as set]))



(defn finalize-regression
  ;; attention: this function might be smile specific
  ;; it assumes a certain order of prediction probbalilities in `reg-tens`
  ;;

  [reg-tens target-cname]
  (let [n-rows (dtype/ecount reg-tens)]
    (-> (dtt/reshape reg-tens [n-rows 1])
        (ds-tens/tensor->dataset)
        (ds/rename-columns {0 target-cname})
        (ds/update-columnwise :all vary-meta assoc :column-type :prediction)
        (vary-meta assoc :model-type :regression))))


(defn finalize-classification
  ;; attention: this function might be smile specific
  ;; it assumes a certain relation in the order of prediction probbalilities in `cls-tens`
  ;; and teh categoricla map

  [cls-tens target-cname target-categorical-maps]

  (let [rename-map (-> (get-in target-categorical-maps
                               [target-cname :lookup-table])
                       (set/map-invert))]
    (-> (dtt/reshape cls-tens (dtype/shape cls-tens))
        (ds-tens/tensor->dataset)
        (ds/rename-columns rename-map)
        (ds/update-columnwise :all vary-meta assoc
                              :column-type :probability-distribution)
        (vary-meta assoc :model-type :classification))))

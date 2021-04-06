(defproject scicloj/scicloj.ml.xgboost "5.03"
  :description "xgboost models for scicloj.ml"
  :url "https://github.com/scicloj/scicloj.ml.xgboost"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.10.2"]
                 [techascent/tech.ml.dataset "5.04"]
                 [ml.dmlc/xgboost4j_2.12 "1.3.1"]
                 [ml.dmlc/xgboost4j-spark_2.12 "1.3.1"]
                 [scicloj/metamorph.ml "0.3.0-beta1"]
                 [org.apache.hadoop/hadoop-common "3.3.0"
                  :exclusions [org.apache.commons/commons-math3
                               org.slf4j/slf4j-log4j12]]
                 [pppmap/pppmap "0.2.1"]]

  )

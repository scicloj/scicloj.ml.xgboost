(defproject scicloj/scicloj.ml.xgboost "5.1.1"
  :description "xgboost models for scicloj.ml"
  :url "https://github.com/scicloj/scicloj.ml.xgboost"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.10.2"]
                 ;; [techascent/tech.ml.dataset "6.025"]
                 [ml.dmlc/xgboost4j_2.12 "1.3.1"]
                 [ml.dmlc/xgboost4j-spark_2.12 "1.3.1"]
                 [scicloj/metamorph.ml "0.6.0"]
                 [org.apache.hadoop/hadoop-common "3.3.0"
                  :exclusions [org.apache.commons/commons-math3
                               org.slf4j/slf4j-log4j12]]
                 [pppmap/pppmap "0.2.1"]])

  

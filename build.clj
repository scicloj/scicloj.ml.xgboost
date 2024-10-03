(ns build
  (:refer-clojure :exclude [test])
  (:require [clojure.tools.build.api :as b] ; for b/git-count-revs
            [org.corfield.build :as bb]))

(def lib 'org.scicloj/scicloj.ml.xgboost)
; alternatively, use MAJOR.MINOR.COMMITS:
;; (def version (format "7.0.%s" (b/git-count-revs nil)))
(def version "6.1.0")
(def class-dir "target/classes")
(def basis (b/create-basis {:project "deps.edn"}))
(def jar-file (format "target/%s-%s.jar" (name lib) version))



(defn compile [_]
  (b/javac {:src-dirs ["java"]
            :class-dir class-dir
            :basis basis
            ;;:javac-opts ["-source" "8" "-target" "8"]
            }))



(defn test "Run the tests." [opts]
  (bb/run-tests opts))

(defn- pom-template []
  [[:description "xgboost models for metamorph.ml and scicloj.ml"]
   [:url "https://github.com/scicloj/scicloj.ml.xgboost"]
   [:licenses
    [:license
     [:name "Eclipse Public License - v 1.0"]
     [:url "https://www.eclipse.org/legal/epl-1.0/"]]]])


(defn jar [_]
  (compile nil)
  (b/write-pom {:class-dir class-dir
                :lib lib
                :version version
                :basis basis
                :src-dirs ["src"]
                :pom-data  (pom-template)})
  (b/copy-dir {:src-dirs ["src" "resources"]
               :target-dir class-dir})
  (b/jar {:class-dir class-dir
          :jar-file jar-file}))

(defn ci "Run the CI pipeline of tests (and build the JAR)." [opts]
  (-> opts
      (assoc :lib lib :version version
             :aliases [:run-tests])
      (test)
      (bb/clean)
      (jar)))


(defn install "Install the JAR locally." [opts]
  (-> opts
      (assoc :lib lib :version version)
      (bb/install)))

(defn deploy "Deploy the JAR to Clojars." [opts]
  (-> opts
      (assoc :lib lib :version version)
      (bb/deploy)))

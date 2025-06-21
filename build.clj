(ns build
  (:refer-clojure :exclude [test])
  (:require [clojure.tools.build.api :as b] ; for b/git-count-revs
            [org.corfield.build :as bb]
            [deps-deploy.deps-deploy :as dd]))

(def lib 'org.scicloj/scicloj.ml.xgboost)
; alternatively, use MAJOR.MINOR.COMMITS:
;; (def version (format "7.0.%s" (b/git-count-revs nil)))
(def version "6.3.0")
(def class-dir "target/classes")
(def basis (b/create-basis {:project "deps.edn"}))
;;(def jar-file (format "target/%s-%s.jar" (name lib) version))




(defn test "Run the tests." [opts]
  (-> opts
      (assoc :aliases [:test :run-tests])
      (bb/run-tests)))

(defn- pom-template []
  [[:description "xgboost models for metamorph.ml and scicloj.ml"]
   [:url "https://github.com/scicloj/scicloj.ml.xgboost"]
   [:licenses
    [:license
     [:name "Eclipse Public License - v 1.0"]
     [:url "https://www.eclipse.org/legal/epl-1.0/"]]]
   [:scm
    [:url "https://github.com/scicloj/scicloj.ml.xgboost"]
    [:connection "scm:git:https://github.com/scicloj/scicloj.ml.xgboost.git"]
    [:developerConnection "scm:git:https://github.com/scicloj/scicloj.ml.xgboost.git"]
    [:tag (str "v" version)]]])


(defn- jar-opts [opts]
  (assoc opts
         :lib lib   :version version
         :jar-file  (format "target/%s-%s.jar" lib version)
         :basis     (b/create-basis {})
         :class-dir class-dir
         :target    "target"
         :src-dirs  ["src"]
         :pom-data  (pom-template)))


(defn jar [_]
  (b/write-pom {:class-dir class-dir
                :lib lib
                :version version
                :basis basis
                :src-dirs ["src"]
                :pom-data  (pom-template)})
  (b/copy-dir {:src-dirs ["src" "resources"]
               :target-dir class-dir})
  (b/jar {:class-dir class-dir
          :jar-file (:jar-file (jar-opts {}))}))

(defn generate-pom [_]
  (b/write-pom {:target "."
                :lib lib
                :version version
                :basis basis
                :pom-data (pom-template)
                :src-dirs ["src"]}))

(defn ci "Run the CI pipeline of tests (and build the JAR)." [opts]
  (-> opts
      (assoc :lib lib :version version
             :aliases [:test :run-tests])
      (test)
      (bb/clean)
      (jar)))



(defn install "Install the JAR locally." [opts]
  (-> opts
      (assoc :lib lib :version version)
      (bb/install)))

;; (defn deploy "Deploy the JAR to Clojars." [opts]
;;   (-> opts
;;       (assoc :lib lib :version version)
;;       (bb/deploy)))


(defn deploy "Deploy the JAR to Clojars." [opts]
  (let [{:keys [jar-file] :as opts} (jar-opts opts)]
    (dd/deploy {:installer :remote :artifact (b/resolve-path jar-file)
                :pom-file (b/pom-path (select-keys opts [:lib :class-dir]))}))
  opts)
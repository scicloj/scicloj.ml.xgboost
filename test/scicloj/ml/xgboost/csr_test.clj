(ns scicloj.ml.xgboost.csr-test
  (:require  [clojure.test :refer [deftest is]] 
             [scicloj.ml.xgboost.csr :as csr]
             [tech.v3.tensor :as t]))


;; scipy
;;1 0 2 0
;;4 0 0 3
;;3 1 2 0
;;csr=csr_matrix (np.array ([[1,0,2,0],[4,0,0,3],[3,1,2,0]]))

;;>>> csr.data    , same as dmatrix.data
;;array ([1, 2, 4, 3, 3, 1, 2])

;; >>> csr.indices, same as dmatrix.colIndex
;;array ([0, 2, 0, 3, 0, 1, 2], dtype=int32)

;;>>> csr.indptr, same as dmatrix.rowHeaders
;;array ([0, 2, 4, 7], dtype=int32)


(deftest ->csr
;;                       3.        1.       2.  
;;>>> csr=coo_array (([5,8,3,6],([0,1,2,3],[0,1,2,1])),shape= (4,4)) .tocsr ()
;;>>> csr.data
;;array ([5, 8, 3, 6])
;;>>> csr.indices
;;array ([0, 1, 2, 1])
;;>>> csr.indptr
;;array ([0, 1, 2, 3, 4])  

  (is (=
       {:values [5.0 8.0 3.0 6.0], :column-indices [0 1 2 1], :row-pointers [0 1 2 3 4]}
       (csr/->csr
        ;; row,col,value
        [[0 0 5]
         [1 1 8]
         [2 2 3]
         [3 1 6]]))))


(deftest ->csr-2
;; matches wikipedia https://en.wikipedia.org/wiki/Sparse_matrix
;;in python
;; coo_array (([10,20,30,40,50,60,70,80],([0,0,1,1,2,2,2,3],[0,1,1,3,2,3,4,5])),shape=(4,6)).todense()
;; array ([[10, 20,  0,  0,  0,  0],
;;         [0, 30,  0, 40,  0,  0],
;;         [0,  0, 50, 60, 70,  0],
;;         [0,  0,  0,  0,  0, 80]])
;;
;;                             3.                       1.                2.
;; >>> csr=coo_array (([10,20,30,40,50,60,70,80],([0,0,1,1,2,2,2,3],[0,1,1,3,2,3,4,5])),shape= (4,6)) .tocsr ()
;; >>> csr.data
;; array ([10, 20, 30, 40, 50, 60, 70, 80])
;; >>> csr.indices
;; array ([0, 1, 1, 3, 2, 3, 4, 5])
;; >>> csr.indptr
;; array ([0, 2, 4, 7, 8])
  (is (= {:values [10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0] 
          :column-indices [0 1 1 3 2 3 4 5] 
          :row-pointers [0 2 4 7 8]}
         (csr/->csr
          [
           [0 0 10.0]
           [0 1 20.0]
           [1 1 30.0]
           [1 3 40.0]
           [2 2 50.0]
           [2 3 60.0]
           [2 4 70.]
           [3 5 80.0]]))))
;;=> 


(deftest unsorted []
  (is (= {:values [10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0]
          :column-indices [0 1 1 3 2 3 4 5]
          :row-pointers [0 2 4 7 8]}
         (csr/->csr
          (shuffle
           [[0 0 10.0]
            [0 1 20.0]
            [1 1 30.0]
            [1 3 40.0]
            [2 2 50.0]
            [2 3 60.0]
            [2 4 70.0]
            [3 5 80.0]]))
         )))


(deftest ->dense
  (is (=
       [[10.0 20.0 0 0 0 0]
        [0 30.0 0 40.0 0 0]
        [0 0 50.0 60.0 70.0 0]
        [0 0 0 0 0 80.0]]

       (csr/->dense {:values [10.0 20.0 30.0 40.0 50.0 60.0 70. 80.0]
                     :column-indices [0  1  1  3  2  3  4  5]
                     :row-pointers [0  2  4  7  8]}
                    4 6)))

  (is (=
       '((5.0 0 0 0) (0 8.0 0 0) (0 0 3.0 0) (0 6.0 0 0))
       (->
        [[0 0 5]
         [1 1 8]
         [2 2 3]
         [4 1 6]]
        (csr/->csr)
        (csr/->dense 4 4)))))

(deftest >tensor
  (is (= 
       [[5.000     0     0 0]
        [0 8.000     0 0]
        [0     0 3.000 0]
        [0 6.000     0 0]]
       (t/->tensor
        (csr/->dense
         (csr/->csr
          [[0 0 5]
           [1 1 8]
           [2 2 3]
           [4 1 6]])
         4 4)))))
  
     
(t/->tensor
 (csr/->dense
  (csr/->csr
   [[0 0 5]
    [1 1 8]
    [2 2 3]
    [4 1 6]])
  4 4))
;;=> #tech.v3.tensor<object>[4 4]
;;   [[5.000     0     0 0]
;;    [    0 8.000     0 0]
;;    [    0     0 3.000 0]
;;    [    0 6.000     0 0]]


(t/->tensor
 (csr/->dense
  (csr/->csr
   [[1 1 8]
    [2 2 3]
    [4 1 6]
    [0 0 5]]
   )
  4 4))
;;=> #tech.v3.tensor<object>[4 4]
;;   [[    0     0     0 0]
;;    [    0 8.000     0 0]
;;    [    0     0 3.000 0]
;;    [5.000 6.000     0 0]]

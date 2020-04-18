####### Installing tensorflow for R #########

install.packages("tensorflow")

library(tensorflow)
install_tensorflow()

## confirm installatioin 
tf$constant("Hello Tensorflow")

## result: tf.Tensor(b'Hellow Tensorflow', shape=(), dtype=string)


###### Installing Keras ##########

install.packages("keras")
library(keras)
install_keras()

######## Converting Spam data set for Learning via Keras in R ###########

if(!file.exists("spam.data")){
  download.file(
    "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data",
    "spam.data")
}

spam.dt <- data.table::fread("spam.data")

label.col <- ncol(spam.dt)

set.seed(1)
fold.vec <- sample(rep(1:5, l=nrow(spam.dt)))
test.fold <- 1
is.test <- fold.vec == test.fold
is.train <- !is.test

X.sc <- scale(spam.dt[, -label.col, with=FALSE])
## -label.col : all the other cols except from label call
## with=FALSE : dont want col name "-label.col" , look for var name "label.col"
## X.sc is numberic matrix

X.train.mat <- X.sc[is.train,]
X.test.mat <- X.sc[is.test,]
X.train.arr <- array(X.train.mat , dim(X.train.mat))
X.test.arr <- array(X.test.mat , dim(X.test.mat))

y.train <- y[is.train]
y.test <- y[is.test]

## setting aside our own validation split insted of randomizing with
## "validation_split= 0.3" function in model

set.seed(1)
is.subtrain <- sample(rep(c(TRUE, FALSE), l=length(y.train)))
table(is.subtrain)
is.validation <- !is.subtrain
X.subtrain.mat <- X.train.mat[is.subtrain,]
X.validation.mat <- X.train.mat[!is.subtrain,]
y.subtrain <- y.train[is.subtrain]
y.validation <- y.train[!is.subtrain]


###### Variable number of hidden layers #######

layers.metrics.list <- list()

(hidden.layers.vec <-  0:5)
hidden.layers.new <- hidden.layers.vec[
  !hidden.layers.vec %in% names(layers.metrics.list)]

for(hidden.layers.i in seq_along(hidden.layers.new)){
  layers.metrics.list.new <- future.apply::future_lapply(
    hidden.layers.new, function(hidden.layers){
      print(hidden.layers)
      model <- keras_model_sequential()
      
      ## loop of how many layers to be added to model for specific iteration
      for( layer.i in (0:hidden.layers)[-1]){
        layer_dense(                                                      # hidden layer
          model, 
          input_shape = ncol(X.train.mat),
          units = 10,
          activation = "relu",
          use_bias = FALSE) 
      }
      
      layer_dense(model, 1 , activation = "sigmoid", use_bias = FALSE)  # output layer
      model %>%
        compile(
          loss= "binary_crossentropy",
          optimizer = optimizer_adam(lr=0.01),
          metrics = "accuracy"
        )
      results <- model %>%
        fit(
          x = X.subtrain.mat, y = y.subtrain,
          epoch = 100,
          validation_data= list(X.validation.mat, y.validation),
          verbose = 2
        )
      
      metrics.wide <- do.call(data.table::data.table, results$metrics)
      metrics.wide[, epoch := 1:.N]
      data.table::data.table(hidden.layers, metrics.wide)
    }
  )
}








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

###### Variable number of hidden units #######
hidden.unit.metrics.list <- list()

for(n.hidden.units in c(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)){
  model <- keras_model_sequential() %>%
    layer_dense(
      input_shape=ncol(X.train.mat),
      units = n.hidden.units,
      activation = "sigmoid",
      use_bias=FALSE) %>% 
    layer_dense(1, activation = "sigmoid", use_bias=FALSE)
  model %>%
    compile(
      loss = "binary_crossentropy",
      optimizer = optimizer_adam(lr=0.01),
      metrics = "accuracy"
    )
  result <- model %>%
    fit(
      x = X.subtrain.mat, y = y.subtrain,
      epochs = 100,
      validation_data=list(X.validation.mat, y.validation),
      verbose = 2
    )
  unit.metrics <- do.call(data.table::data.table, result$metrics)
  unit.metrics[, epoch := 1:.N]
  hidden.unit.metrics.list[[length(hidden.unit.metrics.list)+1]] <- data.table::data.table(
    n.hidden.units, unit.metrics)
  
}

hidden.unit.metrics <- do.call(rbind, hidden.unit.metrics.list)

library(ggplot2)
(units.metrics.tall <- nc::capture_melt_single(
  hidden.unit.metrics,
  set="val_|",
  metric="loss|acc"))
units.metrics.tall[, Set := ifelse(set=="val_", "validation", "subtrain")]

units.loss.tall <- units.metrics.tall[metric=="loss"]
units.loss.min <- units.loss.tall[, .SD[
  which.min(value)], by=.(Set, n.hidden.units)]

ggplot()+
  geom_line(aes(
    x=epoch, y=value, color=Set),
    data=units.loss.tall)+
  geom_point(aes(
    x=epoch, y=value, color=Set),
    data=units.loss.min)+
  theme_bw()+
  facet_wrap("n.hidden.units")

min.loss.dt.list <-list()
uni.unit.vec <- unique(hidden.unit.metrics[, n.hidden.units])
for( units in uni.unit.vec){
  min.loss.dt.list[[length(min.loss.dt.list)+1]] <- hidden.unit.metrics[ n.hidden.units == units][which.min(loss)]
}

min.loss.dt <- do.call(rbind, min.loss.dt.list)

min.val.loss.dt.list <-list()
for( units in uni.unit.vec){
  min.val.loss.dt.list[[length(min.val.loss.dt.list)+1]] <- hidden.unit.metrics[ n.hidden.units == units][which.min(val_loss)]
}

min.val.loss.dt <- do.call(rbind, min.val.loss.dt.list)

min.loss.dt.list <-list()
uni.unit.vec <- unique(hidden.unit.metrics[, n.hidden.units])
for( units in uni.unit.vec){
  min.loss.dt.list[[length(min.loss.dt.list)+1]] <- hidden.unit.metrics[ n.hidden.units == units][which.min(loss)]
}

min.loss.dt <- do.call(rbind, min.loss.dt.list)


min.val.loss.dt.list <-list()
for( units in uni.unit.vec){
  min.val.loss.dt.list[[length(min.val.loss.dt.list)+1]] <- hidden.unit.metrics[ n.hidden.units == units][which.min(val_loss)]
}

min.val.loss.dt <- do.call(rbind, min.val.loss.dt.list)

train.min.dt.list <- list()

#### 2 hidden units with best epoch : 99#####
model <- keras_model_sequential() %>%
  layer_dense(
    input_shape=ncol(X.train.mat),
    units = 2,
    activation = "sigmoid",
    use_bias=FALSE) %>% 
  layer_dense(1, activation = "sigmoid", use_bias=FALSE) 
model %>%
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_adam(lr=0.01),
    metrics = "accuracy" 
  )
result1 <- model %>%
  fit(
    x = X.subtrain.mat, y = y.subtrain,
    epochs = 99,
    validation_data=list(X.validation.mat, y.validation),
    verbose = 2
  )

unit.metrics <- do.call(data.table::data.table , result1$metrics)
unit.metrics[, epoch := 1:.N]
train.min.dt.list[[length(train.min.dt.list)+1]] <- unit.metrics[which.min(loss)]

#### 128 hidden units with best epoch : 98 #####
model <- keras_model_sequential() %>%
  layer_dense(
    input_shape=ncol(X.train.mat),
    units = 128,
    activation = "sigmoid",
    use_bias=FALSE) %>% 
  layer_dense(1, activation = "sigmoid", use_bias=FALSE) 
model %>%
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_adam(lr=0.01),
    metrics = "accuracy" 
  )
result2 <- model %>%
  fit(
    x = X.subtrain.mat, y = y.subtrain,
    epochs = 98,
    validation_data=list(X.validation.mat, y.validation),
    verbose = 2
  )

unit.metrics <- do.call(data.table::data.table , result2$metrics)
unit.metrics[, epoch := 1:.N]
train.min.dt.list[[length(train.min.dt.list)+1]] <- unit.metrics[which.min(loss)]

#### 1024 hidden units with best epoch : 93 #####
model <- keras_model_sequential() %>%
  layer_dense(
    input_shape=ncol(X.train.mat),
    units = 1024,
    activation = "sigmoid",
    use_bias=FALSE) %>% 
  layer_dense(1, activation = "sigmoid", use_bias=FALSE) #
model %>%
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_adam(lr=0.01),
    metrics = "accuracy" 
  )
result3 <- model %>%
  fit(
    x = X.subtrain.mat, y = y.subtrain,
    epochs = 93,
    validation_data=list(X.validation.mat, y.validation),
    verbose = 2
  )

unit.metrics <- do.call(data.table::data.table , result3$metrics)
unit.metrics[, epoch := 1:.N]
train.min.dt.list[[length(train.min.dt.list)+1]] <- unit.metrics[which.min(loss)]
  

###### Baseline prediction ######
y.tab <- table(y.train)
y.baseline <- as.integer(names(y.tab[which.max(y.tab)]))
mean(y.test == y.baseline)








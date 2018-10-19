# Reference: https://tensorflow.rstudio.com/tensorflow/articles/tutorial_mnist_beginners.html
library(tensorflow)

# Load MNIST data
datasets <- tf$contrib$learn$datasets
#MNIST data splits into 3 parts
# 55,000 training data points: mnist$train
# 10,000 test data points: mnist$validation
# Images: mnist$train$images, Label: mnist$train$labrl
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = TRUE)

# Setting up the model y =softmax(Wx+b)
# Initialize placeholder and variables for the model
x <- tf$placeholder(tf$float32, shape(NULL,784L))
W <- tf$Variable(tf$zeros(shape(784L, 10L)))
b <- tf$Variable(tf$zeros(shape(10L)))
# Implement our model
# matmul is matrix multiplication
y <- tf$nn$softmax(tf$matmul(x,W) + b)

# Calculate cross entropy
# H(y) = - sumofi (y_primeofi*log(yofi))
# reduce_mean: compute mean over all the examples in the batch
y_prime <- tf$placeholder(tf$float32, shape(NULL,10L))
cross_entropy <- tf$reduce_mean(-tf$reduce_sum(y_prime*tf$log(y), reduction_indices=1L))

# optimizer with learning rate of 0.01
optimizer <- tf$train$GradientDescentOptimizer(0.01)
train_step <- optimizer$minimize(cross_entropy)

# Create session and initialize  variables
init <- tf$global_variables_initializer()
sess <- tf$Session()
sess$run(init)

# Training model
num_epochs <- 5
for (i in 1:600) {
  batches <- mnist$train$next_batch(100L)
  batch_xs <- batches[[1]]
  batch_ys <- batches[[2]]
  sess$run(train_step,feed_dict = dict(x = batch_xs, y_prime = batch_ys))
  #acc <- result[[2]]
  #sess$run(accuracy, feed_dict=dict(x = batch_xs, y_prime = batch_ys)))
  #acc <- mean(y == batch_ys)
  #cat(sprintf("Accuracy at step %s: %s", i, acc))
}

# Validate the model
# Correct_prediction: vector of booleans
correct_prediction <- tf$equal(tf$argmax(y,1L), tf$argmax(y_prime, 1L))
# What fraction are correct based on correct_prediction
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))
# Accuracy on our test data
sess$run(accuracy, feed_dict=dict(x = mnist$test$images, y_prime = mnist$test$labels))
                    

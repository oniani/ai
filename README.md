# ai

From-scratch impls of AI models, approaches, tricks, and more!

## Contents

- [Activation Functions and Their Derivative Functions from Scratch Using Pytorch](#activation-functions-and-their-derivative-functions-from-scratch-using-pytorch)
- [Functions](#functions)
- [Gradient Descent](#gradient-descent)
- [Deep Learning](#deep-learning)
- [Machine Learning Models from Scratch Using NumPy](#machine-learning-models-from-scratch-using-numpy)

### Activation Functions and Their Derivative Functions from Scratch Using Pytorch

- Sigmoid
  - [:rocket: Implementation][sigmoid]
  - [:orange_book: Theory][sigmoid_theory]
  - [:chart_with_upwards_trend: Plot][sigmoid_plot]
  - [:tv: YouTube Video: Discussing and Implementing Sigmoid and Its Derivative Using PyTorch][sigmoid_youtube]
- ReLU
  - [:rocket: Implementation][relu]
  - [:orange_book: Theory][relu_theory]
  - [:chart_with_upwards_trend: Plot][relu_plot]
  - [:tv: YouTube Video: Discussing and Implementing ReLU and Its Derivative Using PyTorch][relu_youtube]
- Leaky ReLU
  - [:rocket: Implementation][leaky_relu]
  - [:orange_book: Theory][leaky_relu_theory]
  - [:chart_with_upwards_trend: Plot][leaky_relu_plot]
  - [:tv: YouTube Video: Discussing and Implementing Leaky ReLU and Its Derivative Using PyTorch][leaky_relu_youtube]
- GELU
  - [:rocket: Implementation][gelu]
  - [:orange_book: Theory][gelu_theory]
  - [:chart_with_upwards_trend: Plot][gelu_plot]
  - [:tv: YouTube Video: Discussing and Implementing GELU and Its Derivative Using PyTorch][gelu_youtube]
- Swish
  - [:rocket: Implementation][swish]
  - [:orange_book: Theory][swish_theory]
  - [:chart_with_upwards_trend: Plot][swish_plot]
  - [:tv: YouTube Video: Discussing and Implementing GELU and Its Derivative Using PyTorch][swish_youtube]
- SERF
  - [:rocket: Implementation][serf]
  - [:orange_book: Theory][serf_theory]
  - [:chart_with_upwards_trend: Plot][serf_plot]
  - [:tv: YouTube Video: Discussing and Implementing SERF and Its Derivative Using PyTorch (r/MachineLearning special)][serf_youtube]
  - [:newspaper: r/MachineLearning reddit post by u/Shronnin: \[R\] \[D\] SERF activation function - improving Swish][serf_reddit]
- Tanh
  - [:rocket: Implementation][tanh]
  - [:orange_book: Theory][tanh_theory]
  - [:chart_with_upwards_trend: Plot][tanh_plot]
  - [:tv: YouTube Video: Discussing and Implementing Tanh and Its Derivative Using PyTorch][tanh_youtube]

### Deep Learning

- Deep Learning Project Setup (+ CNN for MNIST)
  - [:orange_book: The Reproducible MNIST][the_reproducible_mnist]
  - [:rocket: MNIST CNN Model][mnist_cnn]
  - [:tv: YouTube Video: Deep Learning Project Setup (+ CNN for MNIST)][deep_learning_setup_youtube]
- VGG Models for Image Classification
  - [:rocket: VGG Model][vgg]
  - [:tv: YouTube Video: VGG Models for Image Classification][vgg_youtube]

### Functions

- Distance Functions
  - [:rocket: Implementation][distance_functions]
  - [:orange_book: Theory][distance_functions_theory]
  - [:tv: YouTube Video: Implementing Distance Functions][distance_functions_youtube]
- Convolution
  - [:rocket: Implementation][convolution]
  - [:orange_book: Theory][convolution_theory]
  - [:tv: YouTube Video: Implementing a Convolution? (+ Baby Yoda)][convolution_youtube]

### Gradient Descent

- Discussing Batch, Stochastic, and Mini-Batch Gradient Descent
  - [:orange_book: Theory][gradient_descent_theory]
  - [:chart_with_upwards_trend: Convex and Non-Convex Functions][gradient_descent_plot]
  - [:tv: YouTube Video: Discussing Batch, Stochastic, and Mini-Batch Gradient Descent][gradient_descent_youtube]

### Machine Learning Models from Scratch Using NumPy

- Gaussian Naive Bayes
  - [:rocket: Implementation][gaussian_naive_bayes]
  - [:orange_book: Theory][gaussian_naive_bayes_theory]
  - [:tv: YouTube Video: Implementing Gaussian Naive Bayes from Scratch][gaussian_naive_bayes_youtube]
  - [:tv: YouTube Video: AI/ML Model API Design and Numerical Stability (follow-up)][api_design_and_numerical_stability_youtube]
- K-Nearest Neighbors (k-NN)
  - [:rocket: Implementation][k_nearest_neighbors]
  - [:orange_book: Theory][k_nearest_neighbors_theory]
  - [:tv: YouTube Video: Implementing K-Nearest Neighbors from Scratch][k_nearest_neighbors_youtube]
  - [:tv: YouTube Video: AI/ML Model API Design and Numerical Stability (follow-up)][api_design_and_numerical_stability_youtube]
- Linear Regression
  - [:rocket: Implementation][linear_regression]
  - [:orange_book: Theory][linear_regression_theory]
  - [:tv: YouTube Video: Implementing Linear Regression from Scratch][linear_regression_youtube]
  - [:tv: YouTube Video: AI/ML Model API Design and Numerical Stability (follow-up)][api_design_and_numerical_stability_youtube]
- Logistic Regression
  - [:rocket: Implementation][logistic_regression]
  - [:orange_book: Theory][logistic_regression_theory]
  - [:orange_book: Computing Gradients][logistic_regression_computing_gradients]
  - [:tv: YouTube Video: Implementing Linear Regression from Scratch][logistic_regression_youtube]
- K-Means Clustering
  - [:rocket: Implementation][k_means_clustering]
  - [:orange_book: Theory][k_means_clustering_theory]
  - [:tv: YouTube Video: Implementing K-Means Clustering Using NumPy][k_means_clustering_youtube]

## License

[MIT License][license]

[license]: LICENSE
[sigmoid]: activation/sigmoid.py
[sigmoid_theory]: https://en.wikipedia.org/wiki/Sigmoid_function
[sigmoid_plot]: activation/plots/sigmoid.png
[sigmoid_youtube]: https://www.youtube.com/watch?v=oxC3T_-_Amw
[relu]: activation/relu.py
[relu_theory]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
[relu_plot]: activation/plots/relu.png
[relu_youtube]: https://www.youtube.com/watch?v=93qjwrP7PfE
[leaky_relu]: activation/leaky_relu.py
[leaky_relu_theory]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Leaky_ReLU
[leaky_relu_plot]: activation/plots/leaky_relu.png
[leaky_relu_youtube]: https://www.youtube.com/watch?v=1HLKeWG0qnE
[gelu]: activation/gelu.py
[gelu_theory]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Gaussian-error_linear_unit_(GELU)
[gelu_plot]: activation/plots/gelu.png
[gelu_youtube]: https://www.youtube.com/watch?v=1HLKeWG0qnE
[swish]: activation/swish.py
[swish_theory]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#SiLU
[swish_plot]: activation/plots/swish.png
[swish_youtube]: https://www.youtube.com/watch?v=1HLKeWG0qnE
[serf]: activation/serf.py
[serf_theory]: https://arxiv.org/abs/2108.09598
[serf_plot]: activation/plots/serf.png
[serf_youtube]: https://www.youtube.com/watch?v=CLjmEuCxuT4
[serf_reddit]: https://www.reddit.com/r/MachineLearning/comments/uhgupq/r_d_serf_activation_function_improving_swish/
[tanh]: activation/tanh.py
[tanh_theory]: https://en.wikipedia.org/wiki/Hyperbolic_functions#Exponential_definitions
[tanh_plot]: activation/plots/tanh.png
[tanh_youtube]: https://www.youtube.com/watch?v=MSi1tobj-jg
[distance_functions]: function/distance.py
[distance_functions_theory]: https://en.wikipedia.org/wiki/Similarity_measure
[distance_functions_youtube]: https://www.youtube.com/watch?v=50G47n42-9o
[convolution]: function/convolution.py
[convolution_theory]: https://en.wikipedia.org/wiki/Convolution
[convolution_youtube]: https://www.youtube.com/watch?v=pmyulQwV62k
[gradient_descent_theory]: theory/gradient_descent/gradient_descent.pdf
[gradient_descent_plot]: theory/gradient_descent/convex_and_non_convex_plot.png
[gradient_descent_youtube]: https://www.youtube.com/watch?v=mV247Fe1DJc
[gaussian_naive_bayes]: model/ml/gaussian_naive_bayes.py
[gaussian_naive_bayes_theory]: https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_naive_Bayes
[gaussian_naive_bayes_youtube]: https://www.youtube.com/watch?v=maJIRFeQBVI
[k_nearest_neighbors]: model/ml/k_nearest_neighbors.py
[k_nearest_neighbors_theory]: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
[k_nearest_neighbors_youtube]: https://www.youtube.com/watch?v=8SFTAcZb9i4
[linear_regression]: model/ml/linear_regression.py
[linear_regression_theory]: https://en.wikipedia.org/wiki/Linear_regression
[linear_regression_youtube]: https://www.youtube.com/watch?v=7FdQZ9r41LU
[logistic_regression]: model/ml/logistic_regression.py
[logistic_regression_theory]: https://en.wikipedia.org/wiki/Logistic_regression
[logistic_regression_computing_gradients]: theory/gradients/logistic_regression/logistic_regression.pdf
[logistic_regression_youtube]: https://www.youtube.com/watch?v=YDa3rX9yLCE
[k_means_clustering]: model/ml/k_means_clustering.py
[k_means_clustering_theory]: https://en.wikipedia.org/wiki/K-means_clustering
[k_means_clustering_youtube]: https://www.youtube.com/watch?v=NfPGFSUM-nI
[implement]: https://www.youtube.com/watch?v=maJIRFeQBVI&list=PLG8XxYPkVOUvVzz1ZKcGAJpIBK7GRrFYR
[api_design_and_numerical_stability_youtube]: https://www.youtube.com/watch?v=BOoTX0hkO6k
[the_reproducible_mnist]: https://github.com/davidoniani/mnist
[mnist_cnn]: model/dl/mnist_cnn.py
[deep_learning_setup_youtube]: https://www.youtube.com/watch?v=2JkJZQP9dHg
[vgg]: model/dl/vgg.py
[vgg_youtube]: https://www.youtube.com/watch?v=0Ak4i2j_diM

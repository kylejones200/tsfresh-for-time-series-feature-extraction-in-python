# TSFresh for Time Series Feature Extraction in Python

TSFresh automates feature extraction from time series data by calculating hundreds of statistical characteristics and selecting the most...

### TSFresh for Time Series Feature Extraction in Python
#### TSFresh automates feature extraction from time series data by calculating hundreds of statistical characteristics and selecting the most relevant ones for machine learning tasks.
**TSFresh (Time Series Feature Extraction based on Scalable Hypothesis tests)** is designed to automatically extract features from time series data. These features are useful for ML tasks like classification, regression, and anomaly detection. So, by automating feature extraction, TSFresh saves us time (in theory).

**Automating Feature Extraction goes through** hundreds of features like mean, variance, skewness, and autocorrelation, and then filters out irrelevant or redundant features based on statistical tests. It works with univariate or multivariate time series data.

### Basic Workflow with TSFresh
To get started, we need to format the data into a specific structure, then we extract features using `extract_features` , and, optionally, we select relevant features with `select_features`.

TSFresh requires data in a long format where each time series is identified by an `id` column.

#### Simulated Time Series Data with 100 observations of 100 features


Now let's visualize the data we made.


It is beautifully noisy!

### Extracting Features
Our data looks like a mess (which is typical). So let's use `tsfresh.extract_features` to compute features from the time series.


Output (partial list of features):


### Selecting Relevant Features
We don't need all these features. Let's filter this and only keep the ones that are statistically significant using `select_features`.


This filters the extracted features to include only those that are predictive of the target variable.

### Using the Features for Supervised Learning
The purpose in doing the feature engineering is to have more useful stuff to feed into an ML model. TSFresh works with ML libraries like sklearn.

So let's look at an example of using extracted features for classification.


#### Multivariate Time Series
TSFresh supports extracting features from multiple columns in the dataset.


#### Custom Feature Extraction
You can define your own feature extraction functions using `tsfresh.feature_extraction.feature_calculators`.


And we can visualize the distribution using matplotlib.


### So what?
TSFresh is good at Feature Engineering. It automatically generate features we can use downstream in ML workflows.I worry that the features could lead to overfitting but I haven't tested that enough to know if the concern is real.

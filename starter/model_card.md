---
noteId: "df12a190733011f08b772711d3e2760d"
tags: []

---

# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

#### – **Person or Organization Developing Model**

The `MLPClassifier` is part of the [scikit-learn](https://scikit-learn.org) project, an open-source machine learning library developed and maintained by the [scikit-learn community](https://github.com/scikit-learn/scikit-learn) and primarily coordinated by [Inria](https://www.inria.fr/en), along with external contributors.

---

#### – **Model Date**

Initial release: circa **2011**
Major refactoring and stabilization in **version 0.18 (2016)**
Maintained and updated continuously (latest version: see [scikit-learn changelog](https://scikit-learn.org/stable/whats_new.html))

---

#### – **Model Version**

Depends on the installed scikit-learn version.
Example: `scikit-learn==1.5.0` → `MLPClassifier` version bundled in that release.

---

#### – **Model Type**

* **Type:** Feedforward Neural Network (Multi-Layer Perceptron - MLP)
* **Usage:** Supervised classification tasks
* **Architecture:** Fully-connected layers with configurable depth and activation functions
* **Backend:** Written in Python (NumPy)
* **Training method:** Stochastic Gradient Descent or L-BFGS

---

### **• Training Details**

#### – **Training Algorithms**

* Optimizers: `'adam'`, `'sgd'`, or `'lbfgs'`
* Uses **backpropagation** and **early stopping** if specified
* Regularization: L2 penalty (`alpha`)

#### – **Parameters**

* `hidden_layer_sizes`: tuple, default=(100,)
* `activation`: {'identity', 'logistic', 'tanh', 'relu'}
* `solver`: {'lbfgs', 'sgd', 'adam'}
* `alpha`: float (L2 penalty), default=0.0001
* `learning_rate`: constant, invscaling, adaptive
* `early_stopping`: bool
* `max_iter`: default=200

#### – **Fairness Constraints**

No built-in fairness constraints or bias mitigation. Any bias in the data will be learned by the model. Fairness must be addressed at the data pre-processing or post-processing stage.

#### – **Input Features**

* Accepts dense or sparse numerical feature matrices (`X`)
* Requires all features to be scaled (recommended: `StandardScaler`)
* Handles multiclass classification using one-vs-rest (OvR) by default

---

### **• Paper or Resource for More Information**

There is no standalone academic paper for `MLPClassifier`, but it is based on classical MLPs. Relevant resources:

* [Scikit-learn documentation for MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
* Book: *An Introduction to Statistical Learning* (James et al.)
* Book: *Pattern Recognition and Machine Learning* (Bishop, 2006)

---

### **• Citation Details**

To cite `scikit-learn` and its models, use:

```bibtex
@article{scikit-learn,
  title={Scikit-learn: Machine Learning in Python},
  author={Pedregosa, Fabian and Varoquaux, Gael and Gramfort, Alexandre and Michel, Vincent and Thirion, Bertrand and Grisel, Olivier and Blondel, Mathieu and Prettenhofer, Peter and Weiss, Ron and Dubourg, Vincent and others},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}
```

---

### **• License**

* **License:** [BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause)

---

### **• Where to Send Questions or Comments About the Model**

* **GitHub Issues:** [https://github.com/scikit-learn/scikit-learn/issues](https://github.com/scikit-learn/scikit-learn/issues)
* **User Mailing List:** [scikit-learn@python.org](mailto:scikit-learn@python.org)
* **Stack Overflow:** Use the `scikit-learn` tag for community help

---


## Intended Use

### **– Primary Intended Uses**

The `MLPClassifier` is a general-purpose neural network classifier designed for a wide range of supervised classification tasks. During development, the following use cases were envisioned:

* Binary classification tasks (e.g., spam detection, churn prediction)
* Multiclass classification problems (e.g., digit/image recognition, text classification)
* Structured/tabular data analysis where features are numerical or categorical (with proper encoding)
* Academic teaching, prototyping, and experimentation
* Benchmarking and comparison with other classifiers in the scikit-learn ecosystem
* Intermediate complexity problems that do not require deep neural network architectures

---

### **– Primary Intended Users**

The `MLPClassifier` was developed with the following primary users in mind:

* **Machine learning practitioners** needing a quick neural-network-based baseline
* **Data scientists and ML engineers** working with structured datasets
* **Students and educators** in academic and training environments
* **Researchers** doing exploratory data analysis
* **Developers** building pipelines with scikit-learn and looking for easy integration

---

### **– Out-of-Scope Use Cases**

While powerful and versatile, `MLPClassifier` is not suitable for certain tasks. It is **not intended** for:

* **Large-scale deep learning tasks** involving convolutional, recurrent, or transformer-based architectures (use frameworks like PyTorch or TensorFlow)
* **Image, audio, or text data in raw form** (unstructured data) — unless first converted into suitable feature vectors
* **GPU-accelerated training** – `MLPClassifier` runs only on CPU
* **Very large datasets** where scalability and memory efficiency are critical
* **Online learning (real-time updates)** – although partial fitting is supported, it is limited and less optimized than dedicated online learners like `SGDClassifier`
* **Production-grade high-throughput environments** – better suited for prototyping or low-volume inference

---


## Training Data
Certainly. Below is the **Training Data** section of a model card for a `MLPClassifier` trained on the **Adult Census Income Dataset**, also known as the **"Adult"** or **"Census Income"** dataset.


---

### **– Dataset Description**

The training data used for the `MLPClassifier` is the **Adult Census Income Dataset** from the UCI Machine Learning Repository. The dataset is widely used for binary classification tasks and aims to **predict whether an individual's income exceeds \$50K per year** based on a set of demographic and employment-related attributes.

**Target Variable:**

* `income`: `>50K` or `<=50K`

**Features (Attributes):**
The dataset contains **14 attributes**, including:

* **Demographic**: `age`, `sex`, `race`, `native-country`, `marital-status`
* **Employment**: `occupation`, `workclass`, `hours-per-week`, `education`, `education-num`
* **Financial**: `capital-gain`, `capital-loss`
* **Other**: `relationship`

---

### **– Source of the Data**

* **Origin**: Extracted from the 1994 U.S. Census Bureau database.
* **Access**: [UCI Repository - Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)

---

### **– Dataset Size**

* **Total records**: \~48,842 instances

  * Training set: \~32,561 instances
  * Test set: \~16,281 instances

---

### **– Class Distribution**

* **Income >50K**: \~24%
* **Income <=50K**: \~76%
  This imbalance should be considered during model training (e.g., via `class_weight='balanced'` or resampling).

---

### **– Preprocessing Applied**

* **Missing values**: Rows with missing values (notably in `workclass`, `occupation`, and `native-country`) were removed .
* **Categorical encoding**: One-hot encoding or label encoding applied to categorical variables.
* **Feature scaling**: Standardization (e.g., `StandardScaler`) applied to numerical features for stable training in neural networks.

---

### **– License**

* The dataset is in the **public domain**, but proper attribution to the **U.S. Census Bureau** and **UCI ML Repository** is recommended.


## Evaluation Data

* **Evaluation dataset**: The model is evaluated on the test split of the Adult (Census Income) dataset, with a standard 80/20 train-test ratio.
* **Distribution details**: The test set reflects the same demographics and feature distributions as the training data, including age, workclass, education, marital status, race, and gender.
* **Preprocessing**: Features were normalized, categorical variables were one-hot encoded.
* **Evaluation metrics**: Precision, Recall, and F<sub>β</sub>-score (β=1.0 by default) were computed.

---
## Metrics

### **– Model Performance Measures**

The following evaluation metrics are commonly used to assess the performance of the `MLPClassifier` in real-world classification settings:

* **Precision Score (`precision_score`)**
  Measures the proportion of true positive predictions among all predicted positives. High precision is critical in applications where false positives are costly (e.g., disease diagnosis, fraud detection).

* **Recall Score (`recall_score`)**
  Measures the proportion of true positive predictions among all actual positives. High recall is important in scenarios where missing a positive instance is costly (e.g., safety-critical alerts, medical screening).

* **F-beta Score (`fbeta_score`)**
  A weighted harmonic mean of precision and recall, allowing you to tune the importance of recall (via the β parameter).

  * β > 1 favors recall
  * β < 1 favors precision
  * Common choice: `F1 score` (β=1), a balanced metric for imbalanced classification tasks

The  model's performance on those metrics.

Evaluating model on train data...
Precision: 0.7297830374753451
Recall: 0.6491228070175439
F1: 0.6870937790157846
==============================
Evaluating model on test data...
Precision: 0.7403708987161198
Recall: 0.6607256524506684
F1: 0.698284561049445

## Ethical Considerations

* **Fairness**: Potential bias exists in the dataset (especially regarding race, sex, marital status). No fairness constraints were enforced during training.
* **Accountability**: Predictions about income can have significant real-world impacts, especially if used in hiring or loan decisions.
* **Transparency**: The model is not explainable by default, which may be problematic in regulated settings.
* **Mitigation**: We recommend post hoc analysis using SHAP or LIME, and consider rebalancing or debiasing techniques during preprocessing.

## Caveats and Recommendations

- Dataset Bias: The Adult dataset contains historical demographic biases (especially related to gender and race), which may lead to discriminatory predictions.
- Limited Interpretability: As a neural network, the MLPClassifier lacks transparency, making it difficult to explain predictions in regulated environments.
- Add Explainability Tools: Integrate SHAP or LIME to enhance model transparency.
- Perform Bias Audits: Use tools like AIF360 or Fairlearn to identify and mitigate potential biases
- Implement Continuous Monitoring: If deployed, monitor for data drift and performance degradation over time.

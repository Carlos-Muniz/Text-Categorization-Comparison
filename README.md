# Text Categorization Comparison

This project explores the necessary steps involved in Text Categorization, and performs a comparative evaluation on a variety of Vectorization and Prediction techniques from Sklearn.

## Details
The major steps of Text Categorization are:
* Preprocessing raw text into vectors of various sizes (500, 1000, 2000)
* Training and Testing models
* Evaluation of Models

### Vectorization techniques:
* Bag of Words (Sklearn's CountVectorizer)
* TFIDF (Sklearn's TfidfVectorizer)

### Models:
* Sklearn's Logistic Regression Classifier
* Sklearn's Linear Support Vector Classifier
* Sklearn's Passive Agressive Classifier
* Sklearn's SGD Classifier using Elasticnet Penalty
* Sklearn's Random Forest Classifier
* Sklearn's Perceptron
* Sklearn's K-Nearest Neighbors with 10 neighbors
* Sklearn's Multi-Layer Perceptron Classifier

### Evaluation Statistics:
* Accuracy
* Precision (micro & macro)
* Recall (micro & macro)
* F1 (micro & macro)

![Text Categorization Comparison, Top 10 Results](https://github.com/Carlos-Muniz/Text-Categorization-Comparison/blob/master/RESULTS/eval_results.png)

## Installation

1) Use the package, dependency and environment management system [conda](https://www.anaconda.com/products/individual) to install all dependencies.

```bash
conda create -f environment.yml
conda activate tcc_env
```

2) Download the necessary corpuses
```python
import nltk
nltk.download('reuters')
nltk.download('stopwords')
```


## Usage

```bash
python run.py 
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)

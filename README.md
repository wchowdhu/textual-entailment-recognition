# Textual Entailment Classification using Transfer Learning and Data Augmentation 

Kaggle has initiated a competition, [Contradictory, My Dear Watson](https://www.kaggle.com/c/contradictory-my-dear-watson/overview), to challenge machine learning practitioners to build a system that automatically classifies how pairs of sentences are related from texts in 15 different languages. The aim of this capstone project is to create a multi-class classification model using transfer learning and data augmentation. 

The final model yields an accuracy of 94% on the test dataset with a top 3% ranking currently in the leaderboard.

The final report with model visualizations and validation plots can be accessed [here](https://github.com/wchowdhu/udacity-capstone-project/blob/main/report/report.pdf).



# Dependencies

The project requires Python 3.6 and the latest version of the following libraries installed:  
  - [numpy](https://numpy.org/)
  - [pandas](https://pandas.pydata.org/)
  - [scikit-learn](https://scikit-learn.org/stable/)
  - [transformers](https://huggingface.co/transformers/)
  - [allennlp](https://github.com/allenai/allennlp)
  - [googletrans](https://pypi.org/project/googletrans/)
  - [datasets](https://github.com/huggingface/datasets)
  - [PyTorch](https://pytorch.org/)
  - [tensorflow](https://www.tensorflow.org/install)
 
To train the models, Tensor Processing Units or TPUs with 8 cores were used. TPUs are hardware accelerators specialized in deep learning tasks and are available to use for free in Kaggle. All the implementations were performed in both Tensorflow and Pytorch frameworks with Python programming language.


# Data

The dataset consists of train and test files with the following format:

- train.csv: This file contains the ID, premise, hypothesis, and label, as well as the language of the text and its two-letter abbreviation
- test.csv: This file contains the ID, premise, hypothesis, language, and language abbreviation, without labels.

We also use augmented data in the form of back-translations and auxilary datasets XNLI and MNLI. The `data` directory contains the necessary files.
 

# Code

All the notebook files (with cell outputs) and Python scripts are provided in the `notebooks` and `scripts` directories to get started with the project. 


# Run

To open the .ipynb files in your browser and look at the output of the completed cells, use the following command in your terminal after changing the working directory to the project directory `udacity-capstone-project`:
```
jupyter notebook <file_name>.ipynb
```

To run the python script files:
```
python scripts/run.py --train-file 'data/train.csv' --test-file 'data/test.csv'
```

To generate the back-translations and save them in a csv file, run the `back-translation.ipynb` or the `back-translation_textblob.ipynb` notebook files.


# Outputs

All the model outputs can be found in the `output` directory. The `predictions/1` folder contains the model predictions using all the back-translations and `predictions/0.4` contains predictions using only 40% of the back-translated examples.





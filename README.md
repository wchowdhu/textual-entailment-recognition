# Textual Entailment Recognition using Transfer Learning and Data Augmentation 

Kaggle has initiated a competition, [Contradictory, My Dear Watson](https://www.kaggle.com/c/contradictory-my-dear-watson/overview), to challenge machine learning practitioners to build a system that automatically classifies how pairs of sentences are related from texts in 15 diverse and under-represented languages. The aim of this capstone project is to create a multi-class classification system to detect entailment and contradiction in multi-lingual text using transfer learning and data augmentation. 

The final model yields an accuracy of 94% on the test dataset with a top 3% ranking in the leaderboard at the time of the competition.

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

The dataset consists of total 15,620 images. 
 

# Code

All the notebook files (with cell outputs) and Python scripts are provided  


# Run

Use the following command to train and evaluate the model:
```
jupyter notebook run.ipynb
```






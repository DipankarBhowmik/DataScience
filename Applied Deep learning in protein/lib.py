import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm # Decorate an iterable object, returning an iterator which acts exactly like the original iterable, 
                      # but prints a dynamically updating progressbar every time a value is requested
import re # used to work with Regular Expressions.
import matplotlib.pyplot as plt # Python 2D plotting library
import seaborn as sns #Seaborn is a library for making statistical graphics in Python
from pandas.plotting import table # Basic Plotting
from sklearn.feature_extraction.text import CountVectorizer # Convert a collection of text documents to a matrix of token counts
from sklearn.feature_extraction.text import TfidfVectorizer # Convert a collection of raw documents to a matrix of TF-IDF features
import time # provides various time-related functions
from datetime import timedelta #timedelta:- A duration expressing the difference between two date, time, or datetime.
from google.colab import drive # Downloading Datasets into Google Drive via Google Colab
from collections import Counter # It allows you to count the items in an iterable list
from keras.preprocessing.sequence import pad_sequences #Takes in a sequence of data-points gathered at equal intervals, along with time series parameters
                                                       #such as stride, length of history, etc., to produce batches for training/validation
from keras.utils import to_categorical  # It is used to convert array of labeled data(from 0 to nb_classes-1) to one-hot vector.
from sklearn.preprocessing import LabelEncoder # Convert categorical values into numerical values, used to encode target values, i.e. y
import random # It implements pseudo-random number generators for various distributions
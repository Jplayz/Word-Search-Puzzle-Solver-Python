"""Dummy classification system.

Dummy solution the COM2004/3004 assignment.

REWRITE THE FUNCTIONS BELOW AND REWRITE THIS DOCSTRING

version: v1.0
"""

from typing import List

import numpy as np
import scipy.linalg
import math
from utils import utils
from utils.utils import Puzzle

# The required maximum number of dimensions for the feature vectors.
N_DIMENSIONS = 20


def load_puzzle_feature_vectors(image_dir: str, puzzles: List[Puzzle]) -> np.ndarray:
    """Extract raw feature vectors for each puzzle from images in the image_dir.

    OPTIONAL: ONLY REWRITE THIS FUNCTION IF YOU WANT TO REPLACE THE DEFAULT IMPLEMENTATION

    The raw feature vectors are just the pixel values of the images stored
    as vectors row by row. The code does a little bit of work to center the
    image region on the character and crop it to remove some of the background.

    You are free to replace this function with your own implementation but
    the implementation being called from utils.py should work fine. Look at
    the code in utils.py if you are interested to see how it works. Note, this
    will return feature vectors with more than 20 dimensions so you will
    still need to implement a suitable feature reduction method.

    Args:
        image_dir (str): Name of the directory where the puzzle images are stored.
        puzzle (dict): Puzzle metadata providing name and size of each puzzle.

    Returns:
        np.ndarray: The raw data matrix, i.e. rows of feature vectors.

    """
    return utils.load_puzzle_feature_vectors(image_dir, puzzles)


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    REWRITE THIS FUNCTION AND THIS DOCSTRING

    Takes the raw feature vectors and reduces them down to the required number of
    dimensions. Note, the `model` dictionary is provided as an argument so that
    you can pass information from the training stage, e.g. if using a dimensionality
    reduction technique that requires training, e.g. PCA.

    The dummy implementation below simply returns the first N_DIMENSIONS columns.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """
    train = np.asarray(model["data_train"])

    covx = np.cov(train, rowvar=0)
    N = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals=(N - N_DIMENSIONS, N - 1))
    v = np.fliplr(v)

    reduced_data = np.dot((data - np.mean(train)), v)

    return reduced_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    REWRITE THIS FUNCTION AND THIS DOCSTRING

    This is your classifier's training stage. You need to learn the model parameters
    from the training vectors and labels that are provided. The parameters of your
    trained model are then stored in the dictionary and returned. Note, the contents
    of the dictionary are up to you, and it can contain any serializable
    data types stored under any keys. This dictionary will be passed to the classifier.

    The dummy implementation stores the labels and the dimensionally reduced training
    vectors. These are what you would need to store if using a non-parametric
    classifier such as a nearest neighbour or k-nearest neighbour classifier.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """

    # The design of this is entirely up to you.
    # Note, if you are using an instance based approach, e.g. a nearest neighbour,
    # then the model will need to store the dimensionally-reduced training data and labels
    # e.g. Storing training data labels and feature vectors in the model.
    model = {}
    model["labels_train"] = labels_train.tolist()
    model["data_train"] = fvectors_train.tolist()
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()
    return model


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Dummy implementation of classify squares.

    REWRITE THIS FUNCTION AND THIS DOCSTRING

    This is the classification stage. You are passed a list of unlabelled feature
    vectors and the model parameters learn during the training stage. You need to
    classify each feature vector and return a list of labels.

    In the dummy implementation, the label 'E' is returned for every square.

    Args:
        fvectors_train (np.ndarray): feature vectors that are to be classified, stored as rows.
        model (dict): a dictionary storing all the model parameters needed by your classifier.

    Returns:
        List[str]: A list of classifier labels, i.e. one label per input feature vector.
    """

    train = np.asarray(model["fvectors_train"])
    train_labels = np.asarray(model["labels_train"])

    
    # Using all the features from the training and test data
    features = np.arange(0, train.shape[1])

    #features = (0, 2, 4, 7, 8 ,10, 11, 13, 19)
    train = train[:, features]
    test = fvectors_test[:, features]


    # Super compact implementation of nearest neighbour
    x = np.dot(test, train.transpose())
    modtest = np.sqrt(np.sum(test * test, axis=1))
    modtrain = np.sqrt(np.sum(train * train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose())
    # cosine distance
    nearest = np.argmax(dist, axis=1)
    mdist = np.max(dist, axis=1)
    labels = train_labels[nearest]

    return labels


def find_words(labels: np.ndarray, words: List[str], model: dict) -> List[tuple]:
    """Dummy implementation of find_words.

    REWRITE THIS FUNCTION AND THIS DOCSTRING

    This function searches for the words in the grid of classified letter labels.
    You are passed the letter labels as a 2-D array and a list of words to search for.
    You need to return a position for each word. The word position should be
    represented as tuples of the form (start_row, start_col, end_row, end_col).

    Note, the model dict that was learnt during training has also been passed to this
    function. Most simple implementations will not need to use this but it is provided
    in case you have ideas that need it.

    In the dummy implementation, the position (0, 0, 1, 1) is returned for every word.

    Args:
        labels (np.ndarray): 2-D array storing the character in each
            square of the wordsearch puzzle.
        words (list[str]): A list of words to find in the wordsearch puzzle.
        model (dict): The model parameters learned during training.

    Returns:
        list[tuple]: A list of four-element tuples indicating the word positions.
    """
    
    
    """labels = [['Y', 'E', 'A', 'S', 'T', 'E', 'W', 'E', 'N', 'E', 'D', 'S', 'E', 'N', 'N'],
             ['R', 'B', 'S', 'P', 'E', 'L', 'H', 'G', 'E', 'D', 'N', 'R', 'S', 'I', 'U'],
             ['I', 'T', 'A', 'E', 'D', 'A', 'E', 'R', 'B', 'T', 'A', 'L', 'F', 'P', 'B'],
             ['T', 'B', 'E', 'G', 'C', 'I', 'A', 'B', 'A', 'T', 'T', 'A', 'L', 'N', 'D'],
             ['A', 'R', 'E', 'P', 'U', 'I', 'T', 'S', 'O', 'Y', 'H', 'G', 'U', 'O', 'D'],
             ['P', 'A', 'E', 'Y', 'M', 'E', 'D', 'M', 'A', 'L', 'T', 'L', 'O', 'A', 'F'],
             ['A', 'S', 'T', 'A', 'R', 'U', 'T', 'Y', 'T', 'S', 'U', 'R', 'C', 'L', 'B'],
             ['H', 'R', 'L', 'T', 'T', 'A', 'R', 'T', 'G', 'N', 'S', 'I', 'I', 'A', 'T'],
             ['C', 'B', 'S', 'I', 'E', 'T', 'N', 'C', 'E', 'T', 'K', 'N', 'T', 'N', 'S'],
             ['E', 'A', 'R', 'O', 'C', 'H', 'I', 'A', 'E', 'R', 'I', 'O', 'I', 'N', 'A'],
             ['L', 'N', 'E', 'E', 'I', 'E', 'C', 'P', 'R', 'N', 'N', 'A', 'L', 'B', 'O'],
             ['A', 'N', 'B', 'P', 'A', 'E', 'D', 'S', 'A', 'G', 'R', 'Y', 'S', 'L', 'T'],
             ['E', 'O', 'O', 'A', 'T', 'D', 'C', 'P', 'U', 'G', 'C', 'R', 'E', 'O', 'F'],
             ['M', 'C', 'D', 'I', 'G', 'R', 'S', 'E', 'I', 'R', 'R', 'E', 'N', 'O', 'O'],
             ['E', 'K', 'H', 'I', 'U', 'E', 'H', 'T', 'O', 'E', 'B', 'K', 'S', 'M', 'C'],
             ['L', 'W', 'L', 'M', 'O', 'C', 'L', 'U', 'I', 'G', 'M', 'A', 'H', 'E', 'A'],
             ['O', 'R', 'B', 'R', 'O', 'U', 'T', 'E', 'N', 'C', 'I', 'B', 'S', 'R', 'C'],
             ['H', 'S', 'O', 'I', 'M', 'O', 'Y', 'E', 'O', 'D', 'K', 'L', 'O', 'R', 'C'],
             ['W', 'P', 'R', 'L', 'N', 'M', 'U', 'F', 'F', 'I', 'N', 'S', 'N', 'C', 'I'],
             ['I', 'B', 'E', 'S', 'L', 'S', 'H', 'C', 'I', 'W', 'D', 'N', 'A', 'S', 'A']] """
    

    allWordPositions = []

    for word in words:
        word = word.upper()
        found = False
        for x in range(len(labels)):
            for y in range(len(labels[x])):
                if labels[x][y] == word[0]:
                    if checkRowForward(labels, word, x, y) != None:
                        allWordPositions.append(checkRowForward(labels, word, x , y))
                        found = True

                    elif checkRowBackward(labels, word, x, y) != None:
                        allWordPositions.append(checkRowBackward(labels, word, x , y))
                        found = True

                    elif checkColumnDown(labels, word, x, y) != None:
                        allWordPositions.append(checkColumnDown(labels, word, x , y))
                        found = True

                    elif checkColumnUp(labels, word, x, y) != None:
                        allWordPositions.append(checkColumnUp(labels, word, x , y))  
                        found = True
                
                    elif checkDiagonalDownRight(labels, word, x, y) != None:
                        allWordPositions.append(checkDiagonalDownRight(labels, word, x , y))   
                        found = True
                    
                    elif checkDiagonalDownLeft(labels, word, x, y) != None:
                        allWordPositions.append(checkDiagonalDownLeft(labels, word, x , y))
                        found = True

                    elif checkDiagonalUpRight(labels, word, x, y) != None:
                        allWordPositions.append(checkDiagonalUpRight(labels, word, x , y))   
                        found = True
                    
                    elif checkDiagonalUpLeft(labels, word, x, y) != None:
                        allWordPositions.append(checkDiagonalUpLeft(labels, word, x , y))    
                        found = True

        if found == False:
           allWordPositions.append((0,0,0,0))         

    return allWordPositions


# This checks the row going forward
def checkRowForward(labels: np.ndarray, word: str, x: int, y: int):
    counter = 0
    checkPercent = math.floor(len(word) / 3)
    for w in range(len(word)):
        if (y + w) < len(labels[x]):
            if w == (len(word) - 1) and (labels[x][y+w] == word[w]):
                return (x, y, x, y+w)

            elif labels[x][y+w] == word[w]:
                continue
            
            else:
                break

        else:
            break
    return 

# This checks the row going backwards
def checkRowBackward(labels: np.ndarray, word: str, x: int, y: int):
    for w in range(len(word)):
        if (y - w) > 0:
            if w == (len(word) - 1) and (labels[x][y-w] == word[w]):
                return (x, y, x, y-w)
            elif labels[x][y-w] == word[w]:
                continue

            else:
                break

        else:
            break
    return

def checkColumnDown(labels: np.ndarray, word: str, x: int, y: int):
    for w in range(len(word)):
        if (x + w) < len(labels):
            if w == (len(word) - 1) and (labels[x+w][y] == word[w]):
                return (x, y, x+w, y)
            elif labels[x+w][y] == word[w]:
                continue

            else:
                break

        else:
            break
    return 


def checkColumnUp(labels: np.ndarray, word: str, x: int, y: int):
    for w in range(len(word)):
        if (x - w) >= 0:
            if (w == (len(word) - 1)) and (labels[x-w][y] == word[w]):
                return (x, y, x-w, y)
            elif labels[x-w][y] == word[w]:
                continue

            else:
                break

        else:
            break
    return

def checkDiagonalDownRight(labels: np.ndarray, word: str, x: int, y: int):
    for w in range(len(word)):
        if ((x + w) < len(labels)) and((y + w) < len(labels[x])):
            if w == (len(word) - 1) and labels[x+w][y+w] == word[w]:
                return (x, y, x+w, y+w)
            elif labels[x+w][y+w] == word[w]:
                continue

            else:
                break
        else:
            break
    return

def checkDiagonalDownLeft(labels: np.ndarray, word: str, x: int, y: int):
    for w in range(len(word)):
        if ((x + w) < len(labels)) and((y - w) > 0):
            if w == (len(word) - 1) and labels[x+w][y-w] == word[w]:
                return (x, y, x+w, y-w)
            elif labels[x+w][y-w] == word[w]:
                continue

            else:
                break
        else:
            break
    return

def checkDiagonalUpRight(labels: np.ndarray, word: str, x: int, y: int):
    for w in range(len(word)):
        if ((x - w) > 0) and((y + w) < len(labels[x])):
            if w == (len(word) - 1) and labels[x-w][y+w] == word[w]:
                return (x, y, x-w, y+w)
            elif labels[x-w][y+w] == word[w]:
                continue

            else:
                break
        else:
            break
    return

def checkDiagonalUpLeft(labels: np.ndarray, word: str, x: int, y: int):
    for w in range(len(word)):
        if ((x - w) > 0) and((y - w) > 0):
            if w == (len(word) - 1) and labels[x-w][y-w] == word[w]:
                return (x, y, x-w, y-w)
            elif labels[x-w][y-w] == word[w]:
                continue

            else:
                break
        else:
            break
    return

def listToNumPy(data):
    charArray = []
    for x in range(len(data)):
        charArray.append(data[x])

    return charArray


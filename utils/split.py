import os
import shutil
import random
import argparse
from tqdm import tqdm


def splitFolder(path, outputFolder, splitRatio):
    """
    Recursively split the contents of a folder into two new folders,
    with a specified split ratio.

    Parameters
    ----------
    path : str
        The path to the folder to be split.
    outputFolder : str
        The path to the folder where the split data will be stored.
    splitRatio : float
        The split ratio between the train and valid sets.
        Should be a number between 0 and 1.
    """
    # Make sure the output folder exists
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    # Recursively iterate over all files and folders in the original folder
    for root, dirs, files in tqdm(os.walk(path), desc="Splitting files"):
        # Randomly shuffle the files
        random.shuffle(files)

        # Calculate the number of files in the train and valid sets
        numFiles = len(files)
        numValid = int(numFiles * splitRatio)
        numTrain = numFiles - numValid

        # Split the files into train and valid sets
        trainFiles = files[:numTrain]
        validFiles = files[numTrain:]

        # Copy the files to their respective folders in the output folder
        for file in tqdm(trainFiles, desc="Copying training files"):
            srcPath = os.path.join(root, file)
            dstPath = os.path.join(outputFolder, "train", root[len(path):], file)
            os.makedirs(os.path.dirname(dstPath), exist_ok=True)
            shutil.copy(srcPath, dstPath)

        for file in tqdm(validFiles, desc="Copying validation files"):
            srcPath = os.path.join(root, file)
            dstPath = os.path.join(outputFolder, "valid", root[len(path):], file)
            os.makedirs(os.path.dirname(dstPath), exist_ok=True)
            shutil.copy(srcPath, dstPath)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Recursively split the contents of a folder into two new folders.")
    parser.add_argument("path", type=str, help="The path to the folder to be split.")
    parser.add_argument("outputFolder", type=str, help="The path to the folder where the split data will be stored.")
    parser.add_argument("--splitRatio", type=float, default=0.1,
                        help="The split ratio between the train and valid sets. Default is 0.2.")
    args = parser.parse_args()

    # Call the splitFolder function with the command line arguments
    splitFolder(args.path, args.outputFolder, args.splitRatio)

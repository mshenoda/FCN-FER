import argparse
import os
import pandas as pd


def delete_images(csv_file):
    # Load CSV into pandas DataFrame
    df = pd.read_csv(csv_file)

    # Iterate over rows and delete images that don't match label
    deleted_images = []
    for index, row in df.iterrows():
        image_path = row["path"]
        label = row["label"]
        if not image_path.startswith(label):
            os.remove(image_path)
            deleted_images.append(image_path)

    return deleted_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete images that don't match their label")
    parser.add_argument("csv_file", type=str, help="path to CSV file")
    args = parser.parse_args()

    deleted_images = delete_images(args.csv_file)
    print("Deleted images:", deleted_images)

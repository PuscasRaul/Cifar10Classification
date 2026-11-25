#!/usr/bin/env python3
import glob as gl
import logging
import os
import matplotlib.pyplot
import numpy as np
from typing import Any

from src.core.CifarDataset import CifarDataset
from src.core.Classifier.DecisionTree import DecisionTree, StoppageHeuristic
from src.core.FeatureExtraction.HandcraftedFeatureExtractor import (
    HandcraftedFeatureExtractor
)
from src.core.Classifier.RandomForest import RandomForest
from src.core.metrics import evaluate
from src.core.models.Settings import settings

pudb_logger = logging.getLogger("pudb")
pudb_logger.setLevel(logging.INFO)
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)
matplotlib.pyplot.set_loglevel(level='warning')
logging.basicConfig()
logging.root.setLevel(logging.NOTSET)
logging.basicConfig(level=logging.NOTSET)

database_root_dir = os.path.abspath(settings.database_dir)
databse_files = gl.glob(database_root_dir + "/" + "data_batch_[1-9]")
test_file = gl.glob(database_root_dir + "/" + "test_batch")
loaded_db = [CifarDataset(file) for file in databse_files]
test_db = [CifarDataset(test) for test in test_file]

# Add a normalization process?
# Normalization is done so that all the images have a common scale,
# it allows the models to work as a normal distribution.

# x_normalized = (x^(j) - min^(j)) / max^(j) - min^(j)

feature_extractor = HandcraftedFeatureExtractor(100)

# TODO: Maybe ask some AI about what depth is common
dt_stoppage: dict[StoppageHeuristic, Any] = {
    StoppageHeuristic.DEPTH: 15,
}

rf_stoppage: dict[StoppageHeuristic, Any] = {
    StoppageHeuristic.LEAF_SIZE: 30
}

NO_OF_FOREST_TREES = 100


def main():
    train_x = np.concatenate([db.get_X() for db in loaded_db], axis=0)
    train_y = np.concatenate([db.get_Y() for db in loaded_db], axis=0)

    features = feature_extractor.feature_extraction(train_x)

    forest_trees = []
    k_subsamples = round(np.sqrt(np.shape(features)[-1]))
    for i in range(NO_OF_FOREST_TREES):
        forest_trees.append(
            DecisionTree(rf_stoppage,  k_subsamples)
        )

    dt = DecisionTree(dt_stoppage)
    rf = RandomForest(forest_trees, sample_ratio=0.75)

    rf.fit(features, train_y)
    dt.fit(features, train_y)

    evaluate(dt, features, train_y, "train")
    evaluate(rf, features, train_y, "train")
    # TODO: extract that into a function atleast

    test_x = np.concatenate([db.get_X() for db in test_db], axis=0)
    test_y = np.concatenate([db.get_Y() for db in test_db], axis=0)

    test_features = feature_extractor.feature_extraction(
        test_x
    )

    evaluate(dt, test_features, test_y, "test")
    evaluate(rf, test_features, test_y, "test")


if __name__ == "__main__":
    main()

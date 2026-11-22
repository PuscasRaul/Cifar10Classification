#!/usr/bin/env python3
import glob as gl
import logging
import os
import matplotlib.pyplot
from typing import Any

from src.core.CifarDataset import CifarDataset
from src.core.Classifier.DecisionTree import DecisionTree, StoppageHeuristic
from src.core.FeatureExtraction.HandcraftedFeatureExtractor import (
    HandcraftedFeatureExtractor
)
from src.core.Classifier.RandomForest import RandomForest
from src.core.metrics import evaluate
from src.core.models.Settings import settings
from src.core.Preprocessing.PreprocessorV1 import PreprocessorV1

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

pre_processor = PreprocessorV1(32)

# TODO: Maybe ask some AI about what depth is common
stoppage: dict[StoppageHeuristic, Any] = {
   StoppageHeuristic.DEPTH: 15
}


# TODO: add plots that work for both DecisionTree + RandomForest
def _train(classifier, x, y):
    await classifier.fit(x, y)
    evaluate(classifier, x, y, "train")


def main():
    for database in loaded_db:
        processed = pre_processor.process_vectors(database.get_X())
        feature_extractor = HandcraftedFeatureExtractor(processed, 100)
        features = feature_extractor.feature_extraction()
        y = database.get_Y()

        dt = DecisionTree(stoppage)
        rf = RandomForest(k=10, heuristic=stoppage, sample_ratio=0.45)

        await _train(rf, features, y)
        await _train(dt, features, y)

        for test in test_db:
            x_test = test.get_X()
            test_processed = pre_processor.process_vectors(x_test)
            test_features = feature_extractor.feature_extraction(
                test_processed
            )
            y_test = test.get_Y()
            evaluate(dt, test_features, y_test, "test")
            evaluate(rf, test_features, y_test, "test")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import glob as gl
import os
from typing import Any

from src.core.CifarDataset import CifarDataset
from src.core.Classifier.DecisionTree import DecisionTree, StoppageHeuristic
from src.core.FeatureExtraction.HandcraftedFeatureExtractor import (
    HandcraftedFeatureExtractor
)
from src.core.metrics import evaluate, plot_feature_importance, depth_sweep
from src.core.models.Settings import settings
from src.core.Preprocessing.PreprocessorV1 import PreprocessorV1

database_root_dir = os.path.abspath(settings.database_dir)
databse_files = gl.glob(database_root_dir + "/" + "data_batch_[1-9]")
test_file = gl.glob(database_root_dir + "/" + "test_batch")
loaded_db = [CifarDataset(file) for file in databse_files]
test_db = [CifarDataset(test) for test in test_file]

pre_processor = PreprocessorV1(32)

# TODO: Maybe ask some AI about what depth is common
stoppage: dict[StoppageHeuristic, Any] = {
   StoppageHeuristic.DEPTH: 10
}

# TODO: Add some traceability, show how the shit is performing etc etc

for database in loaded_db:
    processed = pre_processor.process_vectors(database.get_X())
    # For the moment hardcode some values
    # TODO: Change this so it does not take it as a constructor paramter
    #       but rather as a function one for the feature extraction method
    feature_extractor = HandcraftedFeatureExtractor(processed, 100)
    features = feature_extractor.feature_extraction()

    y = database.get_Y()
    classifier = DecisionTree(stoppage)
    classifier.fit(features, y)

    evaluate(classifier, features, y, "train")
    plot_feature_importance(classifier.feature_importances)

    for test in test_db:
        x_test = test.get_X()
        y_test = test.get_Y()
        evaluate(classifier, x_test, y_test, "test")
        depth_sweep(
            lambda d: DecisionTree({StoppageHeuristic.DEPTH: d}),
            features, y, x_test, y_test, depths=range(1, 12),
        )

#!/usr/bin/env python3
from dotenv import load_dotenv
import glob as gl
import os
from typing import Any

from src.core.CifarDataset import CifarDataset
from src.core.Classifier.DecisionTree import DecisionTree, StoppageHeuristic
from src.core.FeatureExtraction.HandcraftedFeatureExtractor import (
    HandcraftedFeatureExtractor
)
from src.core.Preprocessing.PreprocessorV1 import PreprocessorV1

load_dotenv()
database_root_dir = os.path.abspath(os.environ["DATABASE_DIR"])
databse_files = gl.glob(database_root_dir + "/" + "data_batch_[1-9]")
loaded_db = [CifarDataset(file) for file in databse_files]

pre_processor = PreprocessorV1(32)

# TODO: Maybe ask some AI about what depth is common
stoppage: dict[StoppageHeuristic, Any] = {
   StoppageHeuristic.DEPTH: 250
}

# TODO: Add some traceability, show how the shit is performing etc etc
#       Plot the PCA, etc

for database in loaded_db:
    processed = pre_processor.process_vectors(database.get_X())
    # For the moment hardcode some values
    # TODO: Change this so it does not take it as a constructor paramter
    #       but rather as a function one for the feature extraction method
    feature_extractor = HandcraftedFeatureExtractor(processed, 100)
    features = feature_extractor.feature_extraction()
    classifier = DecisionTree(stoppage)
    classifier.fit(features, database.get_Y())

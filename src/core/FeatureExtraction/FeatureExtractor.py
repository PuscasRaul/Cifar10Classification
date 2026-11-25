from src.core.FeatureExtraction.IFeatureExtractor import IFeatureExtractor


class FeatureExtractor(IFeatureExtractor):

    def __init__(self, k) -> None:
        IFeatureExtractor.__init__(self, k)

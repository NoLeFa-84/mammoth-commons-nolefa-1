from mammoth_commons.models.model import Model


class ResearcherRanking(Model):
    def __init__(self, ranking_function, baseline_ranking_function=None):
        super().__init__()
        self.rank = ranking_function
        self.baseline_rank = baseline_ranking_function

    def predict(self, dataset, sensitive):
        assert (
            len(sensitive) == 1
        ), "Researcher ranking accepts only one sensitive attribute"
        return self.rank(dataset, sensitive[0])

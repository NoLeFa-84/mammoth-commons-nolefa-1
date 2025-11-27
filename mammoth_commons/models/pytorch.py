from mammoth_commons.models.predictor import Predictor
from mammoth_commons.datasets.image_pairs import ImagePairs


def get_predictions(threshold, embed1, embed2):
    import torch

    embed1 = torch.nn.functional.normalize(embed1, p=2, dim=1)
    embed2 = torch.nn.functional.normalize(embed2, p=2, dim=1)
    diff = embed1 - embed2
    dist = 1 - torch.exp(-torch.sum(diff**2, dim=1) / 2)
    if threshold == 0:
        threshold = 0.5
    predict_issame = (dist < threshold).int()
    return predict_issame


class Pytorch(Predictor):
    def __init__(self, model, threshold=0):
        super().__init__()
        assert (
            0 <= threshold < 1
        ), "The model threshold should be either in the range (0,1) or zero to be automatically determined."
        self.model = model
        self.threshold = threshold

    def predict(self, dataset, sensitive):
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model.to(device)
        dataloader = dataset.to_torch(sensitive)

        model.eval()
        all_predictions = []
        all_labels = []
        all_sensitive = [[] for _ in sensitive]
        is_numerical = True
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(dataset, ImagePairs):
                    input1 = batch[0].to(device)
                    input2 = batch[1].to(device)
                    targets = batch[2]
                    sens = batch[3]
                    output1 = model(input1)
                    output2 = model(input2)
                    predictions = get_predictions(self.threshold, output1, output2)
                else:
                    assert (
                        self.threshold == 0
                    ), "The loaded dataset is not multiclass, and therefore does not accept models with non-zero thresholds."
                    inputs = batch[0].to(device)
                    targets = batch[1]
                    sens = batch[2]
                    outputs = model(inputs)
                    predictions = torch.argmax(outputs, dim=1)
                all_predictions.append(predictions.cpu())
                all_labels.append(targets.cpu())

                if isinstance(sens[0], tuple):
                    for i in range(len(sensitive)):
                        for j in range(len(sens)):
                            all_sensitive[i] += sens[i]
                    is_numerical = False
                else:
                    assert torch.is_tensor(
                        sens[0]
                    ), "dataloader should return tensors (for numerical sensitive values) or tuples (for categorical sensitive values)"
                    for i in range(len(sensitive)):
                        all_sensitive[i] += [sens[i].cpu() for i in range(len(sens))]
                    is_numerical = True
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        dataset.labels = {"0": 1 - all_labels, "1": all_labels}
        dataset.data = (
            {name: torch.cat(value) for name, value in zip(sensitive, all_sensitive)}
            if is_numerical
            else {name: value for name, value in zip(sensitive, all_sensitive)}
        )
        return all_predictions

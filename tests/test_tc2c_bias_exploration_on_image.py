from mammoth_commons import testing
from mai_bias.catalogue.dataset_loaders.images import data_images
from mai_bias.catalogue.model_loaders.pytorch import model_torch
from mai_bias.catalogue.metrics.interactive_report import interactive_report


def test_bias_exploration():
    with testing.Env(data_images, model_torch, interactive_report) as env:
        dataset = env.data_images(
            path="./data/xai_images/bupt_anno.csv",
            image_root_dir="./data/xai_images/race_per_7000",
            target="task",
            data_transform_path="./data/xai_images/torch_transform.py",
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        model = env.model_torch(
            state_path="./data/torch_model/resnet18.pt",
            model_path="./data/torch_model/torch_model.py",
        )
        env.interactive_report(dataset, model, ["protected"]).show()


if __name__ == "__main__":
    test_bias_exploration()

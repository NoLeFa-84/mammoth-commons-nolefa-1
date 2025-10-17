from mammoth_commons import testing
from mai_bias.catalogue.dataset_loaders.image_pairs import data_image_pairs
from mai_bias.catalogue.model_loaders.pytorch import model_torch
from mai_bias.catalogue.metrics.interactive_report import interactive_report


def test_bias_exploration():
    with testing.Env(data_image_pairs, model_torch, interactive_report) as env:
        dataset = env.data_image_pairs(
            path="./data/xai_images/bupt_pairs_anno.csv",
            image_root_dir="./data/xai_images/race_per_7000",
            target="is_same",
            data_transform_path="./data/xai_images/torch_transform_fv.py",
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        model = env.model_torch(
            state_path="./data/torch_model/toy_adaface.pth",
            model_path="./data/torch_model/adaface_toy.py",
        )
        env.interactive_report(dataset, model, ["race"]).show()


if __name__ == "__main__":
    test_bias_exploration()

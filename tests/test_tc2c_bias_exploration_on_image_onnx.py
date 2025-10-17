from mammoth_commons import testing
from mai_bias.catalogue.dataset_loaders.images import data_images
from mai_bias.catalogue.model_loaders.pytorch2onnx import model_torch2onnx

# from mai_bias.catalogue.metrics.interactive_report import interactive_report
from mai_bias.catalogue.metrics.model_card import model_card


def test_bias_exploration():
    with testing.Env(data_images, model_torch2onnx, model_card) as env:
        dataset = env.data_images(
            path="./data/xai_images/bupt_anno.csv",
            image_root_dir="./data/xai_images/race_per_7000",
            target="task",
            data_transform_path="./data/xai_images/torch_transform.py",
            batch_size=4,
            shuffle=False,
        )
        model = env.model_torch2onnx(
            state_path="./data/torch_model/resnet18.pt",
            model_path="./data/torch_model/torch_model.py",
            input_width=dataset.input_size[0],
            input_height=dataset.input_size[1],
        )
        env.model_card(dataset, model, ["protected"], problematic_deviation=0).show()


if __name__ == "__main__":
    test_bias_exploration()

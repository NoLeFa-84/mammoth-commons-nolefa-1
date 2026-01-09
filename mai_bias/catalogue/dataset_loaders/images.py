from mammoth_commons.datasets import Image
from mammoth_commons.integration import loader
from mammoth_commons.externals import safeexec


@loader(
    namespace="mammotheu",
    version="v054",
    python="3.13",
    packages=("torch", "torchvision", "pandas"),
)
def data_images(
    path: str = "",
    image_root_dir: str = "",
    target: str = "",
    batch_size: int = 4,
    shuffle: bool = False,
    data_transform_path: str = "",
    transform_variable: str = "transform",
    num_workers: int = 0,
    safe_libraries="numpy,torch,torchvision,PIL,io,requests,urllib",
) -> Image:
    """<img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/image.png?raw=true"
    alt="image" style="float: left; margin-right: 15px; height: 36px;"/>

    <h3>images with metadata</h3>
    Loads image data from a CSV file that contains their sensitive and predictive attribute
    data, as well as paths relative to a root directory. Loaded images are accompanied by a preprocessing transformation.

    Args:
        path: The path to the CSV file containing information about the dataset.
        image_root_dir: The root directory where the actual image files are stored.
        target: Indicates the predictive attribute in the dataset.
        batch_size: The batch size at which images should be loaded.
        shuffle: Whether to shuffle the loaded images.
        data_transform_path: A path or implementation of a torchvision data transform. Alternatively, paste the transformation code here.
        transform_variable: The transformation target variable that should be extracted after the namesake code runs.
        num_workers: Number of subprocesses to use for data loading.
        safe_libraries: A comma-separated list of safe libraries that are allowed in the transformation code. As a safety measure against code injection attacks, an error will be created if libraries other than those are encountered.
    """
    from mammoth_commons.externals import pd_read_csv, prepare

    batch_size = int(batch_size)
    num_workers = int(num_workers)
    premature_data = pd_read_csv(path, nrows=1)  # just read one row for verification
    data_transform = safeexec(
        data_transform_path,
        out=transform_variable,
        whitelist=[lib.strip() for lib in safe_libraries.split(",")],
    )
    path = prepare(path)

    dataset = Image(
        path=path,
        root_dir=image_root_dir,
        target=target,
        data_transform=data_transform,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        cols=[col for col in premature_data],
    )

    return dataset

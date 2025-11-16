from mammoth_commons.datasets import ImagePairs
from mammoth_commons.integration import loader
from mammoth_commons.externals import safeexec


@loader(
    namespace="mammotheu",
    version="v053",
    python="3.13",
    packages=("torch", "torchvision", "pandas"),
)
def data_image_pairs(
    path: str = "",
    image_root_dir: str = "./",
    target: str = "",
    batch_size: int = 4,
    shuffle: bool = False,
    data_transform_path: str = "",
    transform_variable: str = "transform",
    num_workers: int = 0,
    safe_libraries="numpy,torch,torchvision,PIL,io,requests,urllib",
) -> ImagePairs:
    """
    Loads image pairs declared in a CSV file.
    The expected format is to have the first image's identifier in the first column,
    and the second image's identifier in the second column, Sensitive attributes
    can be selected from the rest of the columns. The images identifiers read from the columns
    are transformed to loading paths by string specifications that can contain the
    symbols: {root} to refer to the root directory, {col} to refer to the column name, and {id}
    to refer to the column entry.

    Args:
        path: The path to the CSV file containing information about the dataset.
        image_root_dir: The root directory where the actual image files are stored.
        target: Indicates the predictive attribute in the dataset.
        data_transform: A path or implementation of a torchvision data transform.
        batch_size: The number of image pairs in each batch.
        shuffle: Whether to shuffle the dataset.
        data_transform_path: A path or implementation of a torchvision data transform. Alternatively, paste the transformation code here.
        transform_variable: The transformation target variable that should be extracted after the namesake code runs.
        num_workers: Number of subprocesses to use for data loading.
        safe_libraries: A comma-separated list of safe libraries that are allowed in the transformation code. As a safety measure against code injection attacks, an error will be created if libraries other than those are encountered.
    """
    from mammoth_commons.externals import pd_read_csv

    batch_size = int(batch_size)
    num_workers = int(num_workers)
    premature_data = pd_read_csv(path, nrows=1)  # just read one row for verification

    data_transform = safeexec(
        data_transform_path,
        out=transform_variable,
        whitelist=[lib.strip() for lib in safe_libraries.split(",")],
    )

    dataset = ImagePairs(
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

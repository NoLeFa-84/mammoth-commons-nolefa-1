# Create Modules

This document contains instructions on how to contribute modules to the MAMMOth catalogue 
so that they are included in the MAI-BIAS desktop application and server toolkit. Modules depend on the
MAMMOTH-commons library's types. To contribute to the main
library (for example, to add data types) see [here](mammoth_commons/README.md).
Instructions on how to manually build modules or how to trigger continuous integration
as a maintainer are provided [here](mai_bias/catalogue/README.md).

**The catalogue may be hosted in a different repository in the future.**

1. [Create a module](#create-a-module)
2. [Add tests](#locally-test-a-module)
3. [Write documentation](#write-documentation)


## Create a module

Create a fork of the repository. You may work on the `dev` branch and 
create pull requests that repository maintainers will try to merge.
Those requests will trigger continuous integration actions to verify
that contributions are compliant with all technical requirements of the toolkit.
Use the *black* linter, though this can be fixed upon merging too. 
Pull requests with errors other than linting ones will be rejected.

Creating a module is as simple as adding a file in the `mai_bias/catalogue/`
directory, adding an function with typehints, and decorating the latter.
The decorator works as a buffer between
your code and various interfaces. Here are some details: 

1. *Type dependencies.* Import the necessary dataset or model classes
from `mammoth_commons.datasets` and `mammoth_commons.models` respectively. 
Use them to annotate your method's argument
and return types. *Type annotations are mandatory for 
all arguments.*

2. *Parameters.* In addition to some mandatory positional
arguments for each type of module, you may add any number of 
`str`, `bool`, `int` or `float` keyword arguments. These
serve as parameters with default values, where the default `None` should
be set if no common default is known beforehand. 
If a string is an enumeration of different options, prefer replacing the `str` type with
`Options("option1", "option2", ...)`. All these requirements help the toolkit understand
what information to give to the users working with your module. Use the substrings *path*
to have a loading dialog in MAI-BIAS, *delimiter* to enable automatic detection of delimiters,
as well as one of *numeric*, *categorical*, *attribute*, *ignored*, or *target*
to indicate to the UI that it should try to select among CSV column names in provided data by
peeking at them. Delimiters and column names are recognized to
correspond to the last previous path.

3. You must also create
a docstring for your module. This should include both the main description
and parameter descriptions under an `Args:` section (the title of this section is mandatory). 
The parameter descriptions should follow the convention `name: description` and not
specify any type. You cannot have line breaks in the parameter description.

4. *Decorators.* Decorate your module with either the 
`@mammoth.integration.metric(namespace, version, python="3.13", packages=(...))` or 
the `@mammoth.integration.loader(namespace, version, python="3.13", packages=(...))` decorator. 
These require at least one argument to denote
the module's version. The namespace refers to whom the module
should be accredited to (if you are not using continuous integration, it should be the same as your DockerHub 
username). Finally, packaged dependencies to be a tuple of strings
(take care to write something like `packages=("pandas",)` **comma included** if you only have one dependency).
These dependencies are any packages other than the few found in `requirements.txt`, and need to include
any dependencies. For example, add *pandas* as a package dependency if you use or load the `CSV` datatype
because it is needed there. Note that mammoth-commons imports packages for its datatypes only at the
last necessary moment.

5. As a last step, make sure that you add your module to MAI-BIAS desktop by registering it
both in `mai_bias/backend/loaders.py` and in `mai_bias/backend/catalogue_loaders.py`.

Here are some examples of modules:

<details>
<summary>Example metric</summary>

```python
from mammoth_commons.datasets import CSV
from mammoth_commons.models import ONNX
from mammoth_commons.exports import Markdown
from typing import Dict, List
from mammoth_commons.integration import metric


@metric(namespace="...", version="v001", python="3.13")
def new_metric(
        dataset: CSV,
        model: ONNX,
        sensitive: List[str],
        threshold: 0.5
) -> Markdown:
    """Write your metric's description here.
    Args:
        threshold: This is some user-provided threshold.
    """
    return Markdown("#Results\nThese are the results.")
```
</details>


<details>
<summary>Example dataset loader</summary>

```python
from mammoth_commons.datasets import CSV
from mammoth_commons.integration import loader
from mammoth_commons.externals import pd_read_csv
from typing import List, Optional


@loader(
    namespace="maniospas",
    version="v001",
    python="3.13",
    packages=("pandas",),
)
def categorical_csv(
        path: str = "",
        categorical: List[str] = None,
        label: str = None,
) -> CSV:
    """Loads a CSV file that contains categorical and predictive data columns.

    Args:
        path: The local file path or a web URL of the file.
        categorical: A list of column names that hold categorical data.
        label: The name of the categorical column that holds predictive label for each data sample.
    """
    dataset = pd_read_csv(...) # helper method to load a pandas dataset from various sources
    ...
    return CSV(...)
```
</details>


<details>
<summary>Example model loader</summary>

```python
from mammoth_commons.models import ONNX
from mammoth_commons.integration import loader


@loader(namespace="...", version="v001", python="3.13")
def model_onnx(
        path: str
) -> ONNX:
    """This is an ONNX loader.
    Args:
        path: The path from which to retrieve the loader's data.
    """
    return ONNX(path)

```
</details>

## Locally test a module

After decorating a module, you will want to test that it
runs correctly before uploading it for public consumption.
To write tests that
verify but then ignore your decorators to run on local data, 
create a context from which you can access the undecorated methods 
like so:

```Python
import mammoth_commons
from mammoth_commons import dataloader, modelloader, metric  # import your modules here

with mammoth_commons.testing.Env(dataloader, modelloader, metric) as env:
    data = env.dataloader("data_url", data_kwarg1=..., data_kwarg2=..., ...)
    model = env.dataloader("model_url", model_kwarg1=..., model_kwarg2=..., ...)
    sensitive = ["attr1", "attr2", ...]  # list of sensitive attributes
    result = env.metric(data, model, sensitive, metric_kwarg1=..., metric_kwarg2=..., ...)
    print(result.text)
```

If you are planning to create a pull request to mammoth-commons,
create a file `tests/test_...` containing the above code. This will be run
by the `integration_tests.py` script, which is part of the project's
GitHub actions. Everything new is expected to have high 
code coverage (more than 80% right now). Please run the script locally
to ensure that you did not break anything else.

:bulb: Do not forget to add all requirements to the `config/requirements[mai].txt` file.
Also install the libraries in that file for tests to run locally.

Pull requests will be reviewed manually, so if you plan to create a complex one
get in touch with us by opening an issue first. Finally, your module should 
be automatically added to the demonstrator do that you can see how it is going
to appear in the main toolkit.
The demonstrator does not require the steps covered below and you can run it
during development per `python -m mai_bias.app` for the desktop app 
or `python -m mai_bias.cli` for a command line interface.

## Write documentation

You need to populate your module's docstring with adequate information.
This information will automatically appear in the desktop application, 
in the server toolkit, and in the MAMMOth catalogue webpage 
[here](https://mammoth-eu.github.io/mammoth-commons/).
Adhere to the following checkpoints:

- Describe the modules to people that may be inexperience about your 
modality or fairness. The domain and data organization should be clearly explained, as well as
what the fairness analysis consists of.
Add links to examples webpages, libraries, or
notebooks (e.g., to instructions on how to create models that can be
loaded by respective loaders). 
- Aim for 250 words for metric modules, and at least 150 words for the rest.
If you plan to use more words, consider hiding parts of the description
with a `summary` tag.
- Use HTML tags in the description and bootstrap classes for styling (no custom CSS).
- Add warning for modules that are unstable or require performant hardware.
In this case, embed something like the following in your docstring (taken from
the XAI module):

```html
<span class="alert alert-warning alert-dismissible fade show" role="alert" style="display: inline-block; padding: 10px;">
  <i class="bi bi-exclamation-triangle-fill"></i> XAI analysis may be computationally intensive.
</span>
```

- Properly add arguments without annotating their types in the docstring. Each
argument's description can only span one line for now, but do make sure that
you describe all inputs other than `dataset,model,sensitive` which are common
across all metrics. Use underscores to separate multiple words
in argument and module names, as they can be converted to spaces
when shown to users.
Here is an example docstring with args:

```python
"""
This is my extensive module description.

Args:
    arg0: Detailed description for arg0.
    arg1: Detailed description for arg1.
"""
```


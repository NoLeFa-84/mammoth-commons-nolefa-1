import urllib.request
import urllib.parse
import os
from typing import Any

from mammoth_commons.datasets import Labels
from mammoth_commons.integration_callback import notify_progress, notify_end
import zipfile
import bz2
import pathlib
import shutil
import re


def get_import_list(code):
    code = pathlib.Path(code).read_text() if code.endswith(".py") else code
    found_imports = list()
    for line in code.splitlines():
        line = line.strip()
        if line.startswith("import "):
            imported_modules = line.split("import", 1)[1].strip().split(",")
            for module in imported_modules:
                module_name = module.split()[0].split(".")[0].strip()
                found_imports.append(module_name)
        elif line.startswith("from "):
            parts = line.split()
            if len(parts) > 1:
                module_name = parts[1].split(".")[0]
                found_imports.append(module_name)
    return found_imports


def safeexec(code: str, out: str = "commons", whitelist: list[str] = None):
    code = pathlib.Path(prepare(code)).read_text() if code.endswith(".py") else code
    whitelist = () if whitelist is None else set(whitelist)
    for module_name in get_import_list(code):
        assert (
            module_name in whitelist
        ), f"Disallowed import detected: '{module_name}'. Only these are allowed: {','.join(whitelist)}"
    exec_context = locals().copy()
    exec(code, exec_context)
    assert (
        out in exec_context
    ), f"The provided script or file did not contain an {out} variable"
    return exec_context[out]


def get_model_layer_list(model):
    try:
        model = model.model
        return [name for name, _ in model.named_modules() if name]
    except Exception as e:
        print(e)
        return []


def align_predictions(predictions: Any, labels: Labels) -> (Labels, Labels | None):
    if labels is None:
        assert isinstance(
            predictions, Labels
        ), "Internal error: align_predictions with no labels requires predictions of class Labels"
        return predictions, None
    assert isinstance(
        labels, Labels
    ), "Internal error: align_predictions requires labels of class Labels"
    if isinstance(predictions, dict):
        predictions = Labels(predictions)
    if isinstance(predictions, Labels):
        try:
            assert len(predictions) == len(labels)
            for key in predictions:
                assert key in labels
            for key in labels:
                assert key in predictions
        except AssertionError:
            raise Exception(
                "Different predictions to labels: "
                + ",".join(predictions.__iter__())
                + " vs "
                + ",".join(labels.__iter__())
            )
    elif hasattr(predictions, "to_numpy"):
        predictions = predictions.to_numpy()
    elif hasattr(predictions, "to_dict"):
        predictions = Labels(predictions.to_dict(orient="list"))

    if not isinstance(predictions, Labels):
        if "0" in labels and "1" in labels and len(labels) == 2:
            predictions = Labels({"0": 1 - predictions, "1": predictions})
        elif "no" in labels and "yes" in labels and len(labels) == 2:
            predictions = Labels({"no": 1 - predictions, "yes": predictions})
        else:
            raise Exception(
                "The selected model creates a vector of predictions but it is unknown how to match this to "
                f"multiple labels {','.join(labels.__iter__())}. Make the dataset have 0/1 or no/yes labels to "
                "automatically convert the prediction to two columns."
            )
    predictions = Labels({f"class {k}": v for k, v in predictions.items()})
    labels = Labels({f"class {k}": v for k, v in labels.items()})
    return predictions, labels


def fb_categories(it):
    import fairbench as fb

    @fb.v1.Transform
    def categories(iterable):
        is_numeric = True
        values = list()
        for value in iterable:
            try:
                values.append(float(value))
            except Exception:
                is_numeric = False
                break
        # if len(set(v for v in values)) == 2:
        #    is_numeric = False
        if is_numeric:
            values = fb.v1.tobackend(values)
            mx = values.max()
            mn = values.min()
            if mx == mn:
                mx += 1
            values = fb.v1.tobackend((values - mn) / (mx - mn))
            return {
                f"fuzzy min ({mn:.3f})": 1 - values,
                f"fuzzy max ({mx:.3f})": values,
            }
        return fb.categories @ iterable

    return categories @ it


def _download(url, path):
    if os.path.exists(path):
        return path
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    if os.path.isfile(url):
        shutil.copyfile(url, path)
        return path
    with urllib.request.urlopen(url) as response:
        total_size = (
            response.getheader("Content-Length")
            if hasattr(response, "getheader")
            else None
        )
        total_size = int(total_size) if total_size else None
        with open(path, "wb") as out_file:
            chunk_size = 4096 * 8  # 32kbps rate updates each second
            downloaded = 0
            chunk = True
            while chunk:
                chunk = response.read(chunk_size)
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    notify_progress(downloaded / total_size, f"Downloading {url}")
    notify_end()
    return path


def _extract_nested_zip(file, folder):
    os.makedirs(folder, exist_ok=True)
    with zipfile.ZipFile(file, "r") as zfile:
        zfile.extractall(path=folder)
    os.remove(file)
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.endswith(".zip"):
                _extract_nested_zip(
                    os.path.join(root, filename), os.path.join(root, filename[:-4])
                )


def _autoextract(path):
    if path.endswith(".bz2"):
        extract_to = path[:-4]
        if not os.path.exists(extract_to):
            with bz2.BZ2File(path, "rb") as bz2_file:
                with open(extract_to, "wb") as out_file:
                    out_file.write(bz2_file.read())
        return _autoextract(extract_to)
    return path


def _toextract(path):
    if path.endswith(".bz2"):
        return True
    return False


def prepare(url, cache=".cache"):
    url = url.replace("\\", "/")
    if (
        ".zip/" in url
    ):  # we will never unzip full zips (preparing is for one file each time)
        url, path = url.split(".zip/", 1)
        extract_to = os.path.join(cache, os.path.basename(url))
        path = os.path.join(cache, os.path.basename(url), path)
        url += ".zip"
        temp = os.path.join(cache, os.path.basename(url))
        if not os.path.exists(path):
            _download(url, temp)
            _extract_nested_zip(temp, extract_to)
        url = path

    path = (
        url
        if os.path.exists(url) and not _toextract(url)
        else _download(url, os.path.join(cache, os.path.basename(url)))
    )
    path = _autoextract(path)

    return path


def to_file_url(path: str) -> str:
    path = os.path.abspath(path)
    encoded = urllib.parse.quote(path)
    return f"file:///{encoded}"


def prepare_html(html: str) -> str:
    pattern = r'(src|href)=(["\'])([^"\']+)\2'

    def repl(match):
        attr = match.group(1)  # src or href
        quote = match.group(2)  # ' or "
        url = match.group(3)  # the URL value
        if (url.startswith("http://") or url.startswith("https://")) and (
            ".png" in url
            or ".js" in url
            or ".css" in url
            or ".svg" in url
            or ".jpg" in url
            or "githubusercontent" in url
        ):
            try:
                cached_path = prepare(url)
                file_url = to_file_url(cached_path)
                return f"{attr}={quote}{file_url}{quote}"
            except Exception as e:
                print("prepare_html WARNING:", e)
                return match.group(0)  # keep original
        return match.group(0)

    return re.sub(pattern, repl, html)


def pd_read_csv(url, **kwargs):
    import pandas as pd
    import csv
    import string

    path = prepare(url)
    if "delimiter" in kwargs:
        return pd.read_csv(path, **kwargs)
    try:
        with open(path, "r") as file:
            sample = file.read(1024)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            delimiter = str(delimiter)
            if delimiter in string.ascii_letters:
                common_delims = [",", ";", "|", "\t"]
                counts = {d: sample.count(d) for d in common_delims}
                delimiter = max(counts, key=counts.get) if any(counts.values()) else ","
    except Exception:
        delimiter = None
    return pd.read_csv(path, delimiter=delimiter, **kwargs)


def format_name(name):
    ret = " ".join(name.split("_")).replace("data ", "").replace("model ", "")
    ret = ret[0].upper() + ret[1:]
    return ret


def format_description(description, desktopmode=False):
    def transform_doc(doc):
        pre_blocks = re.findall(r"<pre>.*?</pre>", doc, flags=re.DOTALL)
        placeholders = [f"MAMMOTHCOMMONSPREBLOCK{i}" for i in range(len(pre_blocks))]
        for i, block in enumerate(pre_blocks):
            doc = doc.replace(block, placeholders[i])
        doc = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", doc)
        doc = re.sub(r"\*(.+?)\*", r"<i>\1</i>", doc)
        doc = re.sub(r"`(.+?)`", r"<code>\1</code>", doc)

        def list_replacer(match):
            items = match.group(0).strip().splitlines()
            items = [f"<li>{item[2:].strip()}</li>" for item in items]
            return "<ul>" + "".join(items) + "</ul>"

        doc = re.sub(
            r"(?:^|\n)(- .+(?:\n- .+)*)", list_replacer, doc, flags=re.MULTILINE
        )
        # doc = (
        #     doc.replace("\n\n", "<br><br>")
        #     .replace("\n", " ")
        #     .replace("_", " ")
        #     .replace("  ", " ")
        # )
        for i, block in enumerate(pre_blocks):
            doc = doc.replace(placeholders[i], block)
        return doc

    doc = ""
    args_desc = dict()
    args_options = dict()
    started_args = False
    separator_title = " "
    sep_title = separator_title
    started_options = False
    for line in description.split("\n"):
        line = line.strip()
        if line.startswith("Options:"):
            started_options = True
        elif line.startswith("Args:"):
            started_args = True
        elif line.endswith(" args:"):
            separator_title = line[:-5].strip()
            sep_title = separator_title
            if separator_title:
                separator_title = "<br><h3>" + separator_title + "</h3>"
        elif started_options and ":" in line:
            splt = line.split(":", maxsplit=2)
            args_options[splt[0]] = [option.strip() for option in splt[1].split(",")]
        elif started_args and ":" in line:
            splt = line.split(":", maxsplit=2)
            name = format_name(splt[0]).replace(sep_title + " ", "")
            name = name[0].upper() + name[1:]
            # args_desc[splt[0]] = f"{separator_title}<i>{name} - </i> {splt[1]}"
            args_desc[splt[0]] = (
                f"""<h1>{separator_title} {name}</h1> {splt[1]}"""
                if desktopmode
                else f"""<button
                              type="button"
                              class="btn btn-light"
                              data-bs-toggle="tooltip"
                              data-bs-placement="top"
                              title="{splt[1]}"
                              data-description="{splt[1]}"
                              data-name="{name}"
                              onclick="showDescriptionModal(this)">
                              <i class="bi bi-info-circle"></i> {name}
                            </button>"""
            )

            separator_title = ""
        else:
            doc += line + "\n"
    return transform_doc(doc), args_desc, args_options

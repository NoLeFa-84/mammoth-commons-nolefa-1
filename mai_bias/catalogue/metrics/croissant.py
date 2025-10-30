from mammoth_commons.datasets import Dataset
from mammoth_commons.models import EmptyModel
from mammoth_commons.exports import HTML
from typing import List
from mammoth_commons.integration import metric
import json


@metric(
    namespace="mammotheu",
    version="v0049",
    python="3.13",
    packages=(
        "fairbench",
        "pandas",
        "onnxruntime",
        "ucimlrepo",
        "pygrank",
    ),
)
def croissant(
    dataset: Dataset,
    model: EmptyModel,
    sensitive: List[str],
    language: str = "en",
    license: str = "",
    name: str = "",
    description: str = "",
    citation: str = "",
    qualitative_creators: List[str] = "",
    distribution: List[str] = "",
) -> HTML:
    """Generate some json dataset metadata that boostraps conversion of your datasets into the
    <a href="https://github.com/mlcommons/croissant">Croissant</a> format. That format is used to
    standardized how datasets may be indexed and loaded. If your dataset is stored locally, such as
    in minio instances, you can consider either sharing the metadata to explain to others what you
    are working with, or using publicly hosted data by providing https links for files.
    Metadata are displayed as HTML to help you get an overview and are presented as a copy-able block of json.
    """
    if isinstance(qualitative_creators, str):
        qualitative_creators = qualitative_creators.split(",")
    if isinstance(distribution, str):
        distribution = distribution.split(",")
    dataset = dataset.to_csv(sensitive)
    context = {
        "@language": "en",
        "@vocab": "https://schema.org/",
        "arrayShape": "cr:arrayShape",
        "citeAs": "cr:citeAs",
        "column": "cr:column",
        "conformsTo": "dct:conformsTo",
        "cr": "http://mlcommons.org/croissant/",
        "rai": "http://mlcommons.org/croissant/RAI/",
        "data": {"@id": "cr:data", "@type": "@json"},
        "dataBiases": "cr:dataBiases",
        "dataCollection": "cr:dataCollection",
        "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
        "dct": "http://purl.org/dc/terms/",
        "examples": {"@id": "cr:examples", "@type": "@json"},
        "extract": "cr:extract",
        "field": "cr:field",
        "fileProperty": "cr:fileProperty",
        "fileObject": "cr:fileObject",
        "fileSet": "cr:fileSet",
        "format": "cr:format",
        "includes": "cr:includes",
        "isArray": "cr:isArray",
        "isLiveDataset": "cr:isLiveDataset",
        "jsonPath": "cr:jsonPath",
        "key": "cr:key",
        "md5": "cr:md5",
        "parentField": "cr:parentField",
        "path": "cr:path",
        "personalSensitiveInformation": "cr:personalSensitiveInformation",
        "recordSet": "cr:recordSet",
        "references": "cr:references",
        "regex": "cr:regex",
        "repeated": "cr:repeated",
        "replace": "cr:replace",
        "sc": "https://schema.org/",
        "separator": "cr:separator",
        "source": "cr:source",
        "subField": "cr:subField",
        "transform": "cr:transform",
    }

    metadata = {
        "@context": context,
        "@type": "sc:Dataset",
        "distribution": distribution,
        "@language": language,
        "@vocab": "https://schema.org/",
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "name": name,
        "description": description,
        "license": license,
        "citeAs": citation,
        # "url":url
        "creator": [{"name": creator} for creator in qualitative_creators],
        "data": [],
        "columns": [
            {
                "name": col,
                "description": f"Column '{col}' in the dataset. "
                + f"Contains {len(set(dataset.df[col]))} distinct values out of {len(dataset.df[col])} entries. "
                + (
                    "It serves as metadata information for each entry. "
                    if col in dataset.cat or col in dataset.num
                    else "It is used for data loading and does not serve as metadata. "
                ),
                "datatype": (
                    "string"
                    if col in dataset.cat
                    else "float" if col in dataset.num else "string"
                ),
                "isSensitive": col in sensitive,
            }
            for col in dataset.df.columns
        ],
    }

    json_string = (
        json.dumps(metadata, indent=2)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    html = f"""
    <div class="container mt-4">
        <h1 class="text-success">Croissant metadata</h1>
        <b>Title:</b> {name}<br/>
        <b>Description:</b> {description}<br/>
        <b>Citation:</b> {citation}<br/>
        <b>License:</b> {license}<br/>
        <b>Creators:</b> {', '.join(qualitative_creators)}<br/>

        <button class="btn btn-primary mb-2" onclick="copyPre()">Copy json metadata</button>
        <pre id="json-block" style="background:#f8f9fa; border:1px solid #ccc; padding:1em; border-radius:0.5em; overflow:auto;">{json_string}</pre>
    </div>

    <script>function copyPre() {{
        const pre = document.getElementById("json-block");
        const text = pre.innerText;
        const textarea = document.createElement("textarea");
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        try {{
            document.execCommand("copy");
            //alert("Copied!");
        }} catch (err) {{
            //alert("Failed to copy");
        }}
        document.body.removeChild(textarea);
    }}</script>

    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    """

    return HTML(html)

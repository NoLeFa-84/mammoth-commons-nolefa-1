import shutil
import zipfile
import site
from urllib.request import urlretrieve
import os
from mammoth_commons.datasets import Text
from mammoth_commons.integration import metric
from mammoth_commons.models import EmptyModel
from mammoth_commons.exports import HTML
from mammoth_commons.externals import notify_progress, notify_end


def manual_install_wheel(wheel_url_or_path):
    print("Manually installing a broken wheel for the DBias library")
    if wheel_url_or_path.startswith("http://") or wheel_url_or_path.startswith(
        "https://"
    ):

        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            progress = min(downloaded / total_size, 1.0) if total_size > 0 else 0
            message = f"Downloading dbias transformers to cache... {int(progress*100)}%"
            notify_progress(progress, message)

        local_whl = os.path.join(".cache", os.path.basename(wheel_url_or_path))
        if not os.path.exists(local_whl):
            print(f"Downloading wheel from {wheel_url_or_path}...")
            urlretrieve(wheel_url_or_path, local_whl, reporthook=reporthook)
            notify_end()
    else:
        local_whl = wheel_url_or_path
        if not os.path.exists(local_whl):
            raise FileNotFoundError(f"No such file: {local_whl}")
    print(f"Unpacking {local_whl}...")
    unpack_dir = local_whl.replace(".whl", "_unpacked")
    with zipfile.ZipFile(local_whl, "r") as zf:
        zf.extractall(unpack_dir)
    site_packages_dirs = site.getsitepackages()
    if not site_packages_dirs:
        site_packages_dirs = [site.getusersitepackages()]
    site_packages = site_packages_dirs[0]
    print(f"Copying to site-packages at: {site_packages}")
    for item in os.listdir(unpack_dir):
        src_path = os.path.join(unpack_dir, item)
        dst_path = os.path.join(site_packages, item)
        if os.path.exists(dst_path):
            print(f"Overwriting existing: {dst_path}")
            if os.path.isdir(dst_path):
                shutil.rmtree(dst_path)
            else:
                os.remove(dst_path)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)
    print(f"Installed {local_whl} manually into site-packages.")


@metric(
    namespace="mammotheu",
    version="v054",
    python="3.11",
    packages=(
        "dbias --no-deps --upgrade",
        "protobuf==4.25.8",
        "tensorflow",
        "transformers",
        "tf-keras",
        "catalogue==2.0.6",
        "spacy==3.2.0",
        "plotly",
        "torch",
        "pandas",
        "spacy-transformers",
    ),
)
def text_debias(dataset: Text, model: EmptyModel, sensitive: list[str]) -> HTML:
    """
    <p>
    This module uses <a href="https://github.com/dreji18/Fairness-in-AI">DBias</a> library to perform
    unsupervised auditing of text biases in text resembling article titles. If library identifies biases, it is used to
    mitigate them with a more neutral phrasing. Results show a judgement and prediction confidence for the original
    and adjusted text, as well as potentially offending keywords.
    </p>

    <p>
    The DBias library employs three transformer models,
    one pretrained for the English language and two trained
    on the <a href="https://github.com/Media-Bias-Group/Neural-Media-Bias-Detection-Using-Distant-Supervision-With-BABE">MBIC dataset</a>.
    </p>
    """

    # RANT AHEAD WITH SOME USEFUL INFO ABOUT FUTURE CHOICES - BUT NOT EXPECTING ANYONE TO BE ABLE TO MAINTAIN THIS
    #
    # The whole implementation is a complete and utter hack because the original release of DBias
    # has been rendered basically obsolete by the uselessly fast-evolving LLM technology landscape.
    #
    # I don't care if you present the latest SOTA framework if this is going to break in a couple of months
    # forever. To be clear: I am mainly bashing transformers and the huggingface ecosystem's hype
    # and not dbias (who is at most guilty of a couple questionable engineering choices - like we all are).
    #
    # If you follow instructions from the dbias repo prepare for a world of pain. In fact, I have suggested
    # the solution in this file to that repository as a more helpful (!) alternative to installing stuff.
    # And I still needed to create a fork of the repo, remove dependency freezes,
    # and orchestrate a git installation.
    #
    # So the hacks:
    #
    # - MAI-BIAS installs the default version of numpy (which is later than 2.0).
    #   I do not know why people keep freezing versions of numpy without good reason. *Everyone* uses numpy
    #   in AI and this is the easiest way to create a mess.
    # - We also install the latest version of spacy, because older ones often do not compile (I am failing
    #   to compile the required one in kubuntu with Python 3.13 and refuse to investigate a library that should be
    #   working out of the box but failing because everyone is too hyped by new technologies to create maintainable
    #   software)
    # - Thankfully, more recent versions (currently 3.2.0 which I will not freeze because it is more likely that it
    #   will not be maintained than a new release is to break backwards compatibility) still properly load the packaged
    #   models, despite throwing a ton of warnings.
    # - Lastly we manually unpack `en_pipeline-any-py3-none-any.whl` into site-packages WITHOUT INSTALLING ITS
    #   DECLARED DEPENDENCIES. Python's build system works well when the latest version=best import. The ease of docker
    #   has promoted frozen package versions and in response more tooling that makes absolutely zero sense when
    #   you try to do something constructive by actually combining solutions instead of deploying apps or microservices.
    # - Also disabled GPU to prevent JIT-ing because this is the one thing that breaks after all the other sorcery.
    #   Surprised we reached so far, honestly.
    # - There's a chance pip's eager upgrade will work. But this also works and does what I want it to for sure.
    # - Yes, I could also add a large try-catch to guarantee restoration of `os.environ["CUDA_VISIBLE_DEVICES"]`
    #   but this is hacky enough and runs fast enough for me to prefer the occasional user issue rather than
    #   looking at the monstrosity of having another nesting block. I HOPE I'm not breaking other cuda devices
    #   in other modules in the demonstrator due to module caching, but if I do let's make future versions
    #   unload the cached GPU framework's import that dbias uses.
    #
    # Good luck to all out there. - Manios
    assert len(sensitive) == 0, "No sensitive attribute is expected for text_dbias"

    text = dataset.text
    original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        from Dbias.text_debiasing import run
        from Dbias.bias_classification import classify
        from Dbias.bias_recognition import recognizer
        import en_pipeline
    except OSError:
        manual_install_wheel(
            "https://huggingface.co/d4data/en_pipeline/resolve/main/en_pipeline-any-py3-none-any.whl"
        )
    except:
        manual_install_wheel(
            "https://huggingface.co/d4data/en_pipeline/resolve/main/en_pipeline-any-py3-none-any.whl"
        )
    from Dbias.text_debiasing import run
    from Dbias.bias_classification import classify
    from Dbias.bias_recognition import recognizer

    def custom_debiasing(x):
        suggestions = run(x)
        if suggestions is None:
            return []
        return [sent["Sentence"] for sent in suggestions[0:3]]

    def custom_recognizer(x):
        biased_words = recognizer(x)
        return [word["entity"] for word in biased_words]

    classification = classify(text)
    biased_words_list = custom_recognizer(text)
    suggestions = custom_debiasing(text)

    faq_html = f"""
    <style>
    .faq-container {{
      max-width: 600px;
      margin: 20px auto;
      font-family: Arial, sans-serif;
    }}
    .faq-box {{
      border: 1px solid #ccc;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
      box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
      background: #fff;
    }}
    .faq-box h3 {{
      margin-top: 0;
      font-size: 1.2em;
      color: #333;
    }}
    .faq-box p {{
      margin: 0;
      color: #555;
    }}
    </style>

    <div class="faq-container">
        <div class="faq-box">
            <h3>❓ What is this?</h3>
            <p>This module uses the <a href="https://github.com/dreji18/Fairness-in-AI" target="_blank">DBias</a> 
            library to audit text for biases and propose debiased alternatives. It applies transformer-based 
            models trained on the <a href="https://github.com/Media-Bias-Group/Neural-Media-Bias-Detection-Using-Distant-Supervision-With-BABE" target="_blank">MBIC dataset</a> 
            and pretrained language models for English. The output highlights potential biased phrases and 
            suggests neutral phrasings to mitigate them.</p>
        </div>

        <div class="faq-box">
            <h3>❗ Summary</h3>
            <p>The analysis classifies the input as <strong>{classification[0]['label']}</strong> 
            with confidence {classification[0]['score']:.3f}. 
            {f"Biased phrases detected: {', '.join(biased_words_list)}." if biased_words_list else "No biased phrases detected."}</p>
            <br/>
            <p>When biases are found, DBias suggests alternative phrasings that reduce potentially harmful wording. 
            This helps improve fairness and neutrality in textual outputs, making them less prone to reinforcing stereotypes.</p>
        </div>
    </div>
    """

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Text analysis</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="bg-light">
        <div class="container py-5">
            <h1 class="mb-4">{'Biased text fixes' if 'Biased'==classification[0]['label'] else 'Neutral text'}</h1>
            <hr/>
            {faq_html}
            <hr/>
            <div class="card mb-4">
                <div class="card-header bg-primary text-white">Original text</div>
                <div class="card-body">
                    <p>{text}</p>
                </div>
            </div>

            <div class="card mb-4">
                <div class="card-header bg-warning text-dark">Verdict</div>
                <div class="card-body">
                    <p class="mb-0"><strong>{classification[0]['label']}</strong> with self-reported confidence {classification[0]['score']:.3f}</p>

                {f'''
                    <p>Biased phrases: {', '.join(biased_words_list)}</p>
                ''' if biased_words_list else ''}
            
                </div>
            </div>

            {f'''
            <div class="card mb-4">
                <div class="card-header bg-success text-white">Suggested alternatives</div>
                <div class="card-body">
                    {''.join(f'<p>{sugg}</p>' for sugg in suggestions)}
                </div>
            </div>
            ''' if suggestions else ''}
        </div>
    </body>
    </html>
    """

    # restore the original CUDA devices
    if original_cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    return HTML(html)

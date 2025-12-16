from mammoth_commons.datasets import Text
from mammoth_commons.integration import loader


@loader(
    namespace="mammotheu", version="v054", python="3.13", packages=("bs4", "requests")
)
def data_free_text(text: str = "") -> Text:
    """<img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/text.png?raw=true"
    alt="text" style="float: left; margin-right: 15px; height: 36px;"/>
    <h3>input text or document URL</h3>

    Sets a free text that can be used by text-based AI to perform various kinds of analysis,
    such as detecting biases and sentiment. Some modules may also use this text as a prompt
    to feed into large language models (LLMs). You may optionally provide a website's URL (starting
    with <i>http:</i> or <i>https:</i>) to retrieve its textual contents.

    Args:
        text: The text to be analyzed or its URL.
    """
    import requests

    if text.startswith("http://") or text.startswith("https://"):
        response = requests.get(text, timeout=10)
        response.raise_for_status()
        html_content = response.text
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, "html.parser")
        text = "\n\n".join(
            p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)
        )

    return Text(text)

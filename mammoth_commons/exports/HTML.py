from kfp import dsl
import base64
import re
from typing import Literal
from mammoth_commons.reminders import on_results


def _encode_image_to_base64(filepath):
    with open(filepath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def _replace_emojis(text):
    emoji_patterns = {
        ":x:": "❌",
        ":rocket:": "🚀",
        ":checkmark:": "🗸",
        ":smile:": "😄",
        ":thumbsup:": "👍",
        ":heart:": "❤️",
        ":star:": "⭐",
        ":fire:": "🔥",
        ":tada:": "🎉",
        ":clap:": "👏",
        ":heavy_check_mark:": "✔️",
    }
    pattern = re.compile("|".join(re.escape(key) for key in emoji_patterns.keys()))
    return pattern.sub(lambda m: emoji_patterns[m.group(0)], text)


def _highlight_code(html_content):
    try:
        from pygments import highlight
        from pygments.lexers import get_lexer_by_name
        from pygments.formatters import HtmlFormatter

        code_block_pattern = re.compile(
            r'<pre><code class="language-(\w+)">(.*?)</code></pre>', re.DOTALL
        )

        def replace_code_block(match):
            lang = match.group(1)
            code = (
                match.group(2)
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&amp;", "&")
            )
            lexer = get_lexer_by_name(lang)
            formatter = HtmlFormatter()
            return highlight(code, lexer, formatter)

        return code_block_pattern.sub(replace_code_block, html_content)
    except:
        print("Consider install pygments (pip install pygments) to highlight HTML code")
        return html_content


def simplified_formatter(
    outcome: Literal["biased", "fair", "report"],
    title: str,
    about: str,
    methodology: str,
    pipeline: str,
    experts: str,
    technology: str = "",
    warning: str = on_results,
):
    return f"""
        <style>
            .pill-buttons {{display: flex; gap: 12px; margin: 20px 0;}}
            .banner {{
                width: 100%;
                padding: 180px 24px;
                font-size: 64px;
                font-weight: 700;
                text-align: center;
                color: white;
                border-radius: 12px;
                margin-bottom: 25px;
            }}
            .banner.fair {{ background: #2e8b57; }}
            .banner.biased {{ background: #c0392b; }}
            .banner.report {{ background: #7f8c8d; }}
            .pill-btn {{
                width:100%; text-align:center; padding: 10px 18px;
                background: #f5f5f5; border-radius: 10px; border: 1px solid #cccccc;
                cursor: pointer; font-size: 18px; transition: background 0.2s;
            }}
            .pill-btn:hover {{ background: #e0e0e0; }}
            .pill-btn.active {{ background: #d0d0d0; border-color: #999999;}}
            .section-panel {{ display: none; padding: 0px; background: white; }}
            .section-panel.active {{ display: block; }}
            .tablinks {{
                background-color: #ddd;
                padding: 10px;
                cursor: pointer;
                border: none;
                border-radius: 5px;
                margin: 5px;
            }}
            .tablinks.active {{ background-color: #aaa; }}
            .tabcontent {{ display: none; padding: 10px; border: 1px solid #ccc; }}
            .tabcontent.active {{ display: block; }}
        </style>

        <script>
            document.addEventListener("DOMContentLoaded", function() {{
                const buttons = document.querySelectorAll(".pill-btn");
                const sections = document.querySelectorAll(".section-panel");
                buttons.forEach(btn => {{
                    btn.addEventListener("click", () => {{
                        let target = btn.getAttribute("data-target");
                        buttons.forEach(b => b.classList.remove("active"));
                        sections.forEach(s => s.classList.remove("active"));
                        btn.classList.add("active");
                        document.getElementById(target).classList.add("active");
                    }});
                }});
                document.querySelector(".pill-btn").classList.add("active");
                document.querySelector(".section-panel").classList.add("active");
                const tabContainer = document.getElementById("expert-tab-header");
                if (tabContainer) {{
                    tabContainer.addEventListener("click", function(event) {{
                        if (event.target.classList.contains("tablinks")) {{
                            let tabName = event.target.getAttribute("data-tab");
                            document.querySelectorAll(".tablinks").forEach(tab => tab.classList.remove("active"));
                            document.querySelectorAll(".tabcontent").forEach(content => content.classList.remove("active"));
                            event.target.classList.add("active");
                            document.getElementById(tabName).classList.add("active");
                        }}
                    }});
                    let first = tabContainer.querySelector(".tablinks");
                    if (first) {{
                        first.classList.add("active");
                        document.getElementById(first.getAttribute("data-tab")).classList.add("active");
                    }}
                }}
            }});
        </script>
        <h1 class="banner {outcome}">{title}</h1>
        {technology}
        <div class="pill-buttons">
            <div class="pill-btn" data-target="whatis">What is this?
            <br><img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/question.png?raw=true" height="128px"/>
            </div>
            <div class="pill-btn" data-target="warning">Responsible use
            <br><img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/warning.png?raw=true" height="128px"/>
            </div>
            <div class="pill-btn" data-target="method">Analysis methodology
            <br><img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/methodology.png?raw=true" height="128px"/>
            </div>
            <div class="pill-btn" data-target="pipeline">Data pipeline
            <br><img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/data.png?raw=true" height="128px"/>
            </div>
            <div class="pill-btn" data-target="details">For experts
            <br><img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/chart.png?raw=true" height="128px"/>
            </div>
        </div>
        <hr>
        <div id="whatis" class="section-panel">{about}</div>
        <div id="warning" class="section-panel">{warning}</div>
        <div id="method" class="section-panel">{methodology}</div>
        <div id="pipeline" class="section-panel">{pipeline}</div>
        <div id="details" class="section-panel">{experts}</div>
        """


def get_description_header(text: str):
    pos = text.find("</h1>")
    if pos == -1:
        pos = text.find("</h2>")
    if pos == -1:
        pos = text.find("</h3>")
    if pos == -1:
        return ""
    start = text.rfind("<img", 0, pos - 1)
    if start == -1:
        start = text.rfind("<", 0, pos - 1)
    return text[start : (pos + 5)]


class HTML:
    integration = "dsl.HTML"

    def __init__(
        self,
        body: str,
        header: str = "",
        script: str = "",
        images: dict[str, str] = None,
    ):
        # images are encoded as bytes if needed
        self.body = body
        self.header = header
        self.script = script
        self.images = dict() if images is None else images

    def show(self, temppath="temp.html"):
        import webbrowser

        with open(temppath, "w", encoding="utf-8") as file:
            file.write(self.all())
        webbrowser.open_new(file.name)

    def all(self):
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>MAI-BIAS run</title>
            {self.header}
        </head>
        <body>
            {self.text()}
        </body>
        </html>
        """

    def text(self):
        body = self.body
        for image, path in self.images.items():
            data = _encode_image_to_base64(path)
            img = f'<img src="base64,{data}" alt="{image}" />'
            body.replace(image, img)
        return _highlight_code(_replace_emojis(body))

    def export(self, output: dsl.Output[integration]):
        with open(output.path, "w") as f:
            output.name = "result.html"
            f.write(self.all())

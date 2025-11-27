from kfp import dsl


def apply_bootstrap(html_body):
    replacements = {
        # "<h1>": '<h1 class="display-4">',
        # "<h2>": '<h2 class="display-5">',
        # "<h3>": '<h3 class="display-6">',
        "<ul>": '<ul class="list-group">',
        "<li>": '<li class="list-group-item">',
        "<code>": '<code class="language-python">',
        # "<code>": '<code class="text-dark px-2 py-1">',
    }
    for key, value in replacements.items():
        html_body = html_body.replace(key, value)
    html_body = f'<div class="container" style="max-width: 800px; margin: auto;">{html_body}</div>'
    return html_body


class Markdown:
    integration = "dsl.Markdown"

    def __init__(self, text):
        self._text = text

    def text(self):
        import markdown2
        from mammoth_commons.exports.HTML import HTML

        return HTML(
            apply_bootstrap(
                markdown2.markdown(
                    self._text, extras=["tables", "fenced-code-blocks", "code-friendly"]
                )
            )
        ).text()

    # def export(self, output: dsl.Output[integration]):
    #     with open(output.path, "w") as f:
    #         output.name = "result.md"
    #         f.write(self._text)

import pathlib

import mistune


DOC_PATH = pathlib.Path(__file__).resolve().parent

SPHINX_TEMPLATE = DOC_PATH / "templates" / "sphinx.html"


class Renderer(mistune.Renderer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._headers = []

    def header(self, text, level, raw=None):
        self._headers.append((level, text))
        return super().header(text, level, raw=raw)


class MarkdownParser(mistune.Markdown):

    def __init__(self):
        super().__init__(renderer=Renderer())

    def parse(self, text):
        html_content = super().parse(text)
        print(self.renderer._headers)
        return html_content

def markdown_to_html(filename: str):
    with open(filename) as fp:
        markdown = fp.read()

    parser = MarkdownParser()

    html = parser.parse(markdown)

    # print(html)




if __name__ == "__main__":

    markdown_to_html(DOC_PATH / "_doc_template.md")

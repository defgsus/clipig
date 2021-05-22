import pathlib
import warnings

import mistune

from src.doc import render_markdown_documentation
from src.expression import EXPRESSION_ARGS
from src.transforms import transformations
from src.constraints import constraints


DOC_PATH = pathlib.Path(__file__).resolve().parent

SPHINX_TEMPLATE = (DOC_PATH / "templates" / "sphinx.html").read_text()


class Renderer(mistune.Renderer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = None
        self.headers = []

    def header(self, text, level, raw=None):
        if level == 1 and not self.title:
            self.title = text
        self.headers.append((level, text))
        return super().header(text, level, raw=raw)


class MarkdownParser(mistune.Markdown):

    def __init__(self):
        super().__init__(renderer=Renderer())

    def parse(self, text):
        html_content = super().parse(text)
        # print(self.renderer.headers)
        final = SPHINX_TEMPLATE
        final = final.replace("{{title}}", self.renderer.title or "")
        final = final.replace("{{content}}", html_content)
        final = final.replace("{{side-nav}}", self.render_side_navigation())
        return final

    def render_side_navigation(self):
        links = []
        for level, name in self.renderer.headers:
            if level == 1:
                links.append(
                    f"""<p class="caption"><span class="caption-text">{name}</span></p>"""
                )
            else:
                links.append(
                    f"""<li class="toctree-l{level}"><a class="reference internal" href="quickref.html">{name}</a></li>"""
                )
        return "\n".join(links)


def markdown_to_html(template_filename: str):
    markdown = (DOC_PATH / "templates" / template_filename).read_text()

    markdown = render_markdown_documentation(markdown)

    parser = MarkdownParser()

    html = parser.parse(markdown)

    (DOC_PATH / template_filename.replace(".md", ".html")).write_text(html)


if __name__ == "__main__":

    markdown_to_html("reference.md")

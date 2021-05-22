import pathlib
import warnings
from typing import List, Tuple

import mistune

from src.doc import render_markdown_documentation


DOC_PATH = pathlib.Path(__file__).resolve().parent

SPHINX_TEMPLATE = (DOC_PATH / "templates" / "sphinx.html").read_text()


class Renderer(mistune.Renderer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = None
        self.headers = []
        self.links = []

    def header(self, text, level, raw=None):
        if level == 1 and not self.title:
            self.title = text
        self.headers.append((level, strip_html(text)))
        return '<h%d id="%s">%s</h%d>\n' % (
            level, to_anchor(strip_html(text)), text, level
        )

    def link(self, link, title, text):
        self.links.append(link)
        link = link.replace(".md", ".html")
        return super().link(link, title, text)

    def block_code(self, code, lang=None):
        code = mistune.escape(code, quote=True, smart_amp=False)
        return f'<pre class="code {lang}">{code}</pre>'


class MarkdownParser(mistune.Markdown):

    def __init__(self):
        super().__init__(renderer=Renderer())

    def parse(self, text) -> dict:
        self.renderer.headers = []
        self.renderer.links = []

        html_content = super().parse(text)

        final = SPHINX_TEMPLATE
        final = final.replace("{{title}}", self.renderer.title or "")
        final = final.replace("{{content}}", html_content)
        return {
            "markdown": text,
            "template": final,
            "title": self.renderer.title,
            "headers": self.renderer.headers,
            "links": self.renderer.links,
        }


def strip_html(h: str) -> str:
    ret = ""
    tag = False
    for c in h:
        if c == "<":
            tag = True
        elif c == ">":
            tag = False
        else:
            if not tag:
                ret += c
    return ret


def render_navigation(navigation: List[Tuple[int, str, str]]) -> str:
    links = []
    li_open = False
    for level, name, url in navigation:
        if level == 0:
            if li_open:
                links.append("</ul>")
            links.append(
                f"""<p class="caption"><a class="reference internal" href="{url}"><span class="caption-text">{name}</span></a></p>"""
            )
        else:
            if not li_open:
                links.append("<ul>")
                li_open = True
            links.append(
                f"""<li class="toctree-l{level}"><a class="reference internal" href="{url}">{name}</a></li>"""
            )
    if li_open:
        links.append("</ul>")
    return "\n".join(links)


def to_anchor(a: str) -> str:
    a = a.replace(" ", "-")
    a = "".join(
        c for c in a
        if c.isalnum() or c in ("-", "_")
    )
    return a


def validate_links(templates: dict):
    anchors = {
        name: [to_anchor(h[1]) for h in t["headers"]]
        for name, t in templates.items()
    }

    has_error = False
    for name, t in templates.items():
        for link in t["links"]:
            if link.startswith("http"):
                continue

            if link.startswith("#"):
                if link[1:] not in anchors[name]:
                    print(f"Invalid anchor '{link}' in '{name}.md'")
                    has_error = True
            else:
                link_s = link.split("#")
                if not link_s[0].endswith(".md") or link_s[0][:-3] not in anchors:
                    print(f"Invalid linked file '{link}' in '{name}.md'")
                    has_error = True

                if len(link_s) > 1:
                    if link_s[1] not in anchors[link_s[0][:-3]]:
                        print(f"Invalid anchor '{link}' in '{name}.md'")
                        has_error = True

    if has_error:
        exit(-1)


def render_templates(templates: List[str]):
    templates = {
        name: (DOC_PATH / "templates" / f"{name}.md").read_text()
        for name in templates
    }
    templates = {
        name: MarkdownParser().parse(render_markdown_documentation(template))
        for name, template in templates.items()
    }

    validate_links(templates)

    for name, template in templates.items():

        markdown_file = f"{name}.md"
        if name == "index":
            markdown_file = "README.md"

        print("writing", DOC_PATH / markdown_file)
        (DOC_PATH / markdown_file).write_text(template["markdown"])

        navigation = []
        for n, t in templates.items():
            navigation.append((0, t["title"], f"{n}.html"))
            if n == name:
                for level, header in t["headers"]:
                    if header != template["title"]:
                        navigation.append((level, header, f"{n}.html#{to_anchor(header)}"))

        # print(navigation)

        markup = template["template"].replace(
            "{{side-nav}}", render_navigation(navigation)
        )

        print("writing", DOC_PATH / f"{name}.html")
        (DOC_PATH / f"{name}.html").write_text(markup)


if __name__ == "__main__":

    render_templates([
        "index",
        "walkthrough",
        "cli",
        "expressions",
        "transforms",
        "constraints",
        "reference",
    ])

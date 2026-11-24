import re

from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout, QWidget


def markdown_to_html(markdown_text: str | None) -> str:
    """Convert common Markdown elements to HTML for display in Qt."""
    if not markdown_text:
        return ""

    html_text = markdown_text
    html_text = re.sub(
        r"```(?:\w+)?\n(.*?)\n```",
        r'<pre style="background-color:#f6f8fa; padding:10px; '
        r'border-radius:4px; overflow:auto;"><code>\1</code></pre>',
        html_text,
        flags=re.DOTALL,
    )
    html_text = re.sub(
        r"`([^`]+)`",
        r'<code style="background-color:#f6f8fa; padding:2px 4px; '
        r'border-radius:3px;">\1</code>',
        html_text,
    )
    html_text = re.sub(
        r"^# (.*?)$", r"<h1>\1</h1>", html_text, flags=re.MULTILINE
    )
    html_text = re.sub(
        r"^## (.*?)$", r"<h2>\1</h2>", html_text, flags=re.MULTILINE
    )
    html_text = re.sub(
        r"^### (.*?)$", r"<h3>\1</h3>", html_text, flags=re.MULTILINE
    )
    html_text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", html_text)
    html_text = re.sub(
        r"(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)",
        r"<i>\1</i>",
        html_text,
    )

    bullet_pattern = r"^([ \t]*)-[ \t]+(.*?)$"
    if re.search(bullet_pattern, html_text, re.MULTILINE):
        lines = html_text.split("\n")
        in_list = False
        processed_lines: list[str] = []
        for line in lines:
            bullet_match = re.match(bullet_pattern, line, re.MULTILINE)
            if bullet_match:
                if not in_list:
                    processed_lines.append('<ul style="margin-top:0; margin-bottom:0;">')
                    in_list = True
                _, content = bullet_match.groups()
                processed_lines.append(f"<li>{content}</li>")
            else:
                if in_list:
                    processed_lines.append("</ul>")
                    in_list = False
                processed_lines.append(line)
        if in_list:
            processed_lines.append("</ul>")
        html_text = "\n".join(processed_lines)

    html_text = re.sub(r"\[(.*?)\]\((.*?)\)", r'<a href="\2">\1</a>', html_text)
    return html_text.replace("\n", "<br>")


class ChatWithResultsDialog(QDialog):
    """Import-compatible placeholder for search chat until cited RAG is available."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *_legacy_args: object,
        **_legacy_kwargs: object,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Chat with Search Results")
        self.resize(640, 240)
        self.setEnabled(False)

        layout = QVBoxLayout(self)
        unavailable = QLabel(
            "Search-result chat is unavailable until citation-validated local "
            "RAG is implemented."
        )
        unavailable.setWordWrap(True)
        layout.addWidget(unavailable)

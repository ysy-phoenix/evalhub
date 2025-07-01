import argparse
import random

import orjson
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console(width=120, soft_wrap=False)


def print_messages(messages):
    for message in messages:
        role = message.get("role", "unknown")
        content = message.get("content", "").strip()
        tool_calls = message.get("tool_calls", [])

        if role == "user":
            title = "[bold green]👤 User[/]"
            console.print(Panel(content, title=title, border_style="green"))
        elif role == "assistant":
            title = "[bold blue]🤖 Assistant[/]"
            console.print(Panel(content, title=title, border_style="blue"))
            if tool_calls:
                tool_text = Text()
                title = "[bold magenta]🛠️ Tool Calls[/]"
                for tool_call in tool_calls:
                    tool_text.append(
                        f"🔧 Tool Call: {tool_call['function']['name']}\t"
                        f"arguments: {tool_call['function']['arguments']}",
                    )
                console.print(Panel(tool_text, title=title, border_style="magenta"))
        elif role == "tool":
            name = message.get("name", "")
            title = f"[bold magenta]🛠️ Tool Result[/] ([italic]{name}[/])"
            console.print(Panel(content, title=title, border_style="magenta"))
        elif role == "system":
            console.print(Panel(content, title=f"[bold yellow]⚙️ {role}[/]", border_style="yellow"))
        else:
            console.print(Panel(content, title=f"[bold red]? {role}[/]", border_style="red"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, required=True)
    parser.add_argument("--index", "-i", type=int, default=0)
    parser.add_argument("--random", "-r", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = []
    with open(args.file, "rb") as f:
        for line in f:
            data.append(orjson.loads(line))
    failed = [i for i, d in enumerate(data) if sum(d["response"]["reward"].values()) == 0.0]
    if args.random:
        args.index = random.choice(failed)
    print_messages(data[args.index]["response"]["messages"])

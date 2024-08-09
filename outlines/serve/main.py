import typer
from typing_extensions import Annotated

from outlines.function import extract_function_from_file

app = typer.Typer(
    no_args_is_help=True, help="Serve Outlines functions as APIs", add_completion=False
)


@app.command()
def serve(
    path: Annotated[
        str,
        typer.Argument(help="Path to the script that defines the Outlines function."),
    ],
    name: Annotated[
        str,
        typer.Option("--name", "-n", help="The name of the function in the script."),
    ] = "fn",
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Port to serve the function locally"),
    ] = 8000,
):
    """Serve the Outlines function."""

    with open(path) as file:
        content = file.read()

    _ = extract_function_from_file(content, name)

    # 1. Load the file and import objects
    # 2. Find the function by its name
    # 3. Create an API based on the prompt function's parameters and model name
    # 4. Return example of calling the API

    print(f"{path}{port}{name}")


def main():
    app()

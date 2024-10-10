import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

import requests
from pydantic import RootModel

from outlines import generate, models

if TYPE_CHECKING:
    from outlines.generate.api import SequenceGenerator
    from outlines.prompts import Prompt


class NewFunction:
    """Represents an Outlines function.

    Functions are a convenient way to encapsulate a prompt template, a language
    model and a Pydantic model that define the output structure. Once defined,
    the function can be called with arguments that will be used to render the
    prompt template.

    """

    model: str
    prompt_template: Optional[Callable] = None
    pydantic_schema: Optional[RootModel] = None

    def prompt(self, fn: Callable):
        pass

    def schema(self, model: RootModel):
        pass

    def __call__(self, *args, **kwargs):
        pass


@dataclass
class Function:
    """Represents an Outlines function.

    Functions are a convenient way to encapsulate a prompt template, a language
    model and a Pydantic model that define the output structure. Once defined,
    the function can be called with arguments that will be used to render the
    prompt template.

    """

    prompt_template: "Prompt"
    schema: Union[str, Callable, object]
    model_name: str
    generator: Optional["SequenceGenerator"] = None

    @classmethod
    def from_github(cls, program_path: str, function_name: str = "fn"):
        """Load a function stored on GitHub"""
        program_content = download_from_github(program_path)
        function = extract_function_from_file(program_content, function_name)

        return function

    def init_generator(self):
        """Load the model and initialize the generator."""
        model = models.transformers(self.model_name)
        self.generator = generate.json(model, self.schema)

    def __call__(self, *args, **kwargs):
        """Call the function.

        .. warning::

           This currently does not support batching.

        Parameters
        ----------
        args
            Values to pass to the prompt template as positional arguments.
        kwargs
            Values to pass to the prompt template as keyword arguments.

        """
        if self.generator is None:
            self.init_generator()

        prompt = self.prompt_template(*args, **kwargs)
        return self.generator(prompt)


def download_from_github(short_path: str):
    """Download the file in which the function is stored on GitHub."""
    GITHUB_BASE_URL = "https://raw.githubusercontent.com"
    BRANCH = "main"

    path = short_path.split("/")
    if len(path) < 3:
        raise ValueError(
            "Please provide a valid path in the form {USERNAME}/{REPO_NAME}/{PATH_TO_FILE}."
        )
    elif short_path[-3:] == ".py":
        raise ValueError("Do not append the `.py` extension to the program name.")

    username = path[0]
    repo = path[1]
    path_to_file = path[2:]

    url = "/".join([GITHUB_BASE_URL, username, repo, BRANCH] + path_to_file) + ".py"
    result = requests.get(url)

    if result.status_code == 200:
        return result.text
    elif result.status_code == 404:
        raise ValueError(
            f"Program could not be found at {url}. Please make sure you entered the GitHub username, repository name and path to the program correctly."
        )
    else:
        result.raise_for_status()


def extract_function_from_file(content: str, function_name: str) -> Tuple[Callable]:
    """Extract a function object from a downloaded file."""

    spec = importlib.util.spec_from_loader(
        "outlines_function", loader=None, origin="github"
    )
    if spec is not None:
        module = importlib.util.module_from_spec(spec)
        exec(content, module.__dict__)

        try:
            fn = getattr(module, function_name)
        except AttributeError:
            raise AttributeError(
                "Could not find an `outlines.Function` instance in the remote file. Make sure that the path you specified is correct."
            )

        if not isinstance(fn, module.outlines.Function):
            raise TypeError(
                f"The `{function_name}` variable in the program must be an instance of `outlines.Function`"
            )

    return fn

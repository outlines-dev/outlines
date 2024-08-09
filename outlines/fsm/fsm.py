import warnings
from typing import TYPE_CHECKING, Iterable, NewType, Optional

from outlines.fsm.guide import RegexGuide, StopAtEOSGuide

if TYPE_CHECKING:
    from outlines.models.tokenizer import Tokenizer

FSMState = NewType("FSMState", int)


class StopAtEosFSM(StopAtEOSGuide):
    """FSM to generate text until EOS has been generated."""

    def __init__(self, tokenizer: "Tokenizer"):
        warnings.warn(
            UserWarning(
                "The `StopAtTokenFSM` interface is deprecated and will be removed on 2024-06-01. Please use `StopAtEOSGuide` instead."
            )
        )
        super().__init__(tokenizer)

    def allowed_token_ids(self, state: FSMState) -> Optional[Iterable[int]]:
        next_instruction = self.get_next_instruction(state)
        return next_instruction.tokens

    def next_state(self, state: FSMState, token_id: int) -> FSMState:
        return FSMState(self.get_next_state(state, token_id))


class RegexFSM(RegexGuide):
    """FSM to generate text that is in the language of a regular expression."""

    def __init__(self, regex_string: str, tokenizer):
        warnings.warn(
            UserWarning(
                "The `RegexFSM` interface is deprecated and will be removed on 2024-06-01. Please use `RegexGuide` instead."
            )
        )
        super().__init__(regex_string, tokenizer)

    def allowed_token_ids(self, state: FSMState) -> Optional[Iterable[int]]:
        next_instruction = self.get_next_instruction(state)
        return next_instruction.tokens

    def next_state(self, state: FSMState, token_id: int) -> FSMState:
        return FSMState(self.get_next_state(state, token_id))

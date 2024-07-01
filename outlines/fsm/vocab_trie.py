import operator
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numba import njit, typed, types
from numba.cpython.hashing import (
    _Py_uhash_t,
    _PyHASH_XXPRIME_1,
    _PyHASH_XXPRIME_2,
    _PyHASH_XXPRIME_5,
    _PyHASH_XXROTATE,
    process_return,
)
from numba.experimental import jitclass, structref
from numba.extending import overload
from numba.typed import Dict

###########################
# Dict With Int[:] Key Impl
###########################


# Register type
@structref.register
class IntArrayDictType(types.StructRef):
    """
    Represents a dictionary using int64[:] as keys,
    intended for byte-level FSM representation with int64[:] transition.
    """

    def preprocess_fields(self, fields):
        return tuple(
            (name, typ.dtype if isinstance(typ, types.TypeRef) else typ)
            for name, typ in fields
        )


class IntArrayDict(structref.StructRefProxy):
    """Python proxy"""

    @property
    def wrapped_dict(self):
        return IntArrayDict_get_wrapped_dict(self)  # noqa: F821


structref.define_proxy(IntArrayDict, IntArrayDictType, ["wrapped_dict"])


@njit
def hash_key(key):
    """
    XXH64 Hash for int64[:] keys
    adapted from https://github.com/numba/numba/blob/556545/numba/cpython/hashing.py
    """
    acc = _PyHASH_XXPRIME_5
    for i in range(key.shape[0]):
        x = key[i]
        lane = hash(x)
        if lane == _Py_uhash_t(-1):
            return -1
        acc += lane * _PyHASH_XXPRIME_2
        acc = _PyHASH_XXROTATE(acc)
        acc *= _PyHASH_XXPRIME_1

    acc += key.shape[0] ^ (_PyHASH_XXPRIME_5 ^ _Py_uhash_t(3527539))

    if acc == _Py_uhash_t(-1):
        return process_return(1546275796)

    return process_return(acc)


@overload(IntArrayDict)
def custom_int_array_dict_constructor(value_type):
    if isinstance(value_type, types.Type):

        def impl(value_type):
            wrapped_dictionary = Dict.empty(types.intp, value_type)
            return IntArrayDict(wrapped_dictionary)

        return impl


@overload(operator.getitem)
def ol_int_array_dict_getitem(inst, key):
    if isinstance(inst, IntArrayDictType):

        def impl(inst, key):
            return inst.wrapped_dict[hash_key(key)]

        return impl


@overload(operator.setitem)
def ol_int_array_dict_setitem(inst, key, value):
    if isinstance(inst, IntArrayDictType):

        def impl(inst, key, value):
            inst.wrapped_dict[hash_key(key)] = value

        return impl


@overload(operator.contains)
def ol_int_array_dict_contains(inst, key):
    if isinstance(inst, IntArrayDictType):

        def impl(inst, key):
            return hash_key(key) in inst.wrapped_dict

        return impl


#################
# Vocab Trie Impl
#################

nb_int64_array_type = types.int64[:]

# use intp keys as that is the hash type,
# but the true key type is nb_int64_array_type
IntArrayToIntType = IntArrayDictType(
    (("wrapped_dict", types.DictType(types.intp, types.int64)),)
)
IntArrayToIntArrayType = IntArrayDictType(
    (("wrapped_dict", types.DictType(types.intp, nb_int64_array_type)),)
)


@jitclass(
    [
        ("token_to_token_key", IntArrayToIntType),
        ("token_key_to_token", types.DictType(types.int64, nb_int64_array_type)),
        (
            "token_key_to_child_token_keys",
            types.DictType(types.int64, nb_int64_array_type),
        ),
        ("token_to_token_ids", IntArrayToIntArrayType),
    ],
)
class VocabTrie:
    """
    VocabTrie: Class for efficient traversal of the vocabulary
    Bidirectional mapping between trie node ID and nb_unichar_2_type token
    - token_to_token_key: Dict[nb_unichar_2_array_type, int]
    - token_key_to_token: Dict[int, nb_unichar_2_array_type]
    Allow retrieval of children in trie
    - token_key_to_child_token_keys: Dict[int, int64[:]]
    Allow retrieval of of token_ids for a given token
    - token_to_token_ids: Dict[nb_unichar_2_array_type, int64[:]]
    Trie structure:
        Only members of the vocabulary are included as nodes, no intermediates.
        Structured to guarantee that recursive calls to get_children()
        will return every token once, only once.
        Given a vocabulary of ["a", "ab", "abc", "ac", "ace", "apple"],
        the children of "a" are "ab", "ac", "apple".
        "abc" and "ace" are excluded because they have intermediate parents in the vocabulary.
    """

    def __init__(
        self,
        all_token_transitions: List[Sequence[int]],
        vocabulary: List[Tuple[str, Sequence[int]]],
    ):
        self.token_to_token_key = IntArrayDict(
            typed.Dict.empty(types.intp, types.int64)
        )
        self.token_key_to_token = typed.Dict.empty(
            key_type=types.int64, value_type=nb_int64_array_type
        )
        self.token_key_to_child_token_keys = typed.Dict.empty(
            key_type=types.int64, value_type=nb_int64_array_type
        )
        self.token_to_token_ids = IntArrayDict(
            typed.Dict.empty(types.intp, nb_int64_array_type)
        )

        self._insert(all_token_transitions, vocabulary)

    def _insert(
        self,
        all_token_transitions: List[Sequence[int]],
        vocabulary: List[Tuple[str, Sequence[int]]],
    ) -> None:
        # Initialize an empty array for the root token key to store child token keys
        self.token_key_to_child_token_keys[-1] = np.empty((0,), types.int64)

        # It's necessary to insert shorter transition sequences (prefixes) first
        sorted_idx_transition_seq = sorted(
            enumerate(all_token_transitions), key=lambda x: len(x[1])
        )

        for idx, token_transitions in sorted_idx_transition_seq:
            token_ids = vocabulary[idx][1]
            if token_transitions not in self.token_to_token_key:
                # create bimapping between token and token_key (tokens trie node key)
                self.token_to_token_key[token_transitions] = idx
                self.token_key_to_token[idx] = token_transitions

                # find parent token key
                parent_token_key = -1  # root token
                for i in range(len(token_transitions) - 1, -1, -1):
                    prefix_token = token_transitions[:i]

                    if prefix_token in self.token_to_token_key:
                        parent_token_key = self.token_to_token_key[prefix_token]
                        break
                # map parent token to current token
                self.token_key_to_child_token_keys[parent_token_key] = np.append(
                    self.token_key_to_child_token_keys[parent_token_key],
                    np.array([idx]),
                )

                # map current token to empty list of children
                self.token_key_to_child_token_keys[idx] = np.empty((0,), types.int64)

                # set current tokens token ids
                self.token_to_token_ids[token_transitions] = token_ids

            else:
                # if exists, append to current tokens token ids
                self.token_to_token_ids[token_transitions] = np.append(
                    self.token_to_token_ids[token_transitions], token_ids
                )

    def get_children(self, token_transitions: Optional[Sequence[int]] = None):
        """
        Get the token_ids of all children for the given token_id.
        If token_id is None, get the root children.
        """
        if token_transitions is None:
            token_key = -1
        else:
            token_key = self.token_to_token_key[token_transitions]

        child_token_keys = self.token_key_to_child_token_keys[token_key]

        return [self.token_key_to_token[token_key] for token_key in child_token_keys]

    def get_token_ids(self, token):
        return self.token_to_token_ids[token]

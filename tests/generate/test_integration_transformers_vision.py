from io import BytesIO
from urllib.request import urlopen

import pytest
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

import outlines
from outlines.models.transformers_vision import transformers_vision

IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/7/71/2010-kodiak-bear-1.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/b/be/Tamias-rufus-001.jpg",
]


def img_from_url(url):
    img_byte_stream = BytesIO(urlopen(url).read())
    return Image.open(img_byte_stream).convert("RGB")


@pytest.fixture(scope="session")
def model(tmp_path_factory):
    return transformers_vision(
        "trl-internal-testing/tiny-random-LlavaForConditionalGeneration",
        model_class=LlavaForConditionalGeneration,
        device="cpu",
    )


@pytest.fixture(scope="session")
def processor(tmp_path_factory):
    return AutoProcessor.from_pretrained("llava-hf/llava-interleave-qwen-0.5b-hf")


def test_single_image_text_gen(model, processor):
    conversation = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is this?"}, {"type": "image"}],
        },
    ]
    generator = outlines.generate.text(model)
    sequence = generator(
        processor.apply_chat_template(conversation),
        [img_from_url(IMAGE_URLS[0])],
        seed=10000,
        max_tokens=10,
    )
    assert isinstance(sequence, str)


def test_multi_image_text_gen(model, processor):
    """If the length of image tags and number of images we pass are > 1 and equal,
    we should yield a successful generation.
    """
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do all these have in common?"},
            ]
            + [{"type": "image"} for _ in range(len(IMAGE_URLS))],
        },
    ]
    generator = outlines.generate.text(model)
    sequence = generator(
        processor.apply_chat_template(conversation),
        [img_from_url(i) for i in IMAGE_URLS],
        seed=10000,
        max_tokens=10,
    )
    assert isinstance(sequence, str)


def test_mismatched_image_text_gen(model, processor):
    """If the length of image tags and number of images we pass are unequal,
    we should raise an error.
    """
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "I'm passing 3 images, but only 1 image tag"},
                {"type": "image"},
            ],
        },
    ]
    generator = outlines.generate.text(model)
    with pytest.raises(ValueError):
        _ = generator(
            processor.apply_chat_template(conversation),
            [img_from_url(i) for i in IMAGE_URLS],
            seed=10000,
            max_tokens=10,
        )


def test_single_image_choice(model, processor):
    conversation = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is this?"}, {"type": "image"}],
        },
    ]
    choices = ["cat", "dog"]
    generator = outlines.generate.choice(model, choices)
    sequence = generator(
        processor.apply_chat_template(conversation),
        [img_from_url(IMAGE_URLS[0])],
        seed=10000,
        max_tokens=10,
    )
    assert isinstance(sequence, str)
    assert sequence in choices

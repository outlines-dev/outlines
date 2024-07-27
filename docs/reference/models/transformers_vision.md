# Transformers Vision

Outlines allows seamless use of [vision models](https://huggingface.co/learn/computer-vision-course/en/unit4/multimodal-models/tasks-models-part1).

`outlines.models.transformers_vision` has shares interfaces with, and is based on [outlines.models.transformers](./transformers.md).

Tasks supported include

- image + text -> text
- video + text -> text



## Example: Using [Llava-Next](https://huggingface.co/docs/transformers/en/model_doc/llava_next) Vision Models

Install dependencies
`pip install torchvision pillow flash-attn`

Create the model
```python
import outlines

model = outlines.models.transformers_vision(
    "llava-hf/llava-v1.6-mistral-7b-hf",
	device="cuda",
)
```

Create convenience function to load a `PIL.Image` from URL
```python
from PIL import Image
from io import BytesIO
from urllib.request import urlopen

def img_from_url(url):
    img_byte_stream = BytesIO(urlopen(url).read())
    return Image.open(img_byte_stream).convert("RGB")
```

### Describing an image

```python
description_generator = outlines.generate.text(model)
description_generator(
    "<image> detailed description:",
    [img_from_url("https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg")]
)
```

> This is a color photograph featuring a Siamese cat with striking blue eyes. The cat has a creamy coat and a light eye color, which is typical for the Siamese breed. Its features include elongated ears, a long, thin tail, and a striking coat pattern. The cat is sitting in an indoor setting, possibly on a cat tower or a similar raised platform, which is covered with a beige fabric, providing a comfortable and soft surface for the cat to rest or perch. The surface of the wall behind the cat appears to be a light-colored stucco or plaster.

#### Multiple Images

To include multiple images in your prompt you simply add more `<image>` tokens to the prompt

```python
image_urls = [
	"https://cdn1.byjus.com/wp-content/uploads/2020/08/ShapeArtboard-1-copy-3.png",  # triangle
	"https://cdn1.byjus.com/wp-content/uploads/2020/08/ShapeArtboard-1-copy-11.png",  # hexagon
]
description_generator = outlines.generate.text(model)
description_generator(
    "<image><image><image>What shapes are present?",
    list(map(img_from_url, image_urls)),
)
```

> There are two shapes present. One shape is a hexagon and the other shape is an triangle. '


### Classifying an Image

```python
pattern = "Mercury|Venus|Earth|Mars|Saturn|Jupiter|Neptune|Uranus|Pluto"
planet_generator = outlines.generate.regex(model, pattern)

planet_generator(
    "What planet is this: <image>",
    [img_from_url("https://upload.wikimedia.org/wikipedia/commons/e/e3/Saturn_from_Cassini_Orbiter_%282004-10-06%29.jpg")]
)
```

> Saturn


### Extracting Structured Image data

```python
from pydantic import BaseModel
from typing import List, Optional

class ImageData(BaseModel):
    caption: str
    tags_list: List[str]
    object_list: List[str]
    is_photo: bool

image_data_generator = outlines.generate.json(model, ImageData)

image_data_generator(
    "<image> detailed JSON metadata:",
    [img_from_url("https://upload.wikimedia.org/wikipedia/commons/9/98/Aldrin_Apollo_11_original.jpg")]
)
```

> `ImageData(caption='An astronaut on the moon', tags_list=['moon', 'space', 'nasa', 'americanflag'], object_list=['moon', 'moon_surface', 'space_suit', 'americanflag'], is_photo=True)`


## Resources

### Chosing a model
- https://mmbench.opencompass.org.cn/leaderboard
- https://huggingface.co/spaces/WildVision/vision-arena

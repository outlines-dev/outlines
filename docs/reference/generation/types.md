# Custom types

Outlines provides custom Pydantic types so you can focus on your use case rather than on writing regular expressions:

| Category | Type | Import | Description |
|:--------:|:----:|:-------|:------------|
| ISBN | 10 & 13 | `outlines.types.ISBN` | There is no guarantee that the [check digit][wiki-isbn] will be correct |
| Airport | IATA | `outlines.types.airports.IATA` | Valid [airport IATA codes][wiki-airport-iata] |
| Country | alpha-2 code | `outlines.types.airports.Alpha2` | Valid [country alpha-2 codes][wiki-country-alpha-2] |
|  | alpha-3 code | `outlines.types.countries.Alpha3` | Valid [country alpha-3 codes][wiki-country-alpha-3] |
|  | numeric code | `outlines.types.countries.Numeric` | Valid [country numeric codes][wiki-country-numeric] |
|  | name | `outlines.types.countries.Name` | Valid country names |
|  | flag | `outlines.types.countries.Flag` | Valid flag emojis |
| | email | `outlines.types.Email` | Valid email address |

Some types require localization. We currently only support US types, but please don't hesitate to create localized versions of the different types and open a Pull Request. Localized types are specified using `types.locale` in the following way:

```python
from outlines import types

types.locale("us").ZipCode
types.locale("us").PhoneNumber
```

Here are the localized types that are currently available:

| Category | Locale | Import | Description |
|:--------:|:----:|:-------|:------------|
| Zip code | US | `ZipCode` | Generate US Zip(+4) codes |
| Phone number  | US | `PhoneNumber` | Generate valid US phone numbers |


You can use these types in Pydantic schemas for JSON-structured generation:

```python
from pydantic import BaseModel

from outlines import models, generate, types

# Specify the locale for types
locale = types.locale("us")

class Client(BaseModel):
    name: str
    phone_number: locale.PhoneNumber
    zip_code: locale.ZipCode


model = models.transformers("microsoft/Phi-3-mini-4k-instruct")
generator = generate.json(model, Client)
result = generator(
    "Create a client profile with the fields name, phone_number and zip_code"
)
print(result)
# name='Tommy' phone_number='129-896-5501' zip_code='50766'
```

Or simply with `outlines.generate.format`:

```python
from pydantic import BaseModel

from outlines import models, generate, types


model = models.transformers("microsoft/Phi-3-mini-4k-instruct")
generator = generate.format(model, types.locale("us").PhoneNumber)
result = generator(
    "Return a US Phone number: "
)
print(result)
# 334-253-2630
```


We plan on adding many more custom types. If you have found yourself writing regular expressions to generate fields of a given type, or if you could benefit from more specific types don't hesite to [submit a PR](https://github.com/outlines-dev/outlines/pulls) or [open an issue](https://github.com/outlines-dev/outlines/issues/new/choose).


[wiki-isbn]: https://en.wikipedia.org/wiki/ISBN#Check_digits
[wiki-airport-iata]: https://en.wikipedia.org/wiki/IATA_airport_code
[wiki-country-alpha-2]: https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2
[wiki-country-alpha-3]: https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3
[wiki-country-numeric]: https://en.wikipedia.org/wiki/ISO_3166-1_numeric

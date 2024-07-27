---
title: Welcome to Outlines!
---

Outlines〰 is a Python library that allows you to use Large Language Model in a simple and robust way (with structured generation). It is built by [.txt][.txt]{:target="_blank"}, and is already used in production by many companies.

## What models do you support?

We support [Openai](reference/models/openai.md), but the true power of Outlines〰 is unleashed with Open Source models available via the [transformers](reference/models/transformers.md), [llama.cpp](reference/models/llamacpp.md), [exllama2](reference/models/exllamav2.md) and [mamba_ssm](reference/models/mamba.md) libraries. If you want to build and maintain an integration with another library, [get in touch][discord].

## What are the main features?

<div class="grid cards" markdown>
-   :material-code-json:{ .lg .middle } __Make LLMs generate valid JSON__

    ---

    No more invalid JSON outputs, 100% guaranteed

    [:octicons-arrow-right-24: Generate JSON](reference/json.md)

-   :material-keyboard-outline:{ .lg .middle } __JSON mode for vLLM__

    ---

    Deploy a LLM service using Outlines' JSON structured generation and vLLM

    [:octicons-arrow-right-24: Deploy outlines](reference/serve/vllm.md)


-   :material-regex:{ .lg .middle } __Make LLMs follow a Regex__

    ---

    Generate text that parses correctly 100% of the time

    [:octicons-arrow-right-24: Guide LLMs](reference/regex.md)

-    :material-chat-processing-outline:{ .lg .middle } __Powerful Prompt Templating__

     ---

     Better manage your prompts' complexity with prompt templating

    [:octicons-arrow-right-24: Learn more](reference/prompting.md)
</div>

## Why use Outlines?


Outlines〰 is built at [.txt][.txt] by engineers with decades of experience in software engineering, machine learning (Bayesian Statistics and NLP), and compilers. [.txt][.txt] is a VC-backed company fully focused on the topic of structured generation and is committed to make the community benefit from its experience.

We are also open source veterans and have authored/maintained many libraries over the years: the [Aesara][aesara]{:target="_blank"} and [Pythological][pythological]{:target="_blank"} ecosystems, [Blackjax][blackjax]{:target="_blank"} and [Hy][hy]{:target="_blank"} among many others.
.

Outlines does not use unnecessary abstractions that tend to get in your way. We have a laser focus on reliable text generation with LLMs, a clear roadmap to push the state of the art in this area and a commitment to clean and robust code.

And last but not least, unlike alternatives, Outlines' structured generation introduces **no overhead** during inference.


## Who is using Outlines?

Hundreds of organisations and the main LLM serving frameworks ([vLLM][vllm], [TGI][tgi], [LoRAX][lorax], [xinference][xinference], [SGLang][sglang]) are using Outlines. Some of the prominent companies and organizations that are using Outlines include:

<head>
  <style>
  .row {
      display: inline-block;
      width: 100%;
      margin-bottom: 50px;
      margin-top: 0px !important;
      break-inside: avoid;
  }

  /* Create two equal columns that sits next to each other */
  .column {
      column-count: 3;
      column-gap: 20px;
      padding: 20px;
  }

  </style>
</head>
<body>

<div class="column">
  <div class="row"><img src="../logos/amazon.png" width="200"></div>
  <div class="row"><img src="../logos/apple.png" width="200"></div>
  <div class="row"><img src="../logos/best_buy.png" width="200"></div>
  <div class="row"><img src="../logos/canoe.png" width="200"></div>
  <div class="row"><img src="../logos/cisco.png" width="200"></div>
  <div class="row"><img src="../logos/dassault_systems.png" width="200"></div>
  <div class="row"><img src="../logos/databricks.png" width="200"></div>
  <div class="row"><img src="../logos/datadog.png" width="200"></div>
  <div class="row"><img src="../logos/dbt_labs.png" width="200"></div>
  <div class="row"><img src="../assets/images/dottxt.png" width="200"></div>
  <div class="row"><img src="../logos/gladia.jpg" width="200"></div>
  <div class="row"><img src="../logos/harvard.png" width="200"></div>
  <div class="row"><img src="../logos/hf.png" width="200"></div>
  <div class="row"><img src="../logos/johns_hopkins.png" width="200"></div>
  <div class="row"><img src="../logos/meta.png" width="200"></div>
  <div class="row"><img src="../logos/mit.png" width="200"></div>
  <div class="row"><img src="../logos/mount_sinai.png" width="200"></div>
  <div class="row"><img src="../logos/nvidia.png" width="200"></div>
  <div class="row"><img src="../logos/nyu.png" width="200"></div>
  <div class="row"><img src="../logos/safran.png" width="200"></div>
  <div class="row"><img src="../logos/salesforce.png" width="200"></div>
  <div class="row"><img src="../logos/shopify.png" width="200"></div>
  <div class="row"><img src="../logos/smithsonian.png" width="200"></div>
  <div class="row"><img src="../logos/tinder.png" width="200"></div>
  <div class="row"><img src="../logos/upenn.png" width="200"></div>
</div>

</body>

Organizations are included either because they use Outlines as a dependency in a public repository, or because of direct communication between members of the Outlines team and employees at these organizations.

Still not convinced, read [what people say about us](community/feedback.md). And make sure to take a look at what the [community is building](community/examples.md)!

## Philosophy

**Outlines** 〰 is a library for neural text generation. You can think of it as a
more flexible replacement for the `generate` method in the
[transformers](https://github.com/huggingface/transformers) library.

**Outlines** 〰 helps developers *structure text generation* to build robust
interfaces with external systems. It provides generation methods that
guarantee that the output will match a regular expressions, or follow
a JSON schema.

**Outlines** 〰 provides *robust prompting primitives* that separate the prompting
from the execution logic and lead to simple implementations of few-shot
generations, ReAct, meta-prompting, agents, etc.

**Outlines** 〰 is designed as a *library* that is meant to be compatible the
broader ecosystem, not to replace it. We use as few abstractions as possible,
and generation can be interleaved with control flow, conditionals, custom Python
functions and calls to other libraries.

**Outlines** 〰 is *compatible with every auto-regressive model*. It only interfaces with models
via the next-token logits distribution.

## Outlines people

Outlines would not be what it is today without a community of dedicated developers:

<a href="https://github.com/outlines-dev/outlines/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=outlines-dev/outlines" />
</a>

## Acknowledgements

<div class="grid" markdown>


<figure markdown>
  <a href="http://www.dottxt.co">
  ![Normal Computing logo](assets/images/dottxt.png){ width="150" }
  </a>
</figure>

<figure markdown>
  <a href="https://www.normalcomputing.ai">
  ![Normal Computing logo](assets/images/normal_computing.jpg){ width="150" }
  </a>
</figure>

</div>

Outlines was originally developed at [@NormalComputing](https://twitter.com/NormalComputing) by [@remilouf](https://twitter.com/remilouf) and [@BrandonTWillard](https://twitter.com/BrandonTWillard). It is now maintained by [.txt](https://dottxt.co).


[discord]: https://discord.gg/R9DSu34mGd
[aesara]: https://github.com/aesara-devs
[blackjax]: https://github.com/blackjax-devs/blackjax
[pythological]: https://github.com/pythological
[hy]: https://hylang.org/
[.txt]: https://dottxt.co
[vllm]: https://github.com/vllm-project/vllm
[tgi]: https://github.com/huggingface/text-generation-inference
[lorax]: https://github.com/predibase/lorax
[xinference]: https://github.com/xorbitsai/inference
[sglang]: https://github.com/sgl-project/sglang/

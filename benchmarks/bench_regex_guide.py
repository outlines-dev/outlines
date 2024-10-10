from outlines.caching import cache_disabled
from outlines.fsm.guide import RegexGuide

from .common import setup_tokenizer

regex_samples = {
    "email": r"[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?",
    "complex_phone": "\\+?\\d{1,4}?[-.\\s]?\\(?\\d{1,3}?\\)?[-.\\s]?\\d{1,4}[-.\\s]?\\d{1,4}[-.\\s]?\\d{1,9}",
    "simple_phone": "\\+?[1-9][0-9]{7,14}",
    "date": r"([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])(\.|-|/)([1-9]|0[1-9]|1[0-2])(\.|-|/)([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])|([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])(\.|-|/)([1-9]|0[1-9]|1[0-2])(\.|-|/)([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])",
    "time": r"(0?[1-9]|1[0-2]):[0-5]\d\s?(am|pm)?",
    "ip": r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)",
    "url": r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?",
    "ssn": r"\d{3}-\d{2}-\d{4}",
    "complex_span_constrained_relation_extraction": "(['\"\\ ,]?((?:of|resulting|case|which|cultures|a|core|extreme|selflessness|spiritual|various|However|both|vary|in|other|secular|the|religious|among|moral|and|It|object|worldviews|altruism|traditional|material|aspect|or|life|beings|virtue|is|however|opposite|concern|an|practice|it|for|s|quality|religions|In|Altruism|animals|happiness|many|become|principle|human|selfishness|may|synonym)['\"\\ ,]?)+['\"\\ ,]?\\s\\|\\s([^|\\(\\)\n]{1,})\\s\\|\\s['\"\\ ,]?((?:of|resulting|case|which|cultures|a|core|extreme|selflessness|spiritual|various|However|both|vary|in|other|secular|the|religious|among|moral|and|It|object|worldviews|altruism|traditional|material|aspect|or|life|beings|virtue|is|however|opposite|concern|an|practice|it|for|s|quality|religions|In|Altruism|animals|happiness|many|become|principle|human|selfishness|may|synonym)['\"\\ ,]?)+['\"\\ ,]?(\\s\\|\\s\\(([^|\\(\\)\n]{1,})\\s\\|\\s([^|\\(\\)\n]{1,})\\))*\\n)*",
}


class RegexGuideBenchmark:
    params = regex_samples.keys()

    def setup(self, pattern_name):
        self.tokenizer = setup_tokenizer()
        self.pattern = regex_samples[pattern_name]

    @cache_disabled()
    def time_regex_to_guide(self, pattern_name):
        RegexGuide.from_regex(self.pattern, self.tokenizer)


class MemoryRegexGuideBenchmark:
    params = ["simple_phone", "complex_span_constrained_relation_extraction"]

    def setup(self, pattern_name):
        self.tokenizer = setup_tokenizer()
        self.pattern = regex_samples[pattern_name]

    @cache_disabled()
    def peakmem_regex_to_guide(self, pattern_name):
        RegexGuide.from_regex(self.pattern, self.tokenizer)

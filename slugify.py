import re


def slugify(name):
    name = name.split()
    name = re.sub("[\(\[].*?[\)\]]", "", ' '.join(name)).strip().split()
    name = re.sub('[^\w\s-]', '', "_".join(name)).strip().lower()
    name = str(re.sub('[-\s]+', '_', name))
    return name + "_preview_mono.wav.down.wav"

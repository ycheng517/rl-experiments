import argparse


def parse_val(val):
    try:
        int_val = int(val)
        return int_val
    except ValueError:
        pass

    try:
        float_val = float(val)
        return float_val
    except ValueError:
        pass

    return val


def is_int(val):
    try:
        num = int(val)
    except ValueError:
        return False
    return True


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = parse_val(value)

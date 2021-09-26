def round_float_to_str(value):
    if value is None:
        return "None"
    if abs(value) > 1e-3:
        print_value = "{:.4f}".format(value)
    else:
        print_value = "{:.4e}".format(value)
    return print_value

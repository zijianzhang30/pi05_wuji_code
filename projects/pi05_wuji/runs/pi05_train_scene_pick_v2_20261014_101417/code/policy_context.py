def instruction(base, control_step):
    step = int(control_step)
    assert step == control_step and 0 <= step <= 100000
    return base.strip() + f" Elapsed control steps since reset: {step}."

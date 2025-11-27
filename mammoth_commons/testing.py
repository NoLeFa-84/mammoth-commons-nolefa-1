import os

if os.getcwd().endswith(r"mammoth\tests"):
    import warnings

    warnings.warn(
        "\nThere was likely an attempt to import mammoth from within its `...mammoth/tests/` folder."
        "\nThis could fail to import modules from the `...mammoth/mai_bias/catalogue/` folder for the tests,"
        "\nso `os.chdir('..')` command was applied first. If this still fails, make `import mammoth` "
        "\nyour first import. This message will not appear if you correctly set up a run configuration "
        "\nthat uses the top level of mammoth as a working directory when running tests."
    )
    os.chdir("..")


def unwrap(component):
    """import subprocess
    from mammoth_commons.externals import notify_progress, notify_end

    notify_progress(0, "Installing dependencies")
    subprocess.run(
        component.component_spec.implementation.container.command[:2]
        + [
            component.component_spec.implementation.container.command[2].rsplit(
                "&&", 1
            )[0]
        ]
    )
    notify_end()"""
    return component.python_func.__mammoth_wrapped__


class Env:
    def __init__(self, *args):
        for v in args:
            v = unwrap(v)
            setattr(self, v.__name__, v)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

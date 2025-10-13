import json
import re
import string

import readchar
import os
import glob
import shutil
import csv
from datetime import datetime
from mai_bias.backend.loaders import registry
from mammoth_commons.externals import pd_read_csv, get_model_layer_list


tags = {
    key: "<h1>" + key + "</h1>" + module["description"]
    for key, module in (
        registry.dataset_loaders | registry.model_loaders | registry.analysis_methods
    ).items()
}


def find_columns(path, delimiter):
    if path is None:
        print(
            colors.fail
            + f"No previous file in this set of parameters.".rjust(78)
            + colors.reset
        )
        return []
    if len(path) == 0:
        print(colors.fail + f"The previous file was not set.".rjust(78) + colors.reset)
        return []
    if delimiter is not None and len(delimiter) == 0:
        print(
            colors.fail
            + f"The previous delimiter was not set.".rjust(78)
            + colors.reset
        )
        return []
    try:
        if delimiter is None:
            try:
                with open(path, "r") as file:
                    sample = file.read(4096)
                    sniffer = csv.Sniffer()
                    delimiter = sniffer.sniff(sample).delimiter
                    delimiter = str(delimiter)
                    import string

                    if delimiter in string.ascii_letters:
                        common_delims = [",", ";", "|", "\t"]
                        counts = {d: sample.count(d) for d in common_delims}
                        # pick the one with highest count, fallback to ","
                        delimiter = (
                            max(counts, key=counts.get) if any(counts.values()) else ","
                        )
            except Exception as e:
                delimiter = ","
        df = pd_read_csv(path, nrows=3, on_bad_lines="skip", delimiter=delimiter)
        return df.columns.tolist()
    except Exception as e:
        print(colors.fail + str(e).rjust(78) + colors.reset)
        return []


def remove_first_h1(html):
    return re.sub(
        r"<h1\b[^>]*>.*?</h1>", "", html, count=1, flags=re.DOTALL | re.IGNORECASE
    )


def autocomplete_path(partial_path: str) -> list:
    partial_path = os.path.expanduser(partial_path)
    if os.path.isdir(partial_path):
        pattern = os.path.join(partial_path, "*")
    else:
        dirname = os.path.dirname(partial_path) or "."
        basename = os.path.basename(partial_path)
        pattern = os.path.join(dirname, basename + "*")
    matches = glob.glob(pattern)
    matches = [match + "/" if os.path.isdir(match) else match for match in matches]
    return matches


def common_starts(paths):
    if not paths:
        return ""
    try:
        return os.path.commonprefix(paths)
    except ValueError:
        return ""


def now():
    return datetime.now().strftime("%y-%m-%d %H:%M")


def save_all_runs(path, runs):
    copy_runs = list()
    for run in runs:
        copy_run = dict()
        copy_run["timestamp"] = run["timestamp"]
        copy_run["description"] = run["description"]
        copy_run["status"] = run.get("status", None)
        if "dataset" in run:
            copy_run["dataset"] = {
                "module": run["dataset"]["module"],
                "params": run["dataset"]["params"],
            }
        if "model" in run:
            copy_run["model"] = {
                "module": run["model"]["module"],
                "params": run["model"]["params"],
            }
        if "analysis" in run:
            copy_run["analysis"] = {
                "module": run["analysis"]["module"],
                "params": run["analysis"]["params"],
                "return": run["analysis"].get("return", None),
            }
        copy_runs.append(copy_run)
    with open(path, "w", encoding="utf-8") as file:
        file.write(json.dumps(copy_runs))


def load_all_runs(path):
    if not os.path.exists(path):
        return list()
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def format_name(name):
    """Format parameter names for better display."""
    return name.replace("_", " ").capitalize()


def extract_title(run):
    try:
        match = re.search(
            r"<h1\b[^>]*>.*?</h1>",
            run.get("analysis", dict()).get("return", ""),
            re.DOTALL,
        )
        if match:
            return match.group().replace("h1", "span")
    except Exception:
        pass
    return ""


class colorsbg:
    fail = "\033[41m"
    ok = "\033[42m"
    warn = "\033[43m"
    element = "\033[44m"
    # neutral = "\033[45m"
    neutral = "\033[100m"


class colors:
    reset = "\033[0m"
    fail = "\033[31m"
    ok = "\033[32m"
    warn = "\033[33m"
    element = "\033[34m"
    # neutral = "\033[35m"
    neutral = "\033[0m"


class Preview:
    def __init__(self, results, base_state, title):
        results = remove_first_h1(results)
        self.base_state = base_state
        self.title = title
        self.selection = 0
        self.base_state = base_state
        self.unhandled_button = None
        self.modifying_pos = 0
        self.modifying = False
        self.next = self
        self.title = title
        self.runs = runs
        self.height = 15
        import html2text
        from rich.console import Console
        from rich.markdown import Markdown

        results = html2text.html2text(results)
        console = Console()
        with console.capture() as capture:
            console.print(Markdown(results), width=80)
        self.results = capture.get().split("\n")

    def show(self):
        print("\x1b[2J\x1b[H")
        print(self.title)
        print("─" * 80)

        width, height = shutil.get_terminal_size()
        self.height = max(15, height - 10)

        if self.selection >= len(self.results) - self.height:
            self.selection = len(self.results) - self.height
        if self.selection < 0:
            self.selection = 0
        selection_end = min(len(self.results), self.selection + self.height)
        results = self.results[self.selection : selection_end]
        print("\n".join(results))
        # print("─" * 80)

        print(colorsbg.fail + f"Close".ljust(80) + colors.reset)
        print("─" * 80)

        if self.modifying:
            self.modifying = False
            self.next = self.base_state
            self.next.next = self.next


class Select:
    def __init__(self, options, base_state, title=None, runs=None, reference=None):
        self.selection = 0
        self.base_state = base_state
        self.unhandled_button = None
        self.options = options
        self.modifying_pos = 0
        self.modifying = False
        self.next = self
        self.title = title
        self.runs = runs
        self.reference = reference

    def show(self):
        if self.selection < 0:
            self.selection = 0
        if self.selection >= len(self.options):
            self.selection = len(self.options) - 1
        if self.title:
            print("\x1b[2J\x1b[H")
            print(self.title)
            print("─" * 80)
        else:
            self.base_state.show()

        width, height = shutil.get_terminal_size()
        height = max(15, height - 10)
        selection_end = min(len(self.options), self.selection + height)

        for i, option in enumerate(self.options):
            if i < selection_end - height or i >= selection_end:
                continue
            coloring = colorsbg if i == self.selection else colors
            print(f"{option[0](coloring)}{colors.reset}")
        print("─" * 80)

        if self.modifying:
            self.modifying = False
            for i, option in enumerate(self.options):
                if i == state.selection:
                    funcname = option[1]
                    if isinstance(funcname, str):
                        getattr(self, option[1])()
                    else:
                        funcname[0][funcname[1]] = funcname[2]
                        self.cancel()

    def data_loader(self):
        run = self.runs[self.reference]
        results = run.get("dataset", dict()).get("module", "No data loader")
        self.next = Preview(
            tags[results] if results in tags else "No description available.",
            self,
            colors.warn + "Info: " + results + colors.reset,
        )

    def model_loader(self):
        run = self.runs[self.reference]
        results = run.get("model", dict()).get("module", "No model loader")
        self.next = Preview(
            tags[results] if results in tags else "No description available.",
            self,
            colors.warn + "Info: " + results + colors.reset,
        )

    def analysis_method(self):
        run = self.runs[self.reference]
        results = run.get("analysis", dict()).get("module", "No analysis method")
        self.next = Preview(
            tags[results] if results in tags else "No description available.",
            self,
            colors.warn + "Info: " + results + colors.reset,
        )

    def results(self):
        run = self.runs[self.reference]
        results = run.get("analysis", dict()).get("return", "No results available.")
        self.next = Preview(results, self, self.title)

    def html(self):
        run = self.runs[self.reference]
        results = run.get("analysis", dict()).get("return", "No results available.")
        with open("temp.html", "w", encoding="utf-8") as file:
            file.write(results)
        print(colors.ok + f"Saved as temp.html".rjust(78) + colors.reset)
        try:
            import webbrowser

            webbrowser.open_new("temp.html")
            print(
                colors.ok + f"Opened temp.html in the browser".rjust(78) + colors.reset
            )
        except Exception as e:
            print(
                colors.fail
                + f"Failed to open the browser: {str(e)}".rjust(78)
                + colors.reset
            )

    def cancel(self):
        self.next = self.base_state
        self.next.next = self.next

    def delete(self):
        del self.runs[self.reference]
        self.cancel()

    def variation(self):
        new_run = self.runs[self.reference].copy()
        new_run["status"] = "new"
        new_run["timestamp"] = now()
        self.runs.append(new_run)
        self.reference = len(self.runs) - 1
        loaders = [loader for loader, values in registry.dataset_loaders.items()]
        self.next = Step(
            loaders,
            self.base_state,
            colors.warn + "1/3 Dataset loader" + colors.reset,
            new_run,
        )

    def edit(self):
        run = self.runs.pop(self.reference)
        self.runs.append(run)
        self.reference = len(self.runs) - 1
        loaders = [loader for loader, values in registry.dataset_loaders.items()]
        self.next = Step(
            loaders,
            self.base_state,
            colors.warn + "1/3 Dataset loader" + colors.reset,
            run,
        )


class Step:
    def __init__(self, modules, base_state, title, run, module_discovery="dataset"):
        self.base_state = base_state
        self.selection = -1
        self.modifying = False
        self.modifying_pos = 0
        self.run = run
        self.next = self
        self.title = title
        self.unhandled_button = None
        self.inputs = list()
        self.modules = modules
        self.selected_module = 0
        self.input_character = ""
        self.module_discovery = module_discovery
        if module_discovery == "dataset":
            self.registry = registry.dataset_loaders
        elif module_discovery == "model":
            self.registry = registry.model_loaders
        else:
            self.registry = registry.analysis_methods
        for i, module in enumerate(modules):
            if module == run.get(module_discovery, dict()).get("module", ""):
                self.next.selected_module = i

    def cancel(self):
        save_all_runs("history.json", self.base_state.runs)
        self.next = self.base_state
        self.next.next = self.next

    def find_delimiter(self, path, default):
        if path is None:
            print(colors.fail + "There is nor previous file".rjust(78) + colors.reset)
            return default
        try:
            import csv
            import string

            with open(path, "r") as file:
                sample = file.read(4096)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                delimiter = str(delimiter)
                if delimiter in string.ascii_letters:
                    common_delims = [",", ";", "|", "\t"]
                    counts = {d: sample.count(d) for d in common_delims}
                    # pick the one with highest count, fallback to ","
                    delimiter = (
                        max(counts, key=counts.get) if any(counts.values()) else ","
                    )
                return delimiter
        except Exception as e:
            print(
                colors.fail
                + f"Could not read the previous file".rjust(78)
                + colors.reset
            )
            return default

    def show(self):

        print("\x1b[2J\x1b[H")
        print(self.title)
        print("─" * 80)

        module_name = self.modules[self.selected_module]
        module = self.registry[module_name]

        if self.selection < -2:
            self.selection = -2

        if self.selection == -1:
            self.selected_module += self.modifying_pos
            if self.selected_module < 0:
                self.selected_module = len(self.modules) - 1
            if self.selected_module >= len(self.modules):
                self.selected_module = 0

            module_name = self.modules[self.selected_module]
            module = self.registry[module_name]
            if self.module_discovery in self.run:
                self.run[self.module_discovery]["module"] = module_name

            if self.modifying:
                self.modifying = False
                self.next = Preview(
                    (
                        tags[module_name]
                        if module_name in tags
                        else "No description available."
                    )
                    # + "<br><br><i>This appeared because you pressed [enter] during module selection. "
                    # + "Use left/right arrows to change the selection.</i>"
                    ,
                    self,
                    colors.warn
                    + "Info: "
                    + format_name(module_name)
                    + ""
                    + colors.reset,
                )
        elif self.selection == -2 and self.modifying:
            self.cancel()
            self.modifying = False
        else:
            stricter = 2 if self.module_discovery == "analysis" else 0
            if self.selection >= len(module["parameters"]) - stricter:
                self.selection = len(module["parameters"]) - stricter
        stricter = 2 if self.module_discovery == "analysis" else 0

        coloring = colorsbg if -2 == state.selection else colors
        print(f"{coloring.warn}{'Cancel'.ljust(80)}{colors.reset}")
        coloring = colorsbg if -1 == self.selection else colors
        print(
            f"{coloring.element}{'Loader'.ljust(30)} {'← '+format_name(module_name).center(44)+' → '}{colors.reset}"
        )

        if self.module_discovery not in self.run:
            self.run[self.module_discovery] = {"module": module_name, "params": dict()}
        i = 0
        for name, param_type, default, description in module["parameters"]:
            if name == "model" or name == "dataset":
                continue
            if self.modifying and i == self.selection:
                self.modifying = False
                self.next = Preview(
                    description
                    # + "<br><h3>What's this?</h3><i>The above description appeared because you pressed [enter] on a parameter. "
                    # + "Depending on the type of the parameter, use left/right arrows to change a selection, [tab] for autocomplete, or type in a string value.</i>"
                    ,
                    self,
                    colors.warn + "Info: " + format_name(name) + "" + colors.reset,
                )

            coloring = colorsbg if i == self.selection else colors
            if name not in self.run[self.module_discovery]["params"]:
                self.run[self.module_discovery]["params"][name] = (
                    "" if default is None or default == "None" else str(default)
                )

            param_options = module.get("parameter_options", {}).get(name, [])

            if isinstance(self.run[self.module_discovery]["params"][name], list):
                self.run[self.module_discovery]["params"][name] = ", ".join(
                    self.run[self.module_discovery]["params"][name]
                )

            if len(param_options) == 0 and param_type != "bool":
                if (
                    i == self.selection
                    and self.input_character == readchar.key.BACKSPACE
                ):
                    self.run[self.module_discovery]["params"][name] = self.run[
                        self.module_discovery
                    ]["params"][name][:-1]
                elif (
                    i == self.selection
                    and len(self.input_character) == 1
                    and self.input_character.isprintable()
                ):
                    self.run[self.module_discovery]["params"][
                        name
                    ] += self.input_character

            if param_options:
                option_position = 0
                for j, option in enumerate(param_options):
                    if self.run[self.module_discovery]["params"][name] == option:
                        option_position = j
                if self.modifying_pos != 0 and i == self.selection:
                    option_position += self.modifying_pos
                    if option_position < 0:
                        option_position = len(param_options) - 1
                    if option_position >= len(param_options):
                        option_position = 0
                self.run[self.module_discovery]["params"][name] = param_options[
                    option_position
                ]
                print(
                    f"{coloring.neutral}{format_name(name).ljust(30)} {'← '+self.run[self.module_discovery]['params'][name].center(44)+' → '}{colors.reset}"
                )
            elif param_type == "bool":
                if self.modifying_pos != 0 and i == self.selection:
                    self.run[self.module_discovery]["params"][name] = (
                        "True"
                        if self.run[self.module_discovery]["params"][name] == "False"
                        else "False"
                    )
                print(
                    f"{coloring.neutral}{format_name(name).ljust(30)} {'← '+str(self.run[self.module_discovery]['params'][name]).center(44)+' → '}{colors.reset}"
                )
            else:
                print(
                    f'{coloring.neutral}{format_name(name).ljust(30)} {str(self.run[self.module_discovery]["params"][name]).ljust(49)}{colors.reset}'
                )
            # print(param_type)
            i += 1

        coloring = (
            colorsbg
            if self.selection == len(module["parameters"]) - stricter == state.selection
            else colors
        )
        print(f"{coloring.element}{'Next'.ljust(80)}{colors.reset}")

        print("─" * 80)

        i = 0
        last_url = None
        last_delimiter = None
        for name, param_type, default, description in module["parameters"]:
            if name == "model" or name == "dataset":
                continue
            lower_name = name.lower()
            if "delimiter" in lower_name:
                last_delimiter = self.run[self.module_discovery]["params"][name]
            if i == self.selection and self.input_character != readchar.key.TAB:
                if "layer" in lower_name:
                    print(colors.warn + f"[tab] to autodetect".rjust(78) + colors.reset)
                elif "delimiter" in lower_name:
                    print(colors.warn + f"[tab] to autodetect".rjust(78) + colors.reset)
                elif (
                    "numeric" in lower_name
                    or "categorical" in lower_name
                    or "label" in lower_name
                    or "target" in lower_name
                    or "ignored" in lower_name
                    or "attribute" in lower_name
                    or "sensitive" in lower_name
                ):
                    print(
                        colors.warn
                        + f"[tab] to see a list of options".rjust(78)
                        + colors.reset
                    )
                elif param_type == "url" or "path" in lower_name or "dir" in lower_name:
                    paths = autocomplete_path(
                        self.run[self.module_discovery]["params"][name]
                    )
                    if len(paths) == 0:
                        print(
                            colors.fail
                            + "No path starting this way exists".rjust(78)
                            + colors.reset
                        )
                    elif len(paths) == 1 and os.path.exists(
                        self.run[self.module_discovery]["params"][name]
                    ):
                        print(colors.ok + "Path found".rjust(78) + colors.reset)
                    elif len(paths) == 1:
                        print(
                            colors.warn
                            + "[tab] to autocomplete".rjust(78)
                            + colors.reset
                        )
                    elif len(paths) <= 5:
                        print(
                            colors.warn
                            + f"[tab] to choose from {len(paths)} paths".rjust(78)
                            + colors.reset
                        )
                        for path in paths:
                            print(path.rjust(78))
                        print()
                    else:
                        print(
                            colors.warn
                            + f"[tab] to choose from {len(paths)} paths".rjust(78)
                            + colors.reset
                        )
                        for path in paths[:4]:
                            print(path.rjust(78))
                        print("...".rjust(78))
                        print()

            if i == self.selection and self.input_character == readchar.key.TAB:
                self.modifying_pos = 0
                self.input_character = ""
                if "layer" in lower_name:
                    paths = get_model_layer_list(
                        self.run.get("model", dict()).get("return", None)
                    )
                    self.next = Select(
                        [
                            (
                                lambda col: getattr(col, "warn") + "Cancel".ljust(80),
                                "cancel",
                            )
                        ]
                        + [
                            (
                                lambda col, path=path: getattr(col, "neutral")
                                + path.ljust(80),
                                (self.run[self.module_discovery]["params"], name, path),
                            )
                            for path in paths
                        ],
                        self,
                        colors.warn
                        + f"Select from {len(paths)} {format_name(name)} layers"
                        + colors.reset,
                    )
                elif "delimiter" in lower_name:
                    prev = self.run[self.module_discovery]["params"][name]
                    self.run[self.module_discovery]["params"][name] = (
                        self.find_delimiter(last_url, prev)
                    )
                    if self.run[self.module_discovery]["params"][name] != prev:
                        self.show()
                        print(
                            colors.fail
                            + "Successfully detected delimiter".rjust(78)
                            + colors.reset
                        )
                elif (
                    "numeric" in lower_name
                    or "categorical" in lower_name
                    or "label" in lower_name
                    or "target" in lower_name
                    or "ignored" in lower_name
                    or "attribute" in lower_name
                    or "sensitive" in lower_name
                ):
                    if "sensitive" == lower_name:
                        paths = self.run["dataset"]["return"]
                        paths = (
                            [""]
                            if paths is None or not hasattr(paths, "cols")
                            else paths.cols
                        )
                    else:
                        paths = find_columns(last_url, last_delimiter)
                    if len(paths) <= 5:
                        self.show()
                        print(
                            colors.warn
                            + f"Suggesting {len(paths)} {format_name(name)} options".rjust(
                                78
                            )
                            + colors.reset
                        )
                        i = 0
                        for path in paths:
                            print(path.rjust(78))
                            i += 1
                    else:
                        self.next = Preview(
                            "\n<br>".join(paths),
                            self,
                            colors.warn
                            + f"Suggesting {len(paths)} {format_name(name)} options"
                            + colors.reset,
                        )
                elif param_type == "url" or "path" in lower_name or "dir" in lower_name:
                    paths = autocomplete_path(
                        self.run[self.module_discovery]["params"][name]
                    )
                    if len(paths) == 0:
                        print(
                            colors.fail
                            + "No path starting this way exists".rjust(78)
                            + colors.reset
                        )
                    elif len(paths) == 1:
                        self.run[self.module_discovery]["params"][name] = paths[0]
                        self.show()
                        print(colors.ok + "Path autocompleted".rjust(78) + colors.reset)
                    else:
                        self.next = Select(
                            [
                                (
                                    lambda col: getattr(col, "warn")
                                    + "Cancel".ljust(80),
                                    "cancel",
                                )
                            ]
                            + [
                                (
                                    lambda col, path=path: getattr(col, "neutral")
                                    + path.ljust(80),
                                    (
                                        self.run[self.module_discovery]["params"],
                                        name,
                                        path,
                                    ),
                                )
                                for path in paths
                            ],
                            self,
                            colors.warn
                            + f"Could not fully autocomplete due to {len(paths)} options"
                            + colors.reset,
                        )

            if param_type == "url":
                last_url = self.run[self.module_discovery]["params"][name]
            i += 1

        self.modifying_pos = 0
        self.input_character = ""

        if self.selection == len(module["parameters"]) - stricter and self.modifying:
            self.modifying = False
            if self.module_discovery == "dataset":
                try:
                    params = {
                        param[0]: self.run[self.module_discovery]["params"][param[0]]
                        for param in module["parameters"]
                    }
                    self.run[self.module_discovery]["return"] = (
                        registry.name_to_runnable[module_name](**params)
                    )
                    self.run[self.module_discovery]["params"] = params
                    loaders = [
                        loader
                        for loader, values in registry.model_loaders.items()
                        if module_name in values["compatible"]
                    ]
                    self.next = Step(
                        loaders,
                        self.base_state,
                        colors.warn + "2/3 Model loader" + colors.reset,
                        self.run,
                        "model",
                    )
                except Exception as e:
                    print(colors.fail + str(e).rjust(78) + colors.reset)
                save_all_runs("history.json", self.base_state.runs)
            elif self.module_discovery == "model":
                try:
                    params = {
                        param[0]: self.run[self.module_discovery]["params"][param[0]]
                        for param in module["parameters"]
                    }
                    self.run[self.module_discovery]["return"] = (
                        registry.name_to_runnable[module_name](**params)
                    )
                    self.run[self.module_discovery]["params"] = params
                    compatible_methods = [
                        method
                        for method, entries in registry.analysis_methods.items()
                        if issubclass(
                            registry.parameters_to_class[self.run["dataset"]["module"]][
                                "return"
                            ],
                            registry.parameters_to_class[method][
                                entries["parameters"][0][0]
                            ],
                        )
                        and issubclass(
                            registry.parameters_to_class[self.run["model"]["module"]][
                                "return"
                            ],
                            registry.parameters_to_class[method][
                                entries["parameters"][1][0]
                            ],
                        )
                    ]
                    self.next = Step(
                        compatible_methods,
                        self.base_state,
                        colors.warn + "3/3 Analysis method" + colors.reset,
                        self.run,
                        "analysis",
                    )
                except Exception as e:
                    print(colors.fail + str(e).rjust(78) + colors.reset)
                save_all_runs("history.json", self.base_state.runs)
            elif self.module_discovery == "analysis":
                try:
                    params = {
                        param[0]: self.run[self.module_discovery]["params"][param[0]]
                        for param in module["parameters"]
                        if param[0] != "model" and param[0] != "dataset"
                    }
                    params["dataset"] = self.run["dataset"]["return"]
                    params["model"] = self.run["model"]["return"]
                    sensitive = params.get("sensitive", "")
                    if "," in sensitive:
                        sensitive = sensitive.split(",")
                    elif sensitive == "":
                        sensitive = []
                    else:
                        sensitive = [sensitive]
                    sensitive = [s.strip() for s in sensitive]
                    params["sensitive"] = sensitive
                    self.run[self.module_discovery]["return"] = (
                        registry.name_to_runnable[module_name](**params)
                    )
                    del params["model"]
                    del params["dataset"]
                    self.run["status"] = "completed"
                    self.run[self.module_discovery]["params"] = params
                    self.run[self.module_discovery]["return"] = self.run[
                        self.module_discovery
                    ]["return"].all()

                    run = self.run
                    description = run["description"]
                    if not description:
                        description = "..."
                    description = description.ljust(20)
                    title = extract_title(run)
                    title = (
                        title.replace("<span>", "")
                        .replace("<i>", "")
                        .replace("</span>", "")
                        .replace("</i>", "")
                        .ljust(40)
                    )
                    time = run.get("timestamp", "").ljust(18)

                    options = list()
                    options.append(
                        (
                            lambda col: getattr(col, "warn") + "Cancel".ljust(80),
                            "cancel",
                        )
                    )
                    if run.get("analysis", dict()).get("return", ""):
                        options.append(
                            (
                                lambda col: getattr(col, "element")
                                + "Console preview".ljust(80),
                                "results",
                            )
                        )
                        options.append(
                            (
                                lambda col: getattr(col, "element")
                                + "Show html".ljust(80),
                                "html",
                            )
                        )
                        options.append(
                            (
                                lambda col: getattr(col, "element")
                                + "New variation".ljust(80),
                                "variation",
                            )
                        )

                    select = Select(
                        options
                        + [
                            (
                                lambda col: getattr(col, "neutral")
                                + f'Info: {run.get("dataset", dict()).get("module", "No data loader")}'.ljust(
                                    80
                                ),
                                "data_loader",
                            ),
                            (
                                lambda col: getattr(col, "neutral")
                                + f'Info: {run.get("model", dict()).get("module", "No model loader")}'.ljust(
                                    80
                                ),
                                "model_loader",
                            ),
                            (
                                lambda col: getattr(col, "neutral")
                                + f'Info: {run.get("analysis", dict()).get("module", "No analysis method")}'.ljust(
                                    80
                                ),
                                "analysis_method",
                            ),
                            (
                                lambda col: getattr(col, "fail")
                                + "Edit (loses results)".ljust(80),
                                "edit",
                            ),
                            (
                                lambda col: getattr(col, "fail") + "Delete".ljust(80),
                                "delete",
                            ),
                        ],
                        self.base_state,
                        f"{colors.warn}{description} {time} {title}{colors.reset}",
                        self.base_state.runs,
                        -1,
                    )
                    self.next = select
                    self.next.next = self.next
                except Exception as e:
                    print(colors.fail + str(e).rjust(78) + colors.reset)
                save_all_runs("history.json", self.base_state.runs)


class Dashboard:
    def __init__(self, runs):
        self.selection = -1
        self.modifying = False
        self.modifying_pos = 0
        self.runs = runs
        self.next = self
        self.unhandled_button = None

    def show(state):
        runs = state.runs
        print("\x1b[2J\x1b[H")
        print(f"\033[1m{colors.warn}MAI-BIAS command line\033[0m")
        print("─" * 80)
        if state.selection < -2:
            state.selection = -2
        if state.selection >= len(runs):
            state.selection = len(runs) - 1
        coloring = colorsbg if -2 == state.selection else colors
        print(f"{coloring.fail}{'Exit'.ljust(80)}{colors.reset}")
        coloring = colorsbg if -1 == state.selection else colors
        print(f"{coloring.element}{'New run'.ljust(80)}{colors.reset}")
        for i, run in enumerate(runs):
            if i < state.selection - 5:
                continue
            if i > state.selection + 5 + max(0, 5 - state.selection):
                break
            description = run["description"]
            if not description:
                description = "..."
            description = description.ljust(20)
            title = extract_title(run)
            title = (
                title.replace("<span>", "")
                .replace("<i>", "")
                .replace("</span>", "")
                .replace("</i>", "")
                .ljust(40)
            )
            time = run.get("timestamp", "").ljust(18)
            coloring = colorsbg if i == state.selection else colors
            button_color = (
                (
                    coloring.fail
                    if "fail" in title or "bias" in title
                    else (
                        coloring.ok
                        if "report" in title
                        or "audit" in title
                        or "scan" in title
                        or "analysis" in title
                        or "explanation" in title
                        else coloring.neutral
                    )
                )
                if run["status"] == "completed"
                else coloring.warn
            )
            print(f"{button_color}{description} {time} {title}{colors.reset}")
        print("─" * 80)
        if state.modifying:
            state.modifying = False
            if state.selection == -2:
                state.next = None
            elif state.selection == -1:
                state.runs.append(
                    {"description": "", "timestamp": now(), "status": "in_progress"}
                )
                loaders = [
                    loader for loader, values in registry.dataset_loaders.items()
                ]
                state.next = Step(
                    loaders,
                    state,
                    colors.warn + "1/3 Dataset loader" + colors.reset,
                    state.runs[-1],
                )
            else:
                run = runs[state.selection]
                description = run["description"]
                if not description:
                    description = "..."
                description = description.ljust(20)
                title = extract_title(run)
                title = (
                    title.replace("<span>", "")
                    .replace("<i>", "")
                    .replace("</span>", "")
                    .replace("</i>", "")
                    .ljust(40)
                )
                time = run.get("timestamp", "").ljust(18)

                options = list()
                options.append(
                    (lambda col: getattr(col, "warn") + "Cancel".ljust(80), "cancel")
                )
                if run.get("analysis", dict()).get("return", ""):
                    options.append(
                        (
                            lambda col: getattr(col, "element")
                            + "Console preview".ljust(80),
                            "results",
                        )
                    )
                    options.append(
                        (
                            lambda col: getattr(col, "element") + "Show html".ljust(80),
                            "html",
                        )
                    )
                    options.append(
                        (
                            lambda col: getattr(col, "element")
                            + "New variation".ljust(80),
                            "variation",
                        )
                    )
                select = Select(
                    options
                    + [
                        (
                            lambda col: getattr(col, "neutral")
                            + f'Info: {run.get("dataset", dict()).get("module", "No data loader")}'.ljust(
                                80
                            ),
                            "data_loader",
                        ),
                        (
                            lambda col: getattr(col, "neutral")
                            + f'Info: {run.get("model", dict()).get("module", "No model loader")}'.ljust(
                                80
                            ),
                            "model_loader",
                        ),
                        (
                            lambda col: getattr(col, "neutral")
                            + f'Info: {run.get("analysis", dict()).get("module", "No analysis method")}'.ljust(
                                80
                            ),
                            "analysis_method",
                        ),
                        (
                            lambda col: getattr(col, "fail")
                            + "Edit (loses results)".ljust(80),
                            "edit",
                        ),
                        (
                            lambda col: getattr(col, "fail") + "Delete".ljust(80),
                            "delete",
                        ),
                    ],
                    state,
                    f"{colors.warn}{description} {time} {title}{colors.reset}",
                    state.runs,
                    state.selection,
                )
                state.next = select


if __name__ == "__main__":
    runs = load_all_runs("history.json")
    state = Dashboard(runs)
    state.show()
    print("Arrows navigate, page up/down is faster, [enter] selects".rjust(78))
    while state is not None:
        c = readchar.readkey()
        if c == readchar.key.UP:
            state.selection -= 1
        elif c == readchar.key.DOWN:
            state.selection += 1
        elif c == readchar.key.PAGE_UP:
            state.selection -= 5
        elif c == readchar.key.PAGE_DOWN:
            state.selection += 5
        elif c == readchar.key.LEFT:
            state.modifying_pos -= 1
        elif c == readchar.key.RIGHT:
            state.modifying_pos += 1
        elif c == readchar.key.ENTER:
            state.modifying = not state.modifying
        else:
            state.input_character = str(c)
        state.show()
        while state is not None and state != state.next:
            state = state.next
            if state is not None:
                state.show()
        if state is None:
            print("Exiting...".rjust(78))
        else:
            print("Arrows navigate, page up/down is faster, [enter] selects".rjust(78))

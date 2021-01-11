import os
import sys
import json
import pathlib
import inspect
import datetime
import collections.abc as collections
from omni_model import REPO_ROOT


class Logger:

    DEBUG = -1
    INFO = 0
    SUMMARY = 1
    WARNING = 2
    ERROR = 3
    SYSTEM = 4
    _instance = None
    indicator = {
        DEBUG: "D",
        INFO: "I",
        SUMMARY: "S",
        WARNING: "W",
        ERROR: "E",
        SYSTEM: "S",
    }

    class Colors:
        END = "\033[0m"
        BOLD = "\033[1m"
        UNDERLINE = "\033[4m"
        GREY = 30
        RED = 31
        GREEN = 32
        YELLOW = 33
        BLUE = 34
        PURPLE = 35
        SKY = 36
        WHITE = 37
        BACKGROUND = 10
        LIGHT = 60

        @staticmethod
        def code(value):
            return f"\033[{value}m"

    colorcode = {
        DEBUG: Colors.code(Colors.SKY),
        INFO: Colors.code(Colors.GREY + Colors.LIGHT),
        SUMMARY: Colors.code(Colors.BLUE + Colors.LIGHT),
        WARNING: Colors.code(Colors.YELLOW + Colors.LIGHT),
        ERROR: Colors.code(Colors.RED + Colors.LIGHT),
        SYSTEM: Colors.code(Colors.WHITE + Colors.LIGHT),
    }

    logging_root = REPO_ROOT / "logs"
    if not logging_root.exists():
        logging_root.mkdir(parents=True, exist_ok=True)
    write = None
    compactjson = True
    log_level = None  # log level
    path_json = None
    path_txt = None
    file_txt = None
    name = None
    perf_memory = {}
    values = {}
    max_lineno_width = 3

    def __new__(cls, write=False, name="logs"):
        cls.write = write
        if Logger._instance is None:
            Logger._instance = object.__new__(Logger)
            Logger._instance.set_level(Logger._instance.INFO)

            if write:
                Logger._instance.name = name
                Logger._instance.path_txt = cls.logging_root / f"{name}.txt"
                Logger._instance.file_txt = open(cls.logging_root / f"{name}.txt", "a+")
                Logger._instance.path_json = cls.logging_root / f"{name}.json"
                Logger._instance.reload_json()
            else:
                Logger._instance.log_message(
                    f"No logs files will be created - {write = } flag not selected.",
                    log_level=Logger.WARNING,
                )

        return Logger._instance

    def __call__(self, *args, **kwargs):
        return self.log_message(*args, **kwargs)

    def set_level(self, log_level):
        self.log_level = log_level

    def set_json_compact(self, is_compact):
        self.compactjson = is_compact

    def log_message(
        self,
        *message,
        log_level=INFO,
        break_line=True,
        raise_error=True,
    ):
        if log_level < self.log_level:
            return -1

        if self.write and not self.file_txt:
            raise Exception(
                f"CRITICAL: Log file not defined. Do you have write permissions for {self.logging_root}?"
            )

        filename = pathlib.Path(inspect.stack()[0].filename).relative_to(REPO_ROOT)
        message = " ".join([str(m) for m in list(message)])

        message_header = "[{} {:%Y-%m-%d %H:%M:%S}]".format(
            self.indicator[log_level], datetime.datetime.now()
        )
        message_locate = f"{filename}:"
        message_logger = f"{message_header} {message_locate} {message}"
        message_screen = f"{self.Colors.BOLD}{self.colorcode[log_level]}{message_header}{self.Colors.END} {message_locate} {message}"

        if break_line:
            print(message_screen)
            if self.write:
                self.file_txt.write("%s\n" % message_logger)
        else:
            print(message_screen, end="")
            sys.stdout.flush()
            if self.write:
                self.file_txt.write(message_logger)

        if self.write:
            self.file_txt.flush()
        if log_level == self.ERROR and raise_error:
            raise Exception(message)

    def log_value(self, name, value, hide=False, log_level=SUMMARY):
        if log_level < self.log_level:
            return -1

        if name not in self.values:
            self.values[name] = []
        self.values[name].append(value)

        if not hide:
            if type(value) == float:
                if int(value) == 0:
                    message = f"{name}: {value:.6f}"
                else:
                    message = f"{name}: {value:.2f}"
            else:
                message = f"{name}: {value}"
            self.log_message(message, log_level=log_level)

    def log_dict(
        self,
        group,
        dictionary,
        description="",
        hide=False,
        log_level=SUMMARY,
    ):
        if log_level < self.log_level:
            return -1

        if group not in self.perf_memory:
            self.perf_memory[group] = {}
        else:
            for key in self.perf_memory[group].keys():
                if key not in dictionary.keys():
                    self.log_message(
                        f'Key "{key}" not in the dictionary to be logged',
                        log_level=self.ERROR,
                    )
            for key in dictionary.keys():
                if key not in self.perf_memory[group].keys():
                    self.log_message(
                        f'Key "{key}" is unknown. New keys are not allowed',
                        log_level=self.ERROR,
                    )

        for key in dictionary.keys():
            if key in self.perf_memory[group]:
                self.perf_memory[group][key].extend([dictionary[key]])
            else:
                self.perf_memory[group][key] = [dictionary[key]]

        self.values[group] = self.perf_memory[group]
        if not hide:
            self.log_dict_message(group, dictionary, description, log_level)
        self.flush()

    def log_dict_message(self, group, dictionary, description="", log_level=SUMMARY):
        if log_level < self.log_level:
            return -1

        def print_subitem(prefix, subdictionary):
            for key, value in sorted(subdictionary.items()):
                message = prefix + key + ":"
                if not isinstance(value, collections.Mapping):
                    message += " " + str(value)
                self.log_message(message, log_level=log_level)
                if isinstance(value, collections.Mapping):
                    print_subitem(prefix + "  ", value)

        self.log_message(f"{group}: {description}", log_level=log_level)
        print_subitem("  ", dictionary)

    def reload_json(self):
        if self.path_json.is_file():
            try:
                with open(self.path_json, "r") as json_file:
                    self.values = json.load(json_file)
            except FileNotFoundError:
                self.log_message(
                    f"json log file can not be open: {self.path_json}",
                    log_level=self.WARNING,
                )

    def flush(self):
        if self.write:
            self.path_tmp = str(self.path_json) + ".tmp"
            try:
                with open(self.path_tmp, "w") as json_file:
                    if self.compactjson:
                        json.dump(self.values, json_file, separators=(",", ":"))
                    else:
                        json.dump(self.values, json_file, indent=4)
                if os.path.isfile(self.path_json):
                    os.remove(self.path_json)
                os.rename(self.path_tmp, self.path_json)
            except Exception as e:
                print(e)
                # TODO: Map what exception is this, and replace this "except Exception" for the real exception
                # we cannot keep this as is, it will eventually catch things we do not want to catch, like a keyboard interrupt
                raise e
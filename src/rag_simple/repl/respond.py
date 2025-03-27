import shlex
from argparse import ArgumentParser

from ..chatbot import Chatbot


class ErrorPrintParser(ArgumentParser):
    has_error = False

    def error(self, message):
        print(message)
        self.has_error = True


class Router:
    def __init__(self):
        self.map: dict[str, Command] = {}

    def command(self, name, desc=None):
        cmd = Command(name, desc=desc)
        self.map[name] = cmd
        return cmd

    def parse(self, text):
        args = shlex.split(text)
        if len(args) == 0:
            return None
        command_name = args[0]
        cmd = self.map.get(command_name, None)
        if cmd is None:
            print(f"unknown command {command_name}")
            return None
        args = cmd.parse(args[1:])
        args.command_name = command_name
        return args


class Argument:
    def __init__(self, *name_or_flags, nargs=None, default=None, **kwargs):
        self.name_or_flags = name_or_flags
        kwargs["nargs"] = nargs
        kwargs["default"] = default
        self.kwargs = kwargs


class Command:
    def __init__(self, name, desc=None):
        self.name = name
        self.desc = desc
        self.parser = ErrorPrintParser(prog=name, add_help=False, description=desc)
        self.parser.set_defaults(func=None, command_name="unknown", error=False, command=None)
        self.keywords = set()

    def __call__(self, func):
        self.parser.set_defaults(func=func)
        return func

    def parse(self, args):
        args = self.parser.parse_args(args)
        if self.parser.has_error:
            args.error = True
            self.parser.has_error = False
        args.command = self
        return args

    def add_arguments(self, *args: Argument):
        for one in args:
            action = self.parser.add_argument(
                *one.name_or_flags,
                **one.kwargs
            )
            self.keywords.add(action.dest)
        return self

    def print_help(self):
        self.parser.print_help()
        print()

    def filter_keywords(self, data: dict):
        result = dict()
        for key, value in data.items():
            if key in self.keywords:
                result[key] = value
        return result


class ChatResponder:
    def __init__(self, chatbot: Chatbot, default_limit=3):
        self.chatbot = chatbot
        self.default_limit = default_limit

    router = Router()

    @router.command("exit", desc="Exit the program.")
    def exit(self):
        pass

    @router.command("help", desc="Show help.").add_arguments(
        Argument("cmd", default=None, nargs="?",
                 help="Command Name. Use /help to list available commands."),
    )
    def help(self, cmd):
        if cmd is None:
            print("Available Commands:")
            for key, cmd in self.router.map.items():
                print(f"  /{key}\t\t", cmd.desc or "")
            print("Use `/help command` for details.")
            return
        result = self.router.map.get(cmd, None)
        if result is None:
            print(f"Unknown command {cmd}. Use /help to list them.")
            return
        result.print_help()
        print()

    @router.command("chat", desc="Chat with AI.")
    def chat(self, text: list[str], limit=None, retrieve=True):
        text = " ".join(text)
        if limit is None:
            limit = self.default_limit
        if retrieve:
            self.chatbot.retrieve(text, limit=limit).drain()
        self.chatbot.chat(text).print()

    def respond_to(self, user_input: str) -> bool:
        if not user_input.startswith("/"):
            self.chat([user_input], retrieve=True)
            return True
        user_input = user_input[1:]
        args = self.router.parse(user_input)
        if args is None:
            return True
        command = args.command
        command: Command
        if args.error:
            if command is not None:
                command.print_help()
            return True
        if args.command_name == "exit":
            return False
        func = args.func
        if args.func is None or command is None:
            return True
        args = command.filter_keywords(args.__dict__)
        func(self, **args)
        return True

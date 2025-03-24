import argparse
from .project import RAGProject


def cmd_new(args):
    project = RAGProject(args.path)
    return project.new_project()


def cmd_init(args):
    project = RAGProject(args.path)
    return project.init_project()


def cmd_build(args):
    project = RAGProject.find_possible_project()
    if project is None:
        print(f"Unable to find a rag project. Use environ ${RAGProject.Environ} to specify.")
        return -1
    dry_run = bool(args.dry_run)
    run_all = bool(args.all)
    project.build_db(dry_run=dry_run, run_all=run_all)


def main():
    parser = argparse.ArgumentParser(description="simple RAG project")

    def default_func(_args):
        parser.print_help()
        return -1

    parser.set_defaults(func=default_func)

    # sub-commands
    sub_parsers = parser.add_subparsers(description="")

    parser_new = sub_parsers.add_parser("new", help="make a new project")
    parser_new.set_defaults(func=cmd_new)
    parser_new.add_argument("path", default="rag_project", nargs="?",
                            help="project path, default is 'rag_project'")

    parser_init = sub_parsers.add_parser("init", help="init an existing directory")
    parser_init.set_defaults(func=cmd_init)
    parser_init.add_argument("path", help="project path")

    parser_new_doc = sub_parsers.add_parser("new_doc", help="make an example document file")
    parser_new_doc.add_argument("path", help="document path")
    parser_new_doc.add_argument("--force", "-f", action="count", help="overwrite when existing")
    parser_new_doc.set_defaults(func=lambda x: RAGProject.new_doc(x.path, bool(x.force)))

    parser_build = sub_parsers.add_parser("build", help="build chroma database")
    parser_build.add_argument("--dry-run", "-d", action="count", help="show files only")
    parser_build.add_argument("--all", "-a", action="count", help="rebuild all")
    parser_build.set_defaults(func=cmd_build)

    args = parser.parse_args()
    exit(args.func(args) or 0)

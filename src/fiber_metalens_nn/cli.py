import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--output", default="hardware.json")
    audit_parser.add_argument("--require-ready", action="store_true")
    audit_parser.add_argument("--phase", type=int)
    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    smoke_parser.add_argument("--output", required=True)
    guard_parser = sub.add_parser("guard")
    guard_parser.add_argument("--hardware", default="hardware.json")
    guard_parser.add_argument("--log", required=True)
    guard_parser.add_argument("--timeout", type=int, default=600)
    guard_parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "audit":
        from .audit import audit, preflight
        data = audit(args.output, phase=args.phase)
        errors = preflight(data)
        if args.require_ready and errors:
            print("PREFLIGHT STOP: " + "; ".join(errors), file=sys.stderr)
            return 2
        return 0
    if args.command == "smoke":
        from .smoke import smoke
        return smoke(args.backend, args.output)
    if args.command == "guard":
        from .guard import run_guarded
        command = args.args[1:] if args.args[:1] == ["--"] else args.args
        return run_guarded(command, args.hardware, args.log, timeout=args.timeout)


if __name__ == "__main__":
    sys.exit(main())

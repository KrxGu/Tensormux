"""CLI entry point: tensormux serve -c config.yaml"""

from __future__ import annotations

import argparse
import os
import sys


def cli() -> None:
    parser = argparse.ArgumentParser(prog="tensormux", description="Tensormux inference gateway")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="Start the gateway server")
    serve.add_argument("-c", "--config", default="config.yaml", help="Path to config YAML")
    serve.add_argument("--host", default=None, help="Override host")
    serve.add_argument("--port", type=int, default=None, help="Override port")

    args = parser.parse_args()

    if args.command == "serve":
        os.environ["TENSORMUX_CONFIG"] = args.config

        import uvicorn
        import yaml

        with open(args.config) as f:
            raw = yaml.safe_load(f) or {}
        gw = raw.get("gateway", {})
        host = args.host or gw.get("host", "0.0.0.0")
        port = args.port or gw.get("port", 8080)

        uvicorn.run("tensormux.api.main:app", host=host, port=port, log_level="info")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    cli()

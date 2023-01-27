"""
Contains click cli
Use with python cli.py --help
"""

import click
from controller.predict import predict
from controller.train import train


@click.group()
def cli():
    pass


cli.add_command(predict)
cli.add_command(train)


if __name__ == "__main__":
    # Start CLI
    cli()

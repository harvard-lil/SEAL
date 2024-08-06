""" Entry point: Initializes app context and groups commands. """

import click

from context import SEALContext
from compare import compare
from rewrite import rewrite
from taxonomy import taxonomy


@click.group()
@click.pass_context
def cli(ctx):
    ctx.obj = SEALContext()


cli.add_command(compare)
cli.add_command(rewrite)
cli.add_command(taxonomy)

if __name__ == "__main__":
    cli()

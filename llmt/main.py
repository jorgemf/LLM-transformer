import typer
from .config import setup
setup()
from .dataset import app as dataset_app
from .train import train
from .test import test

app = typer.Typer()
app.add_typer(dataset_app, name="dataset")
app.command()(train)
app.command()(test)

if __name__ == '__main__':
    app()

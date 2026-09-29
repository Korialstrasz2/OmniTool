"""Run the local workspace with `python app.py`. No debug server is exposed."""
from omnitool_core.web import create_app, main

app = create_app()

if __name__ == "__main__":
    main(app)

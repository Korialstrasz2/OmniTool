"""Run the local workspace with `python app.py`. No debug server is exposed."""
from omnitool_core.web import create_app as create_base_app, main
from omnitool_core.maintenance_web import register


def create_app(*args, **kwargs):
    """Canonical full-workspace factory, including repaired maintenance pages."""
    application = create_base_app(*args, **kwargs)
    register(application)
    return application


app = create_app()

if __name__ == "__main__":
    main(app)

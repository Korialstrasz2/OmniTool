"""Run the local workspace with `python app.py`. No debug server is exposed."""
from omnitool_core.web import create_app as create_base_app, main
from omnitool_core.maintenance_web import register
from omnitool_core.file_web import register as register_file_workbench
from omnitool_core.content_web import register as register_content_workbench


def create_app(*args, **kwargs):
    """Canonical factory for every registered workspace page."""
    application = create_base_app(*args, **kwargs)
    register(application)
    register_file_workbench(application)
    register_content_workbench(application)
    return application


app = create_app()

if __name__ == "__main__":
    main(app)

"""Local-only workspace: catalog, typed launches, jobs, and opaque vaults."""
from __future__ import annotations

import atexit
import hmac
import io
import os
import secrets
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, abort, flash, jsonify, redirect, render_template, request, send_file, session, url_for

from .catalog import Catalog, arguments, public_spec, validate
from .jobs import Jobs
from .locks import Locked, Vaults
from .vault import VaultError


def create_app(base: Path | None = None, access_token: str | None = None, ttl: int = 600) -> Flask:
    base = (base or Path(__file__).resolve().parents[1]).resolve()
    app = Flask(__name__, template_folder=str(base / "templates"), static_folder=str(base / "static"))
    token = access_token or os.environ.get("OMNITOOL_ACCESS_TOKEN") or secrets.token_urlsafe(32)
    if len(token) < 24:
        raise ValueError("OMNITOOL_ACCESS_TOKEN must contain at least 24 characters")
    app.config.update(SECRET_KEY=secrets.token_bytes(32), MAX_CONTENT_LENGTH=64 * 1024,
                      SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Strict",
                      SESSION_COOKIE_NAME="omnitool_session", SESSION_COOKIE_PATH="/",
                      PERMANENT_SESSION_LIFETIME=8 * 60 * 60,
                      TRUSTED_HOSTS=["127.0.0.1", "localhost", "[::1]"])
    catalog, jobs = Catalog(base), Jobs()
    vaults = Vaults(base / "vaults", jobs, ttl=ttl)
    app.extensions.update(omnitool_catalog=catalog, omnitool_jobs=jobs, omnitool_vaults=vaults,
                          omnitool_access_token=token)
    limits: dict[str, list[float]] = {}
    rate_lock = threading.Lock()

    def rate_limit(key: str, maximum: int, window: int = 60) -> None:
        with rate_lock:
            now = time.monotonic()
            hits = [t for t in limits.get(key, []) if now - t < window]
            limits[key] = hits
            if len(hits) >= maximum:
                abort(429, "Too many attempts; try again later")
            hits.append(now)

    def owner() -> str:
        return session["sid"]

    def find_tool(id: str) -> dict:
        tool = next((t for t in catalog.load() if t["id"] == id), None)
        if tool is None:
            abort(404)
        return tool

    @app.before_request
    def protect():
        if request.remote_addr not in {"127.0.0.1", "::1"}:
            abort(403, "OmniTool is a local-only application")
        parsed = urlsplit(request.host_url)
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            abort(400, "Invalid host")
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("Origin")
            if origin and origin != request.host_url.rstrip("/"):
                abort(403, "Cross-origin request denied")
            expected = session.get("csrf", "")
            supplied = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token", "")
            if not expected or not isinstance(supplied, str) or not hmac.compare_digest(expected.encode("utf-8"), supplied.encode("utf-8")):
                abort(403, "CSRF validation failed")
        if request.endpoint not in {"login", "static"} and not session.get("sid"):
            if request.path.startswith("/api/"):
                abort(401, "Sign in locally first")
            return redirect(url_for("login"))

    @app.after_request
    def headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response

    @app.context_processor
    def common():
        session.setdefault("csrf", secrets.token_urlsafe(32))
        return {"csrf_token": session["csrf"], "signed_in": bool(session.get("sid"))}

    @app.errorhandler(VaultError)
    @app.errorhandler(ValueError)
    def invalid(exc):
        if request.path.startswith("/api/"):
            return jsonify(error=str(exc)), 400
        return render_template("error.html", message=str(exc)), 400

    @app.errorhandler(Locked)
    def locked(exc):
        return jsonify(error=str(exc)), 423

    @app.errorhandler(KeyError)
    def missing(_exc):
        return jsonify(error="Not found"), 404

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            rate_limit("login", 10)
            supplied = request.form.get("access_token", "")
            if not hmac.compare_digest(supplied.encode("utf-8"), token.encode("utf-8")):
                return render_template("login.html", error="Incorrect local access code"), 401
            session.clear()
            session.update(sid=secrets.token_urlsafe(32), csrf=secrets.token_urlsafe(32))
            session.permanent = True
            return redirect(url_for("index"))
        return render_template("login.html")

    @app.post("/logout")
    def logout():
        vaults.lock_all(owner())
        session.clear()
        return redirect(url_for("login"))

    @app.get("/")
    def index():
        return render_template("index.html", mode="catalog")

    @app.get("/jobs")
    def job_page():
        return render_template("index.html", mode="jobs")

    @app.get("/vaults")
    def vault_page():
        return render_template("index.html", mode="vaults")

    @app.get("/tool/<tool_id>")
    def tool_detail(tool_id):
        tool = find_tool(tool_id)
        if tool["kind"] == "page":
            return redirect(tool["page"])
        return render_template("tool.html", tool=tool)

    @app.post("/tool/<tool_id>/run")
    def run_tool(tool_id):
        tool = find_tool(tool_id)
        if tool["kind"] != "python" or tool["availability"] != "ready":
            raise ValueError("Tool is disabled or requires setup")
        if tool.get("risk") == "writes-files" and request.form.get("confirm") != "yes":
            raise ValueError("Confirm that you understand this tool can change files")
        values = {p["name"]: (p["name"] in request.form if p.get("type") == "boolean" else request.form.get(p["name"], ""))
                  for p in tool["parameters"]}
        argv = arguments(tool, values)
        root = tool["_root"]
        entrypoint = (root / tool["entrypoint"]).resolve()
        # Re-check containment at launch, rather than relying on cached UI state.
        if not entrypoint.is_relative_to(root.resolve()):
            abort(403)
        cwd = (root / tool.get("working_directory", ".")).resolve()
        if not cwd.is_relative_to(base) or not cwd.is_dir():
            raise ValueError("Invalid working directory")
        jobs.submit(owner(), tool["name"], [sys.executable, "-B", str(entrypoint), *argv], cwd)
        flash("Job queued. Progress and output are available below.", "success")
        return redirect(url_for("job_page"))

    @app.post("/api/catalog")
    def catalog_api():
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            raise ValueError("Expected an object")
        query = str(data.get("query", ""))[:256].casefold()
        folder, status = str(data.get("folder", "")), str(data.get("status", ""))
        page, size = max(1, int(data.get("page", 1))), max(1, min(100, int(data.get("size", 24))))
        tools = catalog.load()
        folders = sorted({t["folder"] for t in tools}, key=str.casefold)
        filtered = [t for t in tools if (not folder or t["folder"] == folder)
                    and (not status or t["availability"] == status)
                    and (not query or query in " ".join([t["name"], t["description"], t["folder"], *t["tags"]]).casefold())]
        pinned = data.get("pinned", [])
        if not isinstance(pinned, list) or len(pinned) > 5000:
            raise ValueError("Invalid favorites")
        if data.get("favorites_only"):
            filtered = [t for t in filtered if t["id"] in pinned]
        filtered.sort(key=lambda t: (t["id"] not in pinned, t["name"].casefold()))
        pages = max(1, (len(filtered) + size - 1) // size)
        page = min(page, pages)
        return jsonify(items=[public_spec(t) for t in filtered[(page - 1) * size:page * size]],
                       total=len(filtered), total_catalog=len(tools), folders=folders,
                       page=page, pages=pages, errors=catalog.errors)

    @app.get("/api/jobs")
    def job_list():
        visible = []
        for job in jobs.list(owner()):
            if job["vault"]:
                try:
                    vaults.get(owner(), job["vault"])
                except Locked:
                    continue
            visible.append(job)
        return jsonify(items=visible)

    @app.post("/api/jobs/<job_id>/stop")
    def stop_job(job_id):
        jobs.stop(owner(), job_id)
        return jsonify(ok=True)

    @app.get("/api/vaults")
    def vault_list():
        return jsonify(items=vaults.list(owner()))

    @app.post("/api/vaults/<vault_id>/unlock")
    def unlock_vault(vault_id):
        rate_limit("unlock", 6)
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            raise ValueError("Expected an object")
        vaults.unlock(owner(), vault_id, data.get("passphrase", ""))
        return jsonify(ok=True)

    @app.post("/api/vaults/<vault_id>/lock")
    def lock_vault(vault_id):
        vaults.lock(owner(), vault_id)
        return jsonify(ok=True)

    @app.post("/api/vaults/lock-all")
    def lock_all():
        vaults.lock_all(owner())
        return jsonify(ok=True)

    @app.get("/api/vaults/<vault_id>/contents")
    def vault_contents(vault_id):
        item = vaults.get(owner(), vault_id)
        tools = [dict(public_spec(t), index=i, availability=Catalog.availability(t)) for i, t in enumerate(item["meta"]["tools"])]
        page = max(1, int(request.args.get("page", 1)))
        filenames = sorted(item["files"])
        return jsonify(name=item["meta"]["name"], kind=item["meta"]["kind"], tools=tools[(page - 1) * 24:page * 24],
                       tool_pages=max(1, (len(tools) + 23) // 24),
                       files=[{"index": i, "name": filenames[i]} for i in range((page - 1) * 100, min(page * 100, len(filenames)))],
                       file_pages=max(1, (len(filenames) + 99) // 100))

    @app.get("/api/vaults/<vault_id>/files/<int:file_index>")
    def vault_file(vault_id, file_index):
        item = vaults.get(owner(), vault_id)
        names = sorted(item["files"])
        if file_index >= len(names):
            abort(404)
        name = names[file_index]
        # Never execute/render decrypted HTML or JS in the workspace origin.
        return send_file(io.BytesIO(item["files"][name]), mimetype="application/octet-stream", as_attachment=True,
                         download_name=Path(name).name, max_age=0)

    @app.post("/api/vaults/<vault_id>/tools/<int:index>/run")
    def run_private(vault_id, index):
        item = vaults.get(owner(), vault_id)
        if index >= len(item["meta"]["tools"]):
            abort(404)
        tool = item["meta"]["tools"][index]
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            raise ValueError("Expected an object")
        if data.get("confirm") is not True:
            raise ValueError("Explicitly confirm execution of trusted local code")
        if Catalog.availability(tool) != "ready":
            raise ValueError("Locked tool requires setup or is disabled")
        job_id = vaults.run(owner(), vault_id, index, arguments(tool, data.get("values", {})))
        return jsonify(job_id=job_id), 202

    @app.get("/csv-editor")
    def csv_editor():
        return render_template("csv_editor.html")

    # Old direct links remain useful without retaining old launch handlers.
    @app.get("/media-harvester")
    def media_harvester():
        return redirect(url_for("tool_detail", tool_id="media-harvester-manager"))

    @app.get("/lyrics-embedder")
    def lyrics_embedder():
        return redirect(url_for("tool_detail", tool_id="lyrics-embedder-manager"))

    def shutdown():
        try:
            vaults.shutdown()
        finally:
            jobs.shutdown()
    app.extensions["omnitool_shutdown"] = shutdown
    atexit.register(shutdown)
    return app


def main(app: Flask | None = None):
    from waitress import serve
    application = app or create_app()
    print("OmniTool: http://127.0.0.1:5000", flush=True)
    print("Local access code (not your vault passphrase): " + application.extensions["omnitool_access_token"], flush=True)
    print("Keep this terminal private. Vaults start locked. Ctrl+C stops the workspace.", flush=True)
    serve(application, host="127.0.0.1", port=5000, threads=6, max_request_body_size=64 * 1024)

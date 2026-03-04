"""Production launcher: Flask serves built React frontend + API."""

import sys
import os
import argparse
import webbrowser
import threading

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.app import create_app, socketio
from flask import send_from_directory


def main():
    parser = argparse.ArgumentParser(description="RAFTcorr production server")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (default: 127.0.0.1, use 0.0.0.0 for Colab)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port number (default: 5000)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open a browser window on startup")
    args = parser.parse_args()

    app = create_app()

    # Serve the built React frontend
    frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")

    if not os.path.isdir(frontend_dist):
        print(f"ERROR: Frontend build not found at {frontend_dist}")
        print("Run 'cd frontend && npm run build' first.")
        sys.exit(1)

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_frontend(path):
        # Serve static files if they exist, otherwise serve index.html (SPA fallback)
        full_path = os.path.join(frontend_dist, path)
        if path and os.path.isfile(full_path):
            return send_from_directory(frontend_dist, path)
        return send_from_directory(frontend_dist, "index.html")

    host = args.host
    port = args.port
    url = f"http://{host}:{port}"

    if not args.no_browser:
        def open_browser():
            import time
            time.sleep(1.5)
            webbrowser.open(url)
        threading.Thread(target=open_browser, daemon=True).start()

    print(f"Starting RAFTcorr production server at {url}")
    socketio.run(app, host=host, port=port, debug=False, use_reloader=False,
                 allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()

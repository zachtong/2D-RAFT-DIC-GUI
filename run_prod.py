"""Production launcher: Flask serves built React frontend + API."""

import sys
import os
import webbrowser
import threading

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.app import create_app, socketio
from flask import send_from_directory


def main():
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

    host = "127.0.0.1"
    port = 5000
    url = f"http://{host}:{port}"

    # Open browser after a short delay
    def open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    print(f"Starting RAFTcorr production server at {url}")
    socketio.run(app, host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()

"""Serve the browser demo from the project root."""

from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

if __name__ == "__main__":
    print("Open http://127.0.0.1:8000/demo/")
    ThreadingHTTPServer(("127.0.0.1", 8000), SimpleHTTPRequestHandler).serve_forever()


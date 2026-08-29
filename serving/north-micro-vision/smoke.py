"""Send one page image to the server and print what comes back.

    python smoke.py http://localhost:8086 page.png ["Convert this page to markdown."]
"""

import base64
import json
import sys
import time
import urllib.request

endpoint, image = sys.argv[1], sys.argv[2]
prompt = sys.argv[3] if len(sys.argv) > 3 else "Convert this page to markdown."
with open(image, "rb") as handle:
    data = base64.b64encode(handle.read()).decode()
payload = {
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}},
        ],
    }],
    "max_tokens": 4096,
    "temperature": 0,
}
request = urllib.request.Request(
    endpoint.rstrip("/") + "/v1/chat/completions",
    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
)
started = time.monotonic()
with urllib.request.urlopen(request, timeout=3600) as response:
    body = json.load(response)
print(json.dumps(body["usage"]), f"wall={time.monotonic() - started:.1f}s", file=sys.stderr)
print(body["choices"][0]["message"]["content"])

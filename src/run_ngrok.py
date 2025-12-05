from pyngrok import ngrok
import time

# Open a HTTP tunnel on the default port 7860
public_url = ngrok.connect(7860).public_url
print(f"ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:7860\"")

# Keep the process alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Closing tunnel...")
    ngrok.kill()

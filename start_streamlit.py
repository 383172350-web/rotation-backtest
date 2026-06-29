import subprocess
import os
import time
import urllib.request

os.chdir(r'C:\Users\Administrator\Documents\kimi\workspace\rotation-web')

# 启动 Streamlit
p = subprocess.Popen(
    [r'C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe',
     '-m', 'streamlit', 'run', 'streamlit_app.py'],
    stdout=open('streamlit.log', 'w'),
    stderr=subprocess.STDOUT,
    creationflags=0x00000008
)

print(f"PID: {p.pid}")
with open('streamlit.pid', 'w') as f:
    f.write(str(p.pid))

# 等待启动
time.sleep(6)
for port in [8501, 8502, 8503, 8504, 8505, 8506]:
    try:
        r = urllib.request.urlopen(f'http://localhost:{port}/_stcore/health', timeout=3)
        print(f"OK port={port} status={r.read().decode()}")
        with open('streamlit.port', 'w') as f:
            f.write(str(port))
        break
    except Exception as e:
        pass
else:
    print("Failed")

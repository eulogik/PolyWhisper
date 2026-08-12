"""Launch a command fully detached (new session, survives parent/session death).

Usage:
  python run_detached.py <logfile> <cmd...>

Prints/records the child PID. The child runs in its own process session
(start_new_session=True) so killing the parent shell/agent session cannot
take it down — the failure mode that killed today's runs.
"""

import subprocess
import sys
from datetime import datetime

logfile, cmd = sys.argv[1], sys.argv[2:]
p = subprocess.Popen(cmd, stdout=open(logfile, "a"), stderr=subprocess.STDOUT,
                     start_new_session=True)
with open(logfile, "a") as f:
    f.write(f"\n[detached {datetime.now().isoformat()}] pid={p.pid} cmd={' '.join(cmd)}\n")
print(f"PID: {p.pid}")
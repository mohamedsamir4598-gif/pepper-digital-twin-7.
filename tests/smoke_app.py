"""Headless smoke test of app.py with a fake `streamlit` (the real one is not needed).
It exercises every tab and every button branch. It cannot catch real-Streamlit API mistakes."""
import os, sys, tempfile, types, runpy
import pandas as pd

class Rerun(Exception): pass
class SS(dict):
    __getattr__ = dict.get
    def __setattr__(s, k, v): s[k] = v
class Ctx:
    def __enter__(s): return s
    def __exit__(s, *a): return False
    def progress(s, *a, **k): pass

class W:
    def __init__(self, st): self.st = st
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, name):          # any display call is a no-op
        return lambda *a, **k: None
    def radio(self, label, options, index=0, **k):
        return self.st.mode if label == "Data mode" else options[index]
    def selectbox(self, label, options, index=0, **k): return options[index]
    def multiselect(self, label, options, default=None, **k): return default or []
    def number_input(self, label, *a, value=None, **k):
        if value is not None: return value
        return a[2] if len(a) > 2 else a[0]
    def text_input(self, label, value="", **k): return value
    def slider(self, label, a, b, value=None, **k): return value if value is not None else a
    def checkbox(self, label, value=False, **k): return value
    def button(self, label, **k): return label in self.st.clicks
    def download_button(self, *a, **k): return False
    def file_uploader(self, *a, **k): return None
    def data_editor(self, df, **k): return df
    def columns(self, spec): return [W(self.st) for _ in range(spec if isinstance(spec, int) else len(spec))]
    def tabs(self, labels): return [Ctx() for _ in labels]
    def expander(self, *a, **k): return Ctx()
    def spinner(self, *a, **k): return Ctx()
    def progress(self, *a, **k): return Ctx()
    def rerun(self): raise Rerun()

class FakeSt(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = SS(); self.mode = "Manual entry"; self.clicks = set()
        self.w = W(self); self.sidebar = self.w
    def __getattr__(self, name): return getattr(self.w, name)

fake = FakeSt(); sys.modules["streamlit"] = fake
tmp = tempfile.TemporaryDirectory(); os.environ["PEPPER_DB_PATH"] = os.path.join(tmp.name, "t.sqlite3")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")

def run(mode, clicks, label):
    fake.mode, fake.clicks = mode, set(clicks)
    for _ in range(3):
        try:
            runpy.run_path(APP, run_name="__main__"); break
        except Rerun:
            fake.clicks = set()
    print("ok:", label)

run("Manual entry", [], "empty database, manual mode")
run("Virtual demo (synthetic)", ["Generate virtual experiment"], "generate virtual experiment")
for m in ["Manual entry", "CSV import", "Live sensors (API)", "Virtual demo (synthetic)"]:
    run(m, ["Save manual readings"] if m == "Manual entry" else [], f"render with data, {m}")
run("Virtual demo (synthetic)", ["Compute recommendation", "Approve", "Project yield and water need",
    "Optimise season plan", "Yield–water frontier", "Calibrate and run leave-one-replicate-out validation",
    "Run virtual trial", "Build twin scenario library and surrogate"], "all analysis buttons")
run("Virtual demo (synthetic)", ["Use these parameters in the platform"], "adopt calibrated parameters")
run("Virtual demo (synthetic)", [], "render with calibrated parameters")
run("Virtual demo (synthetic)", ["Save site setup", "Save units", "Create units"], "site setup / units")
tmp.cleanup()
print("SMOKE TEST PASSED")

"""
Project Refactor Script — Moves files and fixes imports.
CRITICAL FIX: Uses dirs[:] pruning to skip venv/node_modules/.git during os.walk.
"""
import os
import shutil
import re
from pathlib import Path

BASE_DIR = os.getcwd()

# Directories to NEVER walk into
SKIP_DIRS = {'venv', '.git', '__pycache__', 'node_modules', '.gemini'}

def safe_move(src, dst):
    src_path = os.path.join(BASE_DIR, src)
    dst_path = os.path.join(BASE_DIR, dst)
    
    if not os.path.exists(src_path):
        print(f"  Skip (not found): {src}")
        return
    
    print(f"  {src} → {dst}")
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    
    if os.path.isdir(src_path) and os.path.isdir(dst_path):
        # Merge directories
        for item in os.listdir(src_path):
            s = os.path.join(src_path, item)
            d = os.path.join(dst_path, item)
            if os.path.exists(d):
                if os.path.isdir(d):
                    shutil.rmtree(d)
                else:
                    os.remove(d)
            shutil.move(s, d)
        try:
            os.rmdir(src_path)
        except OSError:
            pass
    else:
        if os.path.exists(dst_path):
            if os.path.isdir(dst_path):
                shutil.rmtree(dst_path)
            else:
                os.remove(dst_path)
        shutil.move(src_path, dst_path)

# ============================================================
# STEP 1: CREATE DIRECTORY STRUCTURE
# ============================================================
print("=== STEP 1: Creating directory structure ===")
folders = [
    "v1/api", "v1/engine", "v1/web",
    "v2/api", "v2/engine", "v2/web", "v2/services",
    "shared/database", "shared/providers", "shared/web_common",
    "shared/config", "shared/utils", "shared/models", "shared/logic",
    "shared/services",
    "scripts/legacy", "data/stock_data", "data/reports",
    "docs/notebooks", "tests"
]
for f in folders:
    os.makedirs(os.path.join(BASE_DIR, f), exist_ok=True)
print("  Done.")

# ============================================================
# STEP 2: MOVE V2 ENGINE
# ============================================================
print("\n=== STEP 2: Moving V2 components ===")
safe_move("api_v2.py", "v2/api/routes.py")
safe_move("src/v2", "v2/engine")
safe_move("web/v2", "v2/web")

# ============================================================
# STEP 3: MOVE SHARED INFRASTRUCTURE
# ============================================================
print("\n=== STEP 3: Moving shared infrastructure ===")
safe_move("src/database", "shared/database")
safe_move("src/data", "shared/providers")
safe_move("src/services", "shared/services")
safe_move("src/strategies", "shared/logic/strategies")
safe_move("src/indicators", "shared/logic/indicators")
safe_move("src/ml", "shared/logic/ml")
safe_move("src/sentiment", "shared/logic/sentiment")
safe_move("src/risk", "shared/logic/risk")
safe_move("src/alerts", "shared/logic/alerts")
safe_move("src/strategy", "shared/logic/strategy")
safe_move("src/reports", "data/reports")
safe_move("reports", "data/reports")
safe_move("src/main.py", "shared/logic/legacy_main.py")
safe_move("src/config", "shared/config")
safe_move("web/common", "shared/web_common")
safe_move("core", "shared/core_legacy")
safe_move("config", "shared/config")

# ============================================================
# STEP 4: MOVE V1 ENGINE
# ============================================================
print("\n=== STEP 4: Moving V1 components ===")
safe_move("api_v1.py", "v1/api/routes.py")
safe_move("src/execution", "v1/engine/execution")
safe_move("src/engine", "v1/engine/core")
safe_move("web/v1", "v1/web")
safe_move("godbot", "v1/engine/godbot")

# ============================================================
# STEP 5: ROOT CLEANUP
# ============================================================
print("\n=== STEP 5: Cleaning up root directory ===")
legacy_scripts = [
    "working_new_bot.py", "run_bot.py", "paper_trade.py",
    "intraday_simulation.py", "discord_bot.py", "main.py", "demo.py"
]
for script in legacy_scripts:
    safe_move(script, f"scripts/legacy/{script}")

notebooks = [
    "readme copy.ipynb", "readme.ipynb", "nifty50_trade_test.ipynb",
    "trade_test.ipynb"
]
for nb in notebooks:
    safe_move(nb, f"docs/notebooks/{nb}")

data_files = [
    "paper_account.json", "trade_journal.json", "posted_headlines.json",
    "sent_headlines.txt", "bot_configs.json.bak"
]
for df_name in data_files:
    safe_move(df_name, f"data/{df_name}")

safe_move("stock_data", "data/stock_data")
safe_move("test_1lk_trade_msft", "data/test_1lk_trade_msft")

# Move .log/.db/.csv files from root to data/
for file in os.listdir(BASE_DIR):
    fp = os.path.join(BASE_DIR, file)
    if os.path.isfile(fp) and (file.endswith(".log") or file.endswith(".db") or file.endswith(".csv")):
        safe_move(file, f"data/{file}")

# ============================================================
# STEP 6: FIX IMPORTS (with proper venv pruning)
# ============================================================
print("\n=== STEP 6: Fixing imports recursively ===")

# Build patterns dynamically to avoid self-replacement
_S = "src"
_C = "config"
_G = "godbot"
_A1 = "api_v1"
_A2 = "api_v2"

IMPORT_MAP = [
    (re.compile(rf"from {_S}\.v2"),                "from v2.engine"),
    (re.compile(rf"import {_S}\.v2"),              "import v2.engine"),
    (re.compile(rf"from {_S}\.execution"),         "from v1.engine.execution"),
    (re.compile(rf"from {_S}\.engine"),            "from v1.engine.core"),
    (re.compile(rf"from {_S}\.database"),          "from shared.database"),
    (re.compile(rf"from {_S}\.data"),              "from shared.providers"),
    (re.compile(rf"from {_S}\.services"),          "from shared.services"),
    (re.compile(rf"from {_S}\.strategies"),        "from shared.logic.strategies"),
    (re.compile(rf"from {_S}\.indicators"),        "from shared.logic.indicators"),
    (re.compile(rf"from {_S}\.ml"),                "from shared.logic.ml"),
    (re.compile(rf"from {_S}\.sentiment"),         "from shared.logic.sentiment"),
    (re.compile(rf"from {_S}\.risk"),              "from shared.logic.risk"),
    (re.compile(rf"from {_S}\.alerts"),            "from shared.logic.alerts"),
    (re.compile(rf"from {_S}\.strategy"),          "from shared.logic.strategy"),
    (re.compile(rf"from {_C}\.settings"),          "from shared.config.settings"),
    (re.compile(rf"import {_C}\.settings"),        "import shared.config.settings"),
    (re.compile(rf"from {_G}"),                    "from v1.engine.godbot"),
    (re.compile(rf"from {_A1}"),                   "from v1.api.routes"),
    (re.compile(rf"from {_A2}"),                   "from v2.api.routes"),
    # Fix static file serving paths in api_server.py
    (re.compile(r"send_from_directory\('web/v1'"), "send_from_directory('v1/web'"),
    (re.compile(r"send_from_directory\('web/v2'"), "send_from_directory('v2/web'"),
    (re.compile(r"send_from_directory\('web/common'"), "send_from_directory('shared/web_common'"),
]

updated_count = 0
scanned_count = 0

for root, dirs, files in os.walk(BASE_DIR, topdown=True):
    # *** CRITICAL: Prune dirs in-place to prevent descent into venv ***
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    
    for file in files:
        if not file.endswith((".py", ".html", ".js")):
            continue
        
        path = os.path.join(root, file)
        if os.path.abspath(path) == os.path.abspath(__file__):
            continue
        
        scanned_count += 1
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            original = content
            for pattern, replacement in IMPORT_MAP:
                content = pattern.sub(replacement, content)
            
            # Static path replacements
            content = content.replace("static/v1/", "v1/web/")
            content = content.replace("static/v2/", "v2/web/")
            content = content.replace("static/common/", "shared/web_common/")
            
            if content != original:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                updated_count += 1
                print(f"  Updated: {os.path.relpath(path, BASE_DIR)}")
        except Exception as e:
            print(f"  Error: {os.path.relpath(path, BASE_DIR)}: {e}")

print(f"\n  Scanned {scanned_count} files, updated {updated_count}.")
print("\n✅ Project Refactor Completed Successfully!")

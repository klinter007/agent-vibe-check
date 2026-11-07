"""
Cross-platform Cursor installation paths.

Handles macOS, Windows, and Linux.
"""

import os
import platform
from pathlib import Path


def get_cursor_projects_dir() -> Path:
    """
    Get the Cursor projects directory.
    
    Returns:
        Path to ~/.cursor/projects/ (or equivalent on other platforms)
    """
    # This is the same across all platforms
    return Path.home() / ".cursor" / "projects"


def get_cursor_user_data_dir() -> Path:
    """
    Get the Cursor user data directory.
    
    Returns:
        Platform-specific path to Cursor's user data
        
    Locations:
        macOS:   ~/Library/Application Support/Cursor/User/
        Windows: %APPDATA%/Cursor/User/
        Linux:   ~/.config/Cursor/User/
    """
    system = platform.system()
    
    if system == "Darwin":  # macOS
        return Path.home() / "Library" / "Application Support" / "Cursor" / "User"
    elif system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        if not appdata:
            appdata = Path.home() / "AppData" / "Roaming"
        return Path(appdata) / "Cursor" / "User"
    else:  # Linux and others
        return Path.home() / ".config" / "Cursor" / "User"


def get_workspace_storage_dir() -> Path:
    """
    Get the workspace storage directory containing per-project SQLite databases.
    
    Returns:
        Path to workspaceStorage directory
    """
    return get_cursor_user_data_dir() / "workspaceStorage"


def get_global_storage_dir() -> Path:
    """
    Get the global storage directory containing the main SQLite database.
    
    Returns:
        Path to globalStorage directory
    """
    return get_cursor_user_data_dir() / "globalStorage"


def get_global_state_db() -> Path:
    """
    Get the global state.vscdb database path.
    
    This is the main database containing all conversation history.
    
    Returns:
        Path to state.vscdb
    """
    return get_global_storage_dir() / "state.vscdb"


# User-owned output ------------------------------------------------------------

def get_vibe_check_home() -> Path:
    """
    Directory where Vibe Check writes all user-owned output (extractions,
    reports, atlas, cache, logs). Lives outside any repo clone so the tool
    is share-safe.

    Resolution order:
      1. $VIBE_CHECK_HOME env var ($CURSOR_VIBE_CHECK_HOME still accepted)
      2. ~/.vibe-check
      3. ~/.cursor-vibe-check, if it exists and ~/.vibe-check does not
         (output folder of versions before the rename)
    """
    override = os.environ.get("VIBE_CHECK_HOME") or os.environ.get("CURSOR_VIBE_CHECK_HOME")
    if override:
        base = Path(override).expanduser()
    else:
        base = Path.home() / ".vibe-check"
        legacy = Path.home() / ".cursor-vibe-check"
        if not base.exists() and legacy.is_dir():
            base = legacy
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_extractions_dir() -> Path:
    """Root folder for all project extractions."""
    d = get_vibe_check_home() / "extractions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_cache_dir() -> Path:
    """Diff-mode / content-hash cache."""
    d = get_vibe_check_home() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_atlas_dir() -> Path:
    """Location of the meta-report / Project Atlas."""
    d = get_vibe_check_home() / "atlas"
    d.mkdir(parents=True, exist_ok=True)
    return d


def check_cursor_installation() -> dict:
    """
    Check if Cursor is installed and what data is available.
    
    Returns:
        Dict with installation status and paths
    """
    result = {
        "installed": False,
        "projects_dir": None,
        "projects_count": 0,           # all folders under ~/.cursor/projects/
        "projects_with_history": 0,    # JSONL agent-transcripts present
        "projects_with_composer_db": 0,  # workspaces with composer sessions in state.vscdb
        "unique_projects_total": 0,    # combined unique projects across both sources
        "user_data_dir": None,
        "workspace_dbs": 0,
        "global_db": None,
        "global_db_size_mb": 0,
    }

    # Check projects directory. Note: Cursor creates a folder here for every
    # path you've opened in the IDE, even if you never chatted. The interesting
    # number for this tool is "folders that have any agent-transcripts".
    projects_dir = get_cursor_projects_dir()
    if projects_dir.exists():
        result["projects_dir"] = str(projects_dir)
        result["installed"] = True
        all_count = 0
        with_history = 0
        for p in projects_dir.iterdir():
            if not p.is_dir():
                continue
            all_count += 1
            t = p / "agent-transcripts"
            if t.is_dir():
                try:
                    if any(t.iterdir()):
                        with_history += 1
                except OSError:
                    pass
        result["projects_count"] = all_count
        result["projects_with_history"] = with_history
    
    # Check user data directory
    user_data_dir = get_cursor_user_data_dir()
    if user_data_dir.exists():
        result["user_data_dir"] = str(user_data_dir)
        result["installed"] = True
    
    # Check workspace storage
    ws_dir = get_workspace_storage_dir()
    if ws_dir.exists():
        result["workspace_dbs"] = len([d for d in ws_dir.iterdir() if d.is_dir()])
    
    # Check global database
    global_db = get_global_state_db()
    if global_db.exists():
        result["global_db"] = str(global_db)
        result["global_db_size_mb"] = global_db.stat().st_size / (1024 * 1024)

    # Composer DB stats. Imported lazily to avoid an import cycle at module
    # load time (paths is imported very early).
    try:
        from .composer_db import enumerate_workspaces
        composer_folders: set[str] = set()
        ws_with_composers = 0
        for ws in enumerate_workspaces():
            ws_with_composers += 1
            composer_folders.add(ws.project_folder)
        result["projects_with_composer_db"] = ws_with_composers
        # Combined unique projects across both sources (same folder name or
        # short name unifies via string equality at this layer).
        jsonl_folders: set[str] = set()
        if projects_dir.exists():
            for p in projects_dir.iterdir():
                if not p.is_dir():
                    continue
                t = p / "agent-transcripts"
                if t.is_dir():
                    try:
                        if any(t.iterdir()):
                            jsonl_folders.add(p.name)
                    except OSError:
                        pass
        result["unique_projects_total"] = len(jsonl_folders | composer_folders)
    except Exception:
        # Never let the composer-DB probe crash the installation check.
        pass

    return result


def print_cursor_paths():
    """Print all Cursor paths for debugging."""
    print(f"Platform: {platform.system()}")
    print()
    print("Cursor Data Locations:")
    print(f"  Projects:          {get_cursor_projects_dir()}")
    print(f"  User Data:         {get_cursor_user_data_dir()}")
    print(f"  Workspace Storage: {get_workspace_storage_dir()}")
    print(f"  Global Storage:    {get_global_storage_dir()}")
    print(f"  Global DB:         {get_global_state_db()}")
    print()
    
    status = check_cursor_installation()
    print("Installation Status:")
    print(f"  Installed: {status['installed']}")
    print(f"  Projects: {status['projects_count']}")
    print(f"  Workspace DBs: {status['workspace_dbs']}")
    if status['global_db']:
        print(f"  Global DB Size: {status['global_db_size_mb']:.1f} MB")


if __name__ == "__main__":
    print_cursor_paths()

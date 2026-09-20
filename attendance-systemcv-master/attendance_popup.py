"""
Attendance History Popup
Shows a styled overlay window with:
  - Student name + roll number
  - Pie chart: Present vs Absent days
  - Numerical stats (total sessions, present, absent, percentage)
  - Countdown bar: auto-closes after AUTO_CLOSE_SECS seconds

Triggered by main.py whenever a student is newly marked present.
All heavy work (Excel scanning + matplotlib render) runs in a
background thread so the GUI stays responsive.
"""

import io
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("Agg")           # Non-interactive backend — no Tk conflict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from attendance_logger import get_student_history

# ── Popup configuration ─────────────────────────────────────────────────────
AUTO_CLOSE_SECS = 5          # How many seconds before the popup closes
POPUP_W, POPUP_H = 520, 380  # Popup dimensions (px)

# Dark colour palette
_BG       = "#0d1117"
_CARD     = "#161b22"
_BORDER   = "#30363d"
_GREEN    = "#3fb950"
_RED      = "#f85149"
_AMBER    = "#d29922"
_FG       = "#e6edf3"
_FG_DIM   = "#8b949e"
_ACCENT   = "#58a6ff"

# ── Public entry point ───────────────────────────────────────────────────────

def show_attendance_popup(parent: tk.Tk, name: str, roll_no: str,
                          mark_date: str = None):
    """
    Spawn the attendance-history popup for a student.
    Runs Excel scanning + chart rendering in a background thread,
    then displays the result on the main thread.

    Args:
        parent    : Root Tk window (used to center the popup).
        name      : Student name.
        roll_no   : Student roll number.
        mark_date : The date this student was just marked present (YYYY-MM-DD).
                    Passed directly to get_student_history so the popup is
                    accurate before the 60-second autosave flushes to Excel.
    """
    def _load_and_show():
        extra = [mark_date] if mark_date else []
        history = get_student_history(name, roll_no, extra_present_dates=extra)
        chart_img = _render_pie_chart(history, name)
        # Schedule the actual window creation on the main thread
        parent.after(0, lambda: _create_popup(parent, name, roll_no, history, chart_img))

    threading.Thread(target=_load_and_show, daemon=True).start()


# ── Internal: chart rendering ────────────────────────────────────────────────

def _render_pie_chart(history: dict, name: str) -> ImageTk.PhotoImage:
    """
    Render a donut-style pie chart using matplotlib and return
    an ImageTk.PhotoImage (MUST be called from a non-main thread;
    ImageTk conversion done back on main — see _create_popup).

    Returns:
        Raw PIL Image (not yet ImageTk, which must be created on main thread).
    """
    present = history["present_days"]
    absent  = history["absent_days"]
    total   = history["total_days"]
    pct     = history["pct"]

    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=100)
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    if total == 0:
        # No data — empty grey ring
        ax.pie([1], colors=[_BORDER], startangle=90,
               wedgeprops={"width": 0.55, "edgecolor": _BG, "linewidth": 2})
        ax.text(0, 0, "No\nData", ha="center", va="center",
                fontsize=13, color=_FG_DIM, fontweight="bold",
                fontfamily="monospace")
    else:
        sizes  = [present, absent] if absent > 0 else [present, 0.001]
        colors = [_GREEN, _RED]
        explode = (0.04, 0)

        wedges, _ = ax.pie(
            sizes, colors=colors, explode=explode,
            startangle=90,
            wedgeprops={"width": 0.55, "edgecolor": _BG, "linewidth": 2.5},
            shadow=False,
        )

        # Glow effect: draw a faint copy slightly larger behind
        ax.pie(
            sizes, colors=[c + "22" for c in [_GREEN, _RED]],
            startangle=90,
            wedgeprops={"width": 0.63, "edgecolor": "none"},
            shadow=False,
        )

        # Centre text: big percentage
        color_pct = _GREEN if pct >= 75 else (_AMBER if pct >= 50 else _RED)
        ax.text(0, 0.12, f"{pct:.1f}%",
                ha="center", va="center",
                fontsize=19, color=color_pct, fontweight="bold",
                fontfamily="sans-serif")
        ax.text(0, -0.22, "Attendance",
                ha="center", va="center",
                fontsize=8.5, color=_FG_DIM,
                fontfamily="sans-serif")

    ax.axis("equal")
    plt.tight_layout(pad=0.2)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight",
                facecolor=_BG, dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()   # .copy() so we can close the buf


# ── Internal: popup window ───────────────────────────────────────────────────

def _create_popup(parent: tk.Tk, name: str, roll_no: str,
                   history: dict, chart_pil: Image.Image):
    """
    Build and display the popup Toplevel window.
    Must be called on the main thread.
    """
    if not parent.winfo_exists():
        return

    popup = tk.Toplevel(parent)
    popup.title("")
    popup.overrideredirect(True)          # Borderless window
    popup.configure(bg=_BG)
    popup.attributes("-topmost", True)

    # ── Sizing & centering ──────────────────────────────────────────────
    popup.update_idletasks()
    px = parent.winfo_rootx() + (parent.winfo_width()  - POPUP_W) // 2
    py = parent.winfo_rooty() + (parent.winfo_height() - POPUP_H) // 2
    popup.geometry(f"{POPUP_W}x{POPUP_H}+{px}+{py}")

    # ── Rounded-ish border frame ────────────────────────────────────────
    outer = tk.Frame(popup, bg=_BORDER, padx=1, pady=1)
    outer.pack(fill=tk.BOTH, expand=True)

    inner = tk.Frame(outer, bg=_BG)
    inner.pack(fill=tk.BOTH, expand=True)

    # ── Header bar ──────────────────────────────────────────────────────
    header = tk.Frame(inner, bg=_CARD, height=52)
    header.pack(fill=tk.X)
    header.pack_propagate(False)

    # Coloured left accent stripe
    tk.Frame(header, bg=_ACCENT, width=4).pack(side=tk.LEFT, fill=tk.Y)

    info_frame = tk.Frame(header, bg=_CARD)
    info_frame.pack(side=tk.LEFT, padx=12, pady=8)

    pct = history["pct"]
    pct_color = _GREEN if pct >= 75 else (_AMBER if pct >= 50 else _RED)

    tk.Label(info_frame, text=f"✅  {name}",
             bg=_CARD, fg=_FG,
             font=("Segoe UI", 13, "bold")).pack(anchor=tk.W)
    tk.Label(info_frame, text=f"Roll: {roll_no}   •   Marked Present",
             bg=_CARD, fg=_FG_DIM,
             font=("Segoe UI", 9)).pack(anchor=tk.W)

    # Countdown label (right side of header)
    countdown_var = tk.StringVar(value=f"Closing in {AUTO_CLOSE_SECS}s")
    tk.Label(header, textvariable=countdown_var,
             bg=_CARD, fg=_FG_DIM,
             font=("Segoe UI", 9)).pack(side=tk.RIGHT, padx=14)

    # ── Body: chart + stats side by side ────────────────────────────────
    body = tk.Frame(inner, bg=_BG)
    body.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

    # Left: pie chart
    chart_frame = tk.Frame(body, bg=_BG)
    chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 4), pady=10)

    chart_imgtk = ImageTk.PhotoImage(chart_pil.resize((240, 240), Image.LANCZOS))
    chart_lbl = tk.Label(chart_frame, image=chart_imgtk, bg=_BG)
    chart_lbl.image = chart_imgtk   # keep reference
    chart_lbl.pack(expand=True)

    # Right: stats card
    stats_frame = tk.Frame(body, bg=_CARD, bd=0,
                            relief="flat", width=210)
    stats_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 10), pady=10)
    stats_frame.pack_propagate(False)

    # Title
    tk.Label(stats_frame, text="📊  Attendance History",
             bg=_CARD, fg=_FG,
             font=("Segoe UI", 10, "bold")).pack(pady=(14, 6), padx=14, anchor=tk.W)

    # Separator
    tk.Frame(stats_frame, bg=_BORDER, height=1).pack(fill=tk.X, padx=14, pady=(0, 8))

    total   = history["total_days"]
    present = history["present_days"]
    absent  = history["absent_days"]

    def _stat_row(label, value, value_color=_FG):
        row = tk.Frame(stats_frame, bg=_CARD)
        row.pack(fill=tk.X, padx=14, pady=3)
        tk.Label(row, text=label, bg=_CARD, fg=_FG_DIM,
                 font=("Segoe UI", 9), width=14,
                 anchor=tk.W).pack(side=tk.LEFT)
        tk.Label(row, text=str(value), bg=_CARD, fg=value_color,
                 font=("Segoe UI", 10, "bold"),
                 anchor=tk.W).pack(side=tk.LEFT)

    _stat_row("Total Sessions",  total)
    _stat_row("Present",         present, _GREEN)
    _stat_row("Absent",          absent,  _RED if absent > 0 else _FG)
    _stat_row("Percentage",      f"{pct:.1f}%", pct_color)

    # Separator
    tk.Frame(stats_frame, bg=_BORDER, height=1).pack(fill=tk.X, padx=14, pady=(10, 8))

    # Status badge
    if pct >= 75:
        badge_text, badge_bg, badge_fg = "✔  GOOD STANDING", "#1a3a22", _GREEN
    elif pct >= 50:
        badge_text, badge_bg, badge_fg = "⚠  NEEDS IMPROVEMENT", "#3a2e0d", _AMBER
    else:
        badge_text, badge_bg, badge_fg = "✘  LOW ATTENDANCE", "#3a1212", _RED

    badge = tk.Label(stats_frame, text=badge_text,
                     bg=badge_bg, fg=badge_fg,
                     font=("Segoe UI", 8, "bold"),
                     padx=8, pady=4)
    badge.pack(padx=14, anchor=tk.W)

    # ── Countdown progress bar ───────────────────────────────────────────
    bar_frame = tk.Frame(inner, bg=_BG, height=6)
    bar_frame.pack(fill=tk.X, side=tk.BOTTOM)
    bar_canvas = tk.Canvas(bar_frame, bg=_BORDER, height=6,
                           highlightthickness=0, bd=0)
    bar_canvas.pack(fill=tk.X)

    # Allow canvas to initialise its width
    popup.update_idletasks()
    bar_w = bar_canvas.winfo_width() or POPUP_W

    bar_rect = bar_canvas.create_rectangle(
        0, 0, bar_w, 6, fill=_ACCENT, outline=""
    )

    # ── Countdown timer ──────────────────────────────────────────────────
    remaining = [AUTO_CLOSE_SECS * 10]   # tenths of a second

    def _tick():
        if not popup.winfo_exists():
            return
        remaining[0] -= 1
        frac = remaining[0] / (AUTO_CLOSE_SECS * 10)
        new_w = int(bar_canvas.winfo_width() * frac)
        bar_canvas.coords(bar_rect, 0, 0, new_w, 6)

        secs_left = max(1, (remaining[0] + 9) // 10)
        countdown_var.set(f"Closing in {secs_left}s")

        if remaining[0] <= 0:
            popup.destroy()
        else:
            popup.after(100, _tick)   # update every 100 ms

    popup.after(100, _tick)

    # Allow manual close with Escape or click outside
    popup.bind("<Escape>", lambda e: popup.destroy())
    popup.bind("<Button-1>", lambda e: popup.destroy())

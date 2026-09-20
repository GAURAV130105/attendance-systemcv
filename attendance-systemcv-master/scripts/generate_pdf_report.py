import os
import sys
import ast
from fpdf import FPDF

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
EXCLUDE_DIRS = {".git", "__pycache__", "scripts", "encodings", "known_faces", "attendance_records", "models"}

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUTPUT_DIR = os.path.join(ROOT, "docs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUT_PDF = os.path.join(OUTPUT_DIR, "codebase_report.pdf")

reqs = []
req_file = os.path.join(ROOT, "requirements.txt")
if os.path.exists(req_file):
    with open(req_file, "r", encoding="utf-8") as f:
        reqs = [ln.strip() for ln in f if ln.strip()]


def analyze_file(path):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    try:
        mod = ast.parse(src)
    except Exception as e:
        return {"path": path, "error": str(e)}
    info = {"path": path, "doc": ast.get_docstring(mod) or "", "classes": [], "functions": []}
    for node in mod.body:
        if isinstance(node, ast.ClassDef):
            methods = []
            for c in node.body:
                if isinstance(c, ast.FunctionDef):
                    sig = c.name + ast.dump(ast.arguments(args=c.args.args, vararg=c.args.vararg, kwarg=c.args.kwarg)) if False else c.name
                    methods.append({"name": c.name, "doc": ast.get_docstring(c) or ""})
            info["classes"].append({"name": node.name, "doc": ast.get_docstring(node) or "", "methods": methods})
        elif isinstance(node, ast.FunctionDef):
            info["functions"].append({"name": node.name, "doc": ast.get_docstring(node) or ""})
    return info


def collect_py_files(root):
    py_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # skip excluded dirs
        parts = set(dirpath.replace(root, '').split(os.sep))
        if parts & EXCLUDE_DIRS:
            continue
        for fn in filenames:
            if fn.endswith('.py'):
                py_files.append(os.path.join(dirpath, fn))
    return sorted(py_files)


def build_report(items, requirements):
    lines = []
    lines.append("Smart Attendance System — Codebase Report")
    lines.append("")
    lines.append("Overview:")
    lines.append("- Project root: %s" % ROOT)
    lines.append("")
    lines.append("Dependencies (from requirements.txt):")
    if requirements:
        for r in requirements:
            lines.append("- %s" % r)
    else:
        lines.append("- (none found)")
    lines.append("")
    lines.append("Modules:")
    for info in items:
        lines.append("")
        lines.append("File: %s" % os.path.relpath(info['path'], ROOT))
        if info.get('error'):
            lines.append("  ERROR PARSING: %s" % info['error'])
            continue
        if info['doc']:
            lines.append("  Docstring: %s" % (info['doc'].splitlines()[0][:200] if info['doc'] else ""))
        if info['classes']:
            lines.append("  Classes:")
            for cls in info['classes']:
                lines.append("    - %s: %s" % (cls['name'], (cls['doc'].splitlines()[0][:150] if cls['doc'] else "")))
                if cls['methods']:
                    for m in cls['methods']:
                        lines.append("       * %s: %s" % (m['name'], (m['doc'].splitlines()[0][:150] if m['doc'] else "")))
        if info['functions']:
            lines.append("  Functions:")
            for fn in info['functions']:
                lines.append("    - %s: %s" % (fn['name'], (fn['doc'].splitlines()[0][:150] if fn['doc'] else "")))
    return "\n".join(lines)


def write_pdf(text, out_path):
    pdf = FPDF(orientation='P', unit='pt', format='A4')
    pdf.set_auto_page_break(auto=True, margin=40)
    pdf.add_page()
    # Use built-in Helvetica font (sufficient for ASCII content)
    pdf.set_font('Helvetica', '', 12)

    # Simple wrapping: write lines with MultiCell
    # Replace problematic unicode with ASCII equivalents
    repl = {
        '\u2014': ' - ',
        '\u2022': '-',
        '\u2026': '...',
        '•': '-',
        '—': ' - ',
        '✓': 'v',
        '✅': '[OK]',
        '→': '->',
        '←': '<-',
        '📊': '',
        '📅': '',
        '📸': '',
        '🔁': '',
        '⚠️': 'WARNING',
        '✔': 'OK',
        '✘': 'X',
        '✅': '[OK]'
    }
    safe_text = text
    for k, v in repl.items():
        safe_text = safe_text.replace(k, v)

    for paragraph in safe_text.split('\n\n'):
        pdf.multi_cell(0, 14, paragraph)
        pdf.ln(6)

    pdf.output(out_path)


if __name__ == '__main__':
    files = collect_py_files(ROOT)
    items = [analyze_file(p) for p in files]
    report_text = build_report(items, reqs)
    write_pdf(report_text, OUT_PDF)
    print('PDF generated at', OUT_PDF)

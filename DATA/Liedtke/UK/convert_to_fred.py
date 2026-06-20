"""Convert all UK data files to FRED format: observation_date, comma separator, period for decimals."""
import csv
import io
import os
from datetime import datetime, date
import calendar

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

MONTH_MAP = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12,
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
}

QUARTER_END = {'Q1': 3, 'Q2': 6, 'Q3': 9, 'Q4': 12}


def write_fred(filename, rows, col_name):
    """Write rows as FRED CSV: observation_date, col_name."""
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['observation_date', col_name])
        for date_str, value in rows:
            writer.writerow([date_str, value])
    print(f"  Written {filename}: {len(rows)} rows")


def parse_double_encoded(filepath):
    """Parse files where each row is a single double-encoded CSV field."""
    with open(filepath, encoding='utf-8') as f:
        content = f.read()
    reader = csv.reader(io.StringIO(content))
    raw_rows = list(reader)
    result = []
    for raw in raw_rows:
        if not raw:
            continue
        inner = next(csv.reader([raw[0]]))
        result.append(inner)
    return result


def month_abbr_to_date(date_str):
    """Convert '1988 JAN' to '1988-01-01'."""
    parts = date_str.strip().split()
    year = int(parts[0])
    month = MONTH_MAP[parts[1]]
    return f"{year:04d}-{month:02d}-01"


def quarter_to_date(date_str):
    """Convert '1982 Q1' to '1982-03-01' (end of quarter, first of that month)."""
    parts = date_str.strip().split()
    year = int(parts[0])
    end_month = QUARTER_END[parts[1]]
    return f"{year:04d}-{end_month:02d}-01"


# ── Files already in FRED format ──────────────────────────────────────────────
print("Already-FRED files: CapacityUtilization, FedFundsRate, Yield10Y, Yield3M")
for fname in ['CapacityUtilization.csv', 'FedFundsRate.csv', 'Yield10Y.csv', 'Yield3M.csv']:
    # Re-write to ensure comma separator and consistent format
    path = os.path.join(DATA_DIR, fname)
    with open(path, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    print(f"  Confirmed {fname}: {len(rows)-1} rows")

# ── Double-encoded files: CoreCPI, CPI, Payroll, TradeSales ──────────────────
print("\nDouble-encoded monthly files:")

DOUBLE_ENC_FILES = {
    'CoreCPI.csv': 'CoreCPI',
    'CPI.csv': 'CPI',
    'Payroll.csv': 'Payroll',
    'TradeSales.csv': 'TradeSales',
}

for fname, col_name in DOUBLE_ENC_FILES.items():
    rows_raw = parse_double_encoded(os.path.join(DATA_DIR, fname))
    out = []
    for row in rows_raw[1:]:  # skip header
        if len(row) < 2:
            continue
        date_str = month_abbr_to_date(row[0])
        out.append((date_str, row[1]))
    write_fred(fname, out, col_name)

# ── Unemployment: mixed annual/quarterly/monthly — keep only monthly ──────────
print("\nUnemployment (keeping monthly section):")
rows_raw = parse_double_encoded(os.path.join(DATA_DIR, 'Unemployment.csv'))
monthly_rows = []
import re
for row in rows_raw[1:]:
    if len(row) < 2:
        continue
    date_field = row[0].strip()
    if re.match(r'^\d{4} [A-Z]{3}$', date_field):
        date_str = month_abbr_to_date(date_field)
        monthly_rows.append((date_str, row[1]))
write_fred('Unemployment.csv', monthly_rows, 'Unemployment')

# ── Housing: semicolon-separated quarterly ────────────────────────────────────
print("\nHousing (quarterly, semicolon-separated):")
with open(os.path.join(DATA_DIR, 'Housing.csv'), encoding='utf-8') as f:
    lines = [l.rstrip('\n') for l in f if l.strip()]
header = lines[0]  # "Period;Started - All Dwellings"
out = []
for line in lines[1:]:
    period, value = line.split(';', 1)
    period = period.strip()
    value = value.strip()
    # Format: "Jan - Mar 1978" → end month = Mar, year = 1978
    # or "Apr - Jun 1978"
    parts = period.split(' - ')
    end_part = parts[1].strip()  # e.g. "Mar 1978"
    end_tokens = end_part.split()
    end_month = MONTH_MAP[end_tokens[0]]
    year = int(end_tokens[1])
    date_str = f"{year:04d}-{end_month:02d}-01"
    out.append((date_str, value))
write_fred('Housing.csv', out, 'HousingStarts')

# ── DXY: "DD MMM YY" end-of-month dates, reverse order ───────────────────────
print("\nDXY (end-of-month dates, reverse order):")
with open(os.path.join(DATA_DIR, 'DXY.csv'), newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)

# Second column name (long description) — shorten it
col_name = 'GBPEER'  # GBP Effective Exchange Rate
out = []
for row in rows[1:]:
    if len(row) < 2 or not row[0].strip():
        continue
    date_raw = row[0].strip()
    value = row[1].strip()
    # Parse "31 May 26" or "31 Mar 90"
    dt = datetime.strptime(date_raw, '%d %b %y')
    date_str = dt.strftime('%Y-%m-%d')
    out.append((date_str, value))
# Sort ascending
out.sort(key=lambda x: x[0])
write_fred('DXY.csv', out, col_name)

# ── M2MoneySupply: end-of-month dates already in ISO format ──────────────────
print("\nM2MoneySupply (end-of-month dates):")
with open(os.path.join(DATA_DIR, 'M2MoneySupply.csv'), newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)
out = []
for row in rows[1:]:
    if len(row) < 2 or not row[0].strip():
        continue
    out.append((row[0].strip(), row[1].strip()))
write_fred('M2MoneySupply.csv', out, 'M2MoneySupply')

# ── T10Y3M: Yield10Y - Yield3M on shared dates ───────────────────────────────
print("\nCalculating T10Y3M term spread:")
def load_fred(fname):
    path = os.path.join(DATA_DIR, fname)
    data = {}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row['observation_date'].strip()
            val = row[list(row.keys())[1]].strip()
            if val and val.upper() != 'NA' and val != '.':
                try:
                    data[d] = float(val)
                except ValueError:
                    pass
    return data

y10 = load_fred('Yield10Y.csv')
y3m = load_fred('Yield3M.csv')

shared = sorted(set(y10.keys()) & set(y3m.keys()))
spread_rows = []
for d in shared:
    spread = round(y10[d] - y3m[d], 4)
    spread_rows.append((d, spread))

write_fred('T10Y3M.csv', spread_rows, 'T10Y3M')

print(f"\nDone. T10Y3M: {len(spread_rows)} shared dates.")
print(f"  Date range: {spread_rows[0][0]} to {spread_rows[-1][0]}")

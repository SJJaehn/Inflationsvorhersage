import os
import pandas as pd
import numpy as np

def seasonal_adjustment(file_path):
    print(f"Lese Daten ein aus: {file_path}")
    
    # 1. Daten einlesen (ohne feste Spaltennamen)
    df = pd.read_csv(file_path)
    
    # Dynamische Spaltenzuweisung: 1. Spalte = Datum, 2. Spalte = Werte
    date_col = df.columns[0]
    val_col = df.columns[1]
    
    print(f"Erkannte Datums-Spalte: '{date_col}'")
    print(f"Erkannte Werte-Spalte: '{val_col}'")
    
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # 2. Prozentuale Veränderung (Inflation) berechnen
    df['infl'] = df[val_col].pct_change() * 100
    df['month'] = df[date_col].dt.month
    
    # Erste Zeile entfernen (ist NaN durch pct_change) für die Effektberechnung
    df_calc = df.dropna().copy()
    
    infl_values = df_calc['infl'].values
    n = len(infl_values)
    
    # Array für die individuellen Abweichungen im Fenster [t-6; t+5]
    deviations = np.full(n, np.nan)
    
    # 3. Berechne die Abweichung für jedes t im Fenster [t-6; t+5]
    for t in range(6, n - 5):
        # Fenster von t-6 bis einschließlich t+5 (insgesamt 12 Monate)
        window = infl_values[t-6 : t+6]
        window_mean = np.mean(window)
        
        # Vergleich der Inflation in t mit dem Mittelwert dieses 12-Monats-Fensters
        deviations[t] = infl_values[t] - window_mean
        
    df_calc['deviation'] = deviations
    
    # 4. Durchschnittlichen Effekt pro Kalendermonat berechnen
    monthly_effects = df_calc.groupby('month')['deviation'].mean().to_dict()
    
    print("\nBerechnete saisonale Effekte pro Monat (Abweichung vom [t-6;t+5] Schnitt):")
    for m in sorted(monthly_effects.keys()):
        print(f"Monat {m:02d}: {monthly_effects[m]:+.4f} Prozentpunkte")
        
    # 5. Saisonalen Effekt von ALLEN Daten abziehen (inkl. Ränder am Anfang/Ende)
    df['infl_adj'] = df.apply(
        lambda row: row['infl'] - monthly_effects[row['month']] if pd.notna(row['infl']) else np.nan, 
        axis=1
    )
    
    # 6. Rekonstruktion des saisonbereinigten Index via klassischer Prozentrechnung
    val_adj = [df[val_col].iloc[0]]
    for i in range(1, len(df)):
        next_val = val_adj[-1] * (1 + df['infl_adj'].iloc[i] / 100)
        val_adj.append(next_val)
        
    adj_col_name = f"{val_col}_adj"
    df[adj_col_name] = val_adj
    
    # 7. Dynamischen Ausgabe-Pfad anpassen
    folder = os.path.dirname(file_path)
    # Holt den reinen Dateinamen ohne Endung (z.B. "CPI" aus "CPI.csv")
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_filename = f"{base_name}_adj.csv"
    output_path = os.path.join(folder, output_filename)
    
    # Spaltenreihenfolge strukturieren und speichern
    df_output = df[[date_col, adj_col_name, val_col, 'infl', 'infl_adj']]
    df_output.to_csv(output_path, index=False)
    print(f"\nErfolgreich gespeichert unter: {output_path}")

if __name__ == "__main__":
    # Pfad zu deiner Datei
    input_path = '/Users/simonjaehn/Downloads/Liedtke/Codes-2/DATA/Liedtke/UK-Adj/CPI.csv'
    
    if os.path.exists(input_path):
        seasonal_adjustment(input_path)
    else:
        print(f"Datei nicht gefunden unter: {input_path}")
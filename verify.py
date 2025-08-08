# verify.py
import numpy as np
import os

BASE_DIR = os.path.dirname(__file__)

# --- Pfade zu den zu vergleichenden Dateien anpassen ---
# Annahme: Deine alten Daten liegen noch in der alten Struktur
# Passe die Pfade an, falls sie woanders liegen.
OLD_STATE_FILE = os.path.join(BASE_DIR, 'Data_old', 'generated_data', 'state_data_uniform.npy') # Beispielpfad
NEW_STATE_FILE = os.path.join(BASE_DIR, 'data', 'generated', 'uniform_run_1_state_state.npy') # Beispielpfad

print(f"Vergleiche alte Datei:\n {OLD_STATE_FILE}")
print(f"mit neuer Datei:\n {NEW_STATE_FILE}\n")

# --- DEBUGGING-SCHRITT HINZUGEFÜGT ---
print(f"Prüfe Existenz von OLD_STATE_FILE: {os.path.exists(OLD_STATE_FILE)}")
print(f"Prüfe Existenz von NEW_STATE_FILE: {os.path.exists(NEW_STATE_FILE)}\n")
# -----------
try:
    # Lade beide Arrays
    old_array = np.load(OLD_STATE_FILE)
    new_array = np.load(NEW_STATE_FILE)

    # Überprüfe, ob Form und Inhalt exakt übereinstimmen
    if old_array.shape == new_array.shape and np.array_equal(old_array, new_array):
        print("✅ SUCCESS: Die Arrays sind exakt identisch!")
    # Falls es minimale Fließkomma-Unterschiede gibt (sollte hier nicht der Fall sein)
    elif old_array.shape == new_array.shape and np.allclose(old_array, new_array):
        print("⚠️ INFO: Die Arrays sind nicht exakt identisch, aber sehr nahe beieinander (numerisch stabil).")
    else:
        print("❌ FAILURE: Die Arrays sind unterschiedlich!")
        print(f"Alte Form: {old_array.shape}, Neue Form: {new_array.shape}")

except FileNotFoundError:
    print("Fehler: Eine oder beide Dateien wurden nicht gefunden. Bitte Pfade überprüfen.")
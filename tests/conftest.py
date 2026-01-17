import sys
import os

# Füge den 'lib' Ordner zum Pfad hinzu, damit 'sharpfin' als Top-Level-Modul gefunden wird
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib'))
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

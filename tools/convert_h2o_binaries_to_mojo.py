#!/usr/bin/env python3
"""
Convierte modelos binarios H2O a MOJO en un entorno legacy.

Uso:
  python tools/convert_h2o_binaries_to_mojo.py \
    --models-file /tmp/model_paths.txt \
    --output-dir /tmp/mojos \
    --genmodel-jar

Archivo model_paths.txt:
  /ruta/modelo_1
  /ruta/modelo_2
  ...
"""

import argparse
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import h2o


def _run_cmd(cmd):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as exc:
        return -1, "", f"{type(exc).__name__}: {exc}"


def _print_env():
    print("=== Entorno ===")
    print("Python:", sys.version)
    print("H2O:", h2o.__version__)
    rc, out, err = _run_cmd(["java", "-version"])
    print("java -version rc:", rc)
    print((out + "\n" + err).strip())
    print()


def _read_models_list(models_file):
    rows = []
    with open(models_file, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if item and not item.startswith("#"):
                rows.append(item)
    return rows


def _convert_model(model_path, output_dir, genmodel_jar):
    model_path = Path(model_path)
    print(f"[INFO] Procesando: {model_path}")
    print(
        f"[INFO] exists={model_path.exists()} is_file={model_path.is_file()} is_dir={model_path.is_dir()}"
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo inexistente: {model_path}")

    model = h2o.load_model(str(model_path))
    downloaded = Path(model.download_mojo(path=str(output_dir), get_genmodel_jar=genmodel_jar))
    canonical = Path(str(model_path) + ".zip")
    canonical.parent.mkdir(parents=True, exist_ok=True)
    if canonical.exists():
        canonical.unlink()
    shutil.copy2(downloaded, canonical)
    print(f"[OK] MOJO exportado: {downloaded}")
    print(f"[OK] MOJO canonical: {canonical}")
    return canonical


def main():
    parser = argparse.ArgumentParser(
        description="Convierte modelos binarios H2O a MOJO (entorno legacy)."
    )
    parser.add_argument(
        "--models-file",
        required=True,
        help="TXT con una ruta de modelo por linea.",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        default="/tmp/h2o_mojo_export",
        help="Directorio temporal de descarga para ZIP MOJO (default: /tmp/h2o_mojo_export).",
    )
    parser.add_argument(
        "--genmodel-jar",
        action="store_true",
        help="Descarga h2o-genmodel.jar junto con los MOJO.",
    )
    args = parser.parse_args()

    models_file = Path(args.models_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _print_env()
    print("[INFO] Iniciando H2O...")
    h2o.init()

    models = _read_models_list(models_file)
    if not models:
        raise RuntimeError(f"El archivo {models_file} no contiene rutas de modelos.")

    print(f"[INFO] Modelos a convertir: {len(models)}")
    ok = 0
    fail = 0

    for model_path in models:
        try:
            _convert_model(model_path, output_dir, args.genmodel_jar)
            ok += 1
        except Exception:
            fail += 1
            print("[ERROR] Fallo conversion")
            print(traceback.format_exc())

    print(f"\n=== Resumen ===\nOK={ok}\nFAIL={fail}")
    if fail > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

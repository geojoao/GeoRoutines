"""
Entrypoint da rotina de produção ``viirs-evi-hexagono`` (kbatch).

Fluxo:
  1. instala as dependências (requirements.txt);
  2. baixa o ``evi_brazil.parquet`` existente do blob de saída (estado
     incremental) e as partes de um run interrompido (resume);
  3. monta o universo de hexágonos (grid H3 do Brasil a partir de um boundary
     no blob de entrada, ou de uma lista de hexágonos, ou de um bbox);
  4. prepara as máscaras MapBiomas do run (baixa os COGs por padrão; leitura
     remota /vsicurl opcional via VIIRS_MAPBIOMAS_REMOTE=1);
  5. processa balde a balde, **incrementalmente** (só datas/hexágonos que
     faltam), enviando cada parte ao blob assim que fica pronta (durável);
  6. faz o merge canônico + partes -> novo ``evi_brazil.parquet`` e envia ao
     blob; limpa as partes.

Variáveis de ambiente relevantes (ver viirs_evi/config.py):
  AZURE_STORAGE_CONNECTION_STRING   (obrigatória)
  EARTHDATA_USERNAME / EARTHDATA_PASSWORD  (ou ~/.netrc)
  FENOLOGIA_ACCESS=s3               (recomendado in-region us-west-2)
  VIIRS_START_DATE=2012-01-17       (início da série; ajuste p/ reduzir custo)
  VIIRS_BOUNDARY_BLOB=brazil.geojson  (contorno no container de input)
  VIIRS_HEX_LIST_BLOB=...           (alternativa: CSV/parquet com id_hexagono)
  VIIRS_BBOX="minx miny maxx maxy"  (alternativa de teste)
  VIIRS_LIMIT=N                     (processa só N hexágonos — teste)
"""
from __future__ import annotations


def _install_requirements():
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def _log(msg: str):
    import datetime as dt
    print(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {msg}", flush=True)


def _build_hex_ids(config, blob):
    """Universo de hexágonos: lista explícita, boundary ou bbox."""
    import os

    from viirs_evi.grid import build_h3_grid

    hex_list_blob = os.environ.get("VIIRS_HEX_LIST_BLOB", "").strip()
    if hex_list_blob:
        import pandas as pd

        dest = config.WORK_DIR / os.path.basename(hex_list_blob)
        if not blob.download_input(hex_list_blob, dest):
            raise FileNotFoundError(f"Lista de hexágonos '{hex_list_blob}' não achada no input.")
        df = pd.read_parquet(dest) if str(dest).endswith(".parquet") else pd.read_csv(dest)
        col = "id_hexagono" if "id_hexagono" in df.columns else df.columns[0]
        return sorted(set(df[col].astype(str)))

    bbox_env = os.environ.get("VIIRS_BBOX", "").strip()
    if bbox_env:
        bbox = tuple(float(x) for x in bbox_env.split())
        _log(f"Grid a partir do bbox {bbox}")
        return build_h3_grid(bbox=bbox, resolution=config.H3_RESOLUTION)["h3"].tolist()

    # padrão: boundary do Brasil no blob de input
    import os as _os

    boundary_blob = _os.environ.get("VIIRS_BOUNDARY_BLOB", "brazil.geojson")
    if blob.download_input(boundary_blob, config.BRAZIL_BOUNDARY):
        import geopandas as gpd

        _log(f"Grid a partir do boundary {boundary_blob}")
        gdf = gpd.read_file(config.BRAZIL_BOUNDARY)
        return build_h3_grid(boundary=gdf, resolution=config.H3_RESOLUTION)["h3"].tolist()

    raise RuntimeError(
        "Sem universo de hexágonos: defina VIIRS_HEX_LIST_BLOB, ou coloque um "
        "boundary (VIIRS_BOUNDARY_BLOB) no input, ou use VIIRS_BBOX."
    )


def main():
    _install_requirements()

    import os
    import time

    import pandas as pd

    from viirs_evi import blob, config, extract, mapbiomas, merge, pipeline, state, tiles, viirs

    _log("=== viirs-evi-hexagono ===")
    start_d = config.start_date()
    end_d = config.end_date()
    _log(f"Janela: {start_d} .. {end_d} | acesso VIIRS: {config.VIIRS_ACCESS_MODE}")

    # --- 1) estado incremental: baixa o parquet canônico + partes de resume ---
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / "_parts").mkdir(parents=True, exist_ok=True)
    canonical_local = config.OUTPUT_DIR / config.OUTPUT_PARQUET
    blob.download_output(blob.output_blob_name(config.OUTPUT_PARQUET), canonical_local)

    part_prefix = blob.output_blob_name("_parts") + "/"
    existing_parts = blob.list_blobs(part_prefix, config.BLOB_OUTPUT_CONTAINER)
    done_labels = set()
    for pb in existing_parts:
        name = pb.split("/")[-1]  # balde_<label>.parquet
        local = config.OUTPUT_DIR / "_parts" / name
        blob.download_output(pb, local)
        done_labels.add(name[len("balde_"):-len(".parquet")])
    if done_labels:
        _log(f"Resume: {len(done_labels)} baldes já concluídos num run anterior.")

    # --- 2) universo de hexágonos ---
    hex_ids = _build_hex_ids(config, blob)
    limit = os.environ.get("VIIRS_LIMIT", "").strip()
    if limit:
        hex_ids = hex_ids[: int(limit)]
    _log(f"Hexágonos: {len(hex_ids)}")

    last_date_by_hex = state.load_last_date_by_hexagon(canonical_local)
    _log(f"Hexágonos já presentes no parquet: {len(last_date_by_hex)}")

    # --- 3) máscaras MapBiomas usadas neste run (leitura remota; resolve URLs) ---
    years = pipeline.processing_years(last_date_by_hex, hex_ids, start_d, end_d)
    _log(f"Anos a tocar neste run: {years[0] if years else '-'}..{years[-1] if years else '-'}")
    mapbiomas.prefetch(years)

    # --- 4) processa balde a balde (incremental) ---
    buckets = pipeline.buckets_for_hexagons(hex_ids)
    _log(f"Baldes de tiles: {len(buckets)}")

    n_new_rows = 0
    times = []
    for i, (tiles_hv, bucket_hexes) in enumerate(buckets.items(), 1):
        label = tiles.bucket_label(list(tiles_hv))
        if label in done_labels:
            continue
        t0 = time.time()
        try:
            results = pipeline.process_tile_bucket_incremental(
                bucket_hexes, tiles_hv, last_date_by_hex, start_d, end_d,
            )
        except Exception as exc:
            _log(f"[{label}] ERRO {exc!r}")
            continue

        frames = [df for df in results.values() if len(df)]
        part_local = config.OUTPUT_DIR / "_parts" / f"balde_{label}.parquet"
        bucket_df = (
            pd.concat(frames, ignore_index=True)
            if frames else pd.DataFrame(columns=extract.full_wide_columns())
        )
        bucket_df.to_parquet(part_local, index=False)
        # envia a parte já (durável p/ resume)
        blob.upload_output(part_local, blob.output_blob_name("_parts", f"balde_{label}.parquet"))

        n_new_rows += len(bucket_df)
        dt_s = time.time() - t0
        times.append(dt_s)
        eta_min = (sum(times) / len(times)) * (len(buckets) - i) / 60.0
        _log(f"[{label}] {i}/{len(buckets)} | {len(bucket_hexes)} hex | "
             f"+{len(bucket_df)} linhas | {dt_s:.0f}s | ETA ~{eta_min:.0f} min")

    # --- 5) merge canônico + partes -> novo parquet, envia e limpa ---
    part_files = sorted((config.OUTPUT_DIR / "_parts").glob("balde_*.parquet"))
    merged_local = config.OUTPUT_DIR / f"_merged_{config.OUTPUT_PARQUET}"
    total = merge.merge_to_canonical(canonical_local, part_files, merged_local)
    os.replace(merged_local, canonical_local)
    blob.upload_output(canonical_local, blob.output_blob_name(config.OUTPUT_PARQUET))
    _log(f"Parquet canônico atualizado: {total} linhas totais (+{n_new_rows} novas).")

    # limpa as partes do blob (run concluído)
    for pb in blob.list_blobs(part_prefix, config.BLOB_OUTPUT_CONTAINER):
        blob.delete(pb, config.BLOB_OUTPUT_CONTAINER)
    _log("Partes limpas. Concluído.")


if __name__ == "__main__":
    main()

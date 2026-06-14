"""
CLI da rotina de fenologia (EVI VIIRS x MapBiomas por hexágono H3).

Exemplos
--------
# Teste em uma região pequena (bbox) e só 2 hexágonos:
uv run python run.py --bbox -48.2 -16.1 -47.8 -15.7 --limit 2

# Brasil inteiro (precisa de data/brazil.geojson):
uv run python run.py --boundary data/brazil.geojson

# Lista explícita de hexágonos:
uv run python run.py --hex 85a8d3b7fffffff 85a8d3a3fffffff
"""
from __future__ import annotations

import argparse

from fenologia import config
from fenologia.pipeline import run


def main():
    p = argparse.ArgumentParser(description="EVI VIIRS x MapBiomas por hexágono H3 (res 5).")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--boundary", help="Arquivo vetorial com o contorno do Brasil.")
    src.add_argument("--bbox", nargs=4, type=float, metavar=("MINX", "MINY", "MAXX", "MAXY"),
                     help="Caixa para gerar o grid (região de teste).")
    src.add_argument("--hex", nargs="+", help="Lista explícita de IDs de hexágonos H3.")
    p.add_argument("--years", nargs="+", type=int, default=config.YEARS,
                   help=f"Anos a processar (padrão {config.YEARS}).")
    p.add_argument("--output", default=str(config.OUTPUT_DIR), help="Pasta de saída dos parquets.")
    p.add_argument("--limit", type=int, default=None, help="Processa apenas N hexágonos.")
    p.add_argument("--no-resume", action="store_true", help="Reprocessa mesmo se o parquet existir.")
    p.add_argument("--no-prepare", action="store_true",
                   help="Não pré-gera os COGs (VIIRS + MapBiomas) por tile antes do loop.")
    args = p.parse_args()

    boundary = None
    if args.boundary:
        from fenologia.grid import load_brazil_boundary
        boundary = load_brazil_boundary(args.boundary)

    run(
        boundary=boundary,
        bbox=tuple(args.bbox) if args.bbox else None,
        hex_ids=args.hex,
        years=args.years,
        output_dir=args.output,
        limit=args.limit,
        resume=not args.no_resume,
        prepare=not args.no_prepare,
    )


if __name__ == "__main__":
    main()

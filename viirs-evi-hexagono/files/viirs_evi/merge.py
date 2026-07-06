"""
Merge em *streaming* do parquet canônico com as partes por balde.

As partes são deltas disjuntos (cada balde tem hexágonos distintos, e cada delta
traz só datas ainda não presentes no canônico), então basta **concatenar** —
sem dedup. O merge é feito row-group a row-group via pyarrow para não carregar
o Brasil inteiro na memória (o primeiro run pode ter milhões de linhas).
"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .extract import full_wide_columns


def target_schema() -> pa.Schema:
    """Schema fixo do wide: id_hexagono (str), data (timestamp), stats float64."""
    cols = full_wide_columns()
    fields = [pa.field("id_hexagono", pa.string()), pa.field("data", pa.timestamp("ns"))]
    for c in cols[2:]:
        fields.append(pa.field(c, pa.float64()))
    return pa.schema(fields)


def _aligned(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Reordena/completa colunas e faz cast para o schema alvo."""
    cols = [f.name for f in schema]
    arrays = []
    for name in cols:
        if name in table.column_names:
            arrays.append(table.column(name))
        else:
            arrays.append(pa.nulls(table.num_rows, type=schema.field(name).type))
    return pa.Table.from_arrays(arrays, names=cols).cast(schema)


def merge_to_canonical(canonical_path, part_paths, out_path) -> int:
    """
    Escreve ``out_path`` = ``canonical_path`` (se existir) + todas as
    ``part_paths``. Retorna o total de linhas escritas.
    """
    schema = target_schema()
    sources = []
    if canonical_path and Path(canonical_path).exists() and Path(canonical_path).stat().st_size:
        sources.append(Path(canonical_path))
    sources += [Path(p) for p in part_paths if Path(p).exists() and Path(p).stat().st_size]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(out_path, schema)
    total = 0
    try:
        for src in sources:
            pf = pq.ParquetFile(src)
            for i in range(pf.num_row_groups):
                table = _aligned(pf.read_row_group(i), schema)
                if table.num_rows:
                    writer.write_table(table)
                    total += table.num_rows
    finally:
        writer.close()
    return total

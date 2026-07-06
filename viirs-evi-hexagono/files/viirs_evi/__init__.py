"""
viirs_evi — extração incremental de séries de EVI do VIIRS (VNP13A1) por
hexágono H3, mascaradas pela macroclasse Agricultura do MapBiomas.

Versão de produção da rotina ``fenologia``, adaptada para rodar via kbatch num
pod (disco não-persistente) com:
- download prévio das máscaras MapBiomas usadas no run;
- acesso S3 direto ao VIIRS (in-region us-west-2);
- processamento **incremental**: lê o ``evi_brazil.parquet`` existente e só
  processa as datas/hexágonos que faltam até a última composição VIIRS
  disponível;
- entrada/saída via Azure Blob Storage.
"""

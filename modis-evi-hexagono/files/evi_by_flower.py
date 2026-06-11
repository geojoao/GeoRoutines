if __name__ == '__main__':
    import subprocess
    import sys
    import os
    

    print(
    os.getcwd(),
    os.listdir()
    )

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r","requirements.txt"])

    import time
    import ast
    import json

    import salem
    import shutil
    os.environ['USE_PYGEOS'] = '0'
    import pyproj
    import rasterio
    import rioxarray
    import xarray as xr
    import planetary_computer
    import pystac_client
    import stackstac
    import pandas as pd
    import geopandas as gpd
    import numpy as np
    from scipy import stats as st
    from shapely import wkt,geometry
    from shapely.geometry import Polygon, MultiPolygon, mapping, shape
    from dask.distributed import Client
    import datetime, shapely
    import zipfile
    import dask
    import dask.dataframe as dd
    from dask.distributed import Client, LocalCluster, progress
    # from tqdm import tqdm
    import os
    from azure.storage.blob import BlobServiceClient, BlobBlock ,generate_blob_sas, BlobSasPermissions,BlobClient
    from azure.core.exceptions import ServiceResponseError
    import json
    from h3 import h3
    from itertools import chain
    from dask.distributed import Client, LocalCluster
    from multiprocessing import Process, Queue
    import psutil


    import subprocess
    import sys
    import warnings
    warnings.filterwarnings("ignore")

    try:
        terras_file_name = sys.argv[1]
        output_name_space = sys.argv[2]
    except IndexError:
        terras_file_name = 'terras.csv'
        output_name_space = ''

    # terras_file_name = 'TERRAS_ESTEIRA_CARTEIRA_GRUPO_PAUTADO.csv'
    # output_name_space = 'ESTEIRA_CARTEIRA_GRUPO_PAUTADO'
    
    print(f"INPUT_FILE_NAME={terras_file_name}")
    print(f"OUTPUT_NAME_SPACE={output_name_space}")

    ############################################################################################
    ########################## Funções para baixar e upar arquivos no blob, ####################
    ########################## retornar os grupos a serem atualizados ##########################
    ########################## e clipar rasters (que n está sendo utilizada, ###################
    ########################## estou usando salem) #############################################

    def upa_file_blob(local_filepath=str, blob_filepath=str, delete_original_file=bool, blob_container=str):
        # Set the connection string to your Azure Storage account
        connect_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]

        # Set the name of the container you want to create or upload to
        # container_name = "rain-routine"
        container_name = blob_container

        # Create a BlobServiceClient object using the connection string
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)

        # connect to the container
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_filepath)

        # checa se o arquivo (blob) já existe online, se já existir vc exclui ele.
        if blob_client.exists():
            # Delete the blob if it already exists
            blob_client.delete_blob()

        #Upload the file to Azure Blob Storage
        try:
            with open(local_filepath, "rb") as data:
                blob_client.upload_blob(data)
                # print(f"[Sucesso] O arquivo {local_filepath} foi upado para o blob no container {container_name} e local {blob_filepath}")
                if delete_original_file:
                    os.remove(local_filepath)
        except:
            pass



    ######## Função para baixar um arquivo do blob
    def baixa_file_blob(file_name=str, blob_container=str):
        # Set the connection string to your Azure Storage account
        connect_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]

        # Set the name of the container you want to create or upload to
        # container_name = "rain-routine"
        container_name = blob_container

        # Create a BlobServiceClient object using the connection string
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)

        # connect to the container
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)

        # checa se o arquivo (blob) já existe online, se já existir vc tenta baixar.
        if blob_client.exists():
            # Download the file from Azure Blob Storage
            with open(file_name, "wb") as download_file:
                download_stream = blob_client.download_blob()
                download_file.write(download_stream.readall())
                print(f"[Sucesso] O arquivo {file_name} foi baixado para o diretório local")
        else:
            print(f"[Falha] O arquivo {file_name} não existe no blob")


    ############################################################################################
    ########################## Seleciona os aois a serem processados ###########################
    ########################## baixa o shape com todas as geometrias do brasil #################
    ########################## baixa os json com os produtores desejados #######################
    ########################## pega somente os aois que tem produtores que estao no json #######
    ############################################################################################

    # Set the connection string to your Azure Storage account
    connect_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    # Set the name of the container you want to create or upload to
    container_name = "planetary-routines-input"
    # Set the name of the file you want to upload
    file_name = terras_file_name
    # Set the path to the file you want to upload
    file_path = terras_file_name
    # Create a BlobServiceClient object using the connection string
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    # Get a BlobClient object for the file you want to download
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)
    print(datetime.datetime.now().strftime('%Y-%m-%d %T')+" | "+f"Downloading {file_name} from blob")
    # Download the file from Azure Blob Storage
    with open(file_path, "wb") as download_file:
        download_stream = blob_client.download_blob()
        download_file.write(download_stream.readall())
    print(datetime.datetime.now().strftime('%Y-%m-%d %T')+" | "+f"File '{file_name}' downloaded from container '{container_name}' and saved to '{file_path}'")


    def tryParseWKT(x):
        try:
            return wkt.loads(x)
        except:
            print('Geometria inválida!')
            return None

    df = pd.read_csv(terras_file_name)
    df['geometry'] = df['GEOM_WKT'].apply(lambda x: tryParseWKT(x))
    df = df[pd.notnull(df['geometry'])].drop(columns=['GEOM_WKT'])

    df.rename(columns = {col:col.lower() for col in df.columns}, inplace = True)

    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf.set_geometry(gdf.geometry) 
    gdf.crs=pyproj.CRS('EPSG:4326')

    ################################# seleciona os aois

    aois = gdf.id_hexagono.unique()
    ############################################################################################
    ########################## Pega todos os hexagonos de interesse + ##########################
    ########################## todos os hexagonos ao redor (1 nível de distância) ##############
    ############################################################################################

    dict_aois = {}
    for aoi in aois:
        aois_vizinhos = list(h3.k_ring(aoi,1))
        
        hex_geometries = []
        for aoi_i in aois_vizinhos:
            hex_geom = Polygon(h3.h3_to_geo_boundary(str(aoi_i), geo_json=True))
            hex_geometries.append(hex_geom)

        hex_geometries_df = gpd.GeoDataFrame(data=aois_vizinhos, geometry=hex_geometries)
        hex_geometries_df.columns=['hex','geometry']
        dict_aois[aoi] = hex_geometries_df




    ############################################################################################
    ########################## função que itera por "flor" de 7 hexagonos ######################
    ########################## baixa e une os tiles que tocam essa flor ########################
    ########################## e ai, pra cada hexagono, cria um cubo de evi ####################
    ########################## e concatena todos eles em bandas diferentes #####################
    ########################## então cada resultado disso aqui tem 7 bandas (1 por hex) ########
    ############################################################################################
        
    def retorna_evi(geometry_df):
        geometry_df['geometry'] = geometry_df['geometry'].make_valid()
        dissolved_geometry = geometry_df.dissolve().geometry[0]
        geometry_bounds = tuple(np.array(dissolved_geometry.bounds) + [-0.005, -0.005, 0.005, 0.005]) # para que não fiquem partes da geometria fora, + 550m em cada bound

        now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        daterange=["2010-01-01T00:00:00Z",now]
        catalog = pystac_client.Client.open(
            #"https://planetarycomputer.microsoft.com/api/stac/v1",
            "https://data.inpe.br/bdc/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

        search = catalog.search(
            #collections=["modis-13Q1-061"],
            collections=["mod13q1-6.1"],
            datetime=daterange,
            intersects = dissolved_geometry
        )
        items = search.item_collection()
        vegetation_indexes_da = (
            stackstac.stack(
                items,
                epsg=4326,
                assets=["EVI"],  # 250m_16_days_EVI
                chunksize=256,  # set chunk size to 256 to get one chunk per time step
                resolution=0.0025,
                bounds=geometry_bounds
            )
        )
        
        vegetation_indexes_da = vegetation_indexes_da.groupby('end_datetime').mean()
        
        list_of_arrays = []
        for hexagono in geometry_df.hex:
            sub_df = geometry_df[geometry_df['hex']==hexagono]
            array = vegetation_indexes_da.salem.roi(geometry=sub_df.geometry.values[0])
            array['hexagono'] = hexagono
            list_of_arrays.append(array)
            
        concatenado = xr.concat(list_of_arrays, dim='band')
        return concatenado.compute()


            
    ############################################################################################
    ########################## dado um evi_array (time,lat,lon), proveniente ###################
    ########################## de uma das bandas da função "retorna_evi" #######################
    ########################## calcula o evi e a área de cada classe do mapbiomas ##############
    ########################## dado pelo array "classificação_rescaled", que deve ##############
    ########################## estar na mesma resolução evi_array ##############################
    ############################################################################################

    def cria_evi_series_old(classificacao_rescaled, evi_array):
        
        desired_classes = [15,39,20,40,62,41,46,47,35,48,9,21]
        class_names = {15:'pastagem',
                    39:'soja',
                    20:'cana',
                    40:'arroz',
                    62:'algodao',
                    41:'plantacao_temporaria',
                    46:'cafe',
                    47:'citrus',
                    35:'dende',
                    48:'plantacoes_perenes',
                    9:'silvicultura',
                    21:'mosaico_de_usos'}
        
        evi_dataset = evi_array.to_dataset(name='evi')
        lista_dfs_classes = []
        for classe in desired_classes:
            nome_classe = class_names[classe]
            evi_dataset_classe_i = evi_dataset.where(classificacao_rescaled['classes']==classe)
            df_media_classe = evi_dataset_classe_i.mean(['x','y']).to_dataframe()
            df_media_classe = df_media_classe.reset_index()
            df_media_classe = df_media_classe.rename(columns={'end_datetime':'data_referencia'})
            df_media_classe['evi'] = df_media_classe.evi.fillna(0)
            df_media_classe['evi'] = df_media_classe['evi']/10000
            df_media_classe = df_media_classe.set_index(pd.to_datetime(df_media_classe.data_referencia))
            df_media_classe = df_media_classe[['hexagono','evi']]
            df_media_classe = df_media_classe.rename(columns={'hexagono':'id_hexagono', 'evi':'evi_'+nome_classe})  #{:02d}'.format(classe)
            # -3000 é o nodata do produto mod13q1v061, e os dados vão de -2000 a 10000
            df_media_classe['area_'+nome_classe] = (evi_dataset_classe_i.evi>-2500).sum(dim=['x','y']).values*250*250  ##{:02d}'.format(classe) 
            lista_dfs_classes.append(df_media_classe)
            
        concatenado = pd.concat(lista_dfs_classes, axis=1)
        concatenado = concatenado.loc[:,~concatenado.columns.duplicated()].copy()
        
        return concatenado




    def cria_evi_series(classificacao_rescaled, evi_array):

        dict_classes = {'pastagem':[15],
                    'soja':[39],
                    'cana':[20],
                    'arroz':[40],
                    'algodao':[62],
                    'outras_plantacoes_temporarias':[41],
                    'cafe':[46],
                    'citrus':[47],
                    'dende':[35],
                    'outras_plantacoes_perenes':[48],
                    'silvicultura':[9],
                    'mosaico_de_usos':[21],
                    'plantacoes_temporarias':[39,20,40,62,41],
                    'plantacoes_perenes':[46,47,35,48]}

        evi_dataset = evi_array.to_dataset(name='evi')
        lista_dfs_classes = []
        for nome_classe in list(dict_classes.keys()):
            lista_classe = dict_classes[nome_classe]
            evi_dataset_classe_i = evi_dataset.where(np.in1d(ar1=classificacao_rescaled['classes'].values.reshape([-1]), ar2=lista_classe).reshape(classificacao_rescaled['classes'].shape))
            
            
            if nome_classe == 'plantacoes_temporarias':
                
                df_media_classe = evi_dataset_classe_i.mean(['x','y']).to_dataframe()
                df_media_classe = df_media_classe.reset_index()
                df_media_classe = df_media_classe.rename(columns={'end_datetime':'data_referencia'})
                df_media_classe['evi'] = df_media_classe.evi.fillna(0)
                df_media_classe['evi'] = df_media_classe['evi']/10000
                df_media_classe = df_media_classe.set_index(pd.to_datetime(df_media_classe.data_referencia))
                df_media_classe = df_media_classe[['hexagono','evi']]
                df_media_classe = df_media_classe.rename(columns={'hexagono':'id_hexagono', 'evi':'evi_'+nome_classe})  #{:02d}'.format(classe)
                # -3000 é o nodata do produto mod13q1v061, e os dados vão de -2000 a 10000
                df_media_classe['area_'+nome_classe] = (evi_dataset_classe_i.evi>-2500).sum(dim=['x','y']).values*250*250  ##{:02d}'.format(classe) 
                
                df_evi_q80 = evi_dataset_classe_i.quantile(0.8, dim=['x','y']).to_dataframe()
                df_evi_q80 = df_evi_q80.reset_index()
                df_evi_q80 = df_evi_q80.rename(columns={'end_datetime':'data_referencia', 'evi':'q80_evi_'+nome_classe})
                df_evi_q80['q80_evi_'+nome_classe] = df_evi_q80['q80_evi_'+nome_classe].fillna(0)
                df_evi_q80['q80_evi_'+nome_classe] = df_evi_q80['q80_evi_'+nome_classe]/10000
                df_evi_q80 = df_evi_q80.set_index(pd.to_datetime(df_evi_q80.data_referencia))
                df_evi_q80 = pd.DataFrame(df_evi_q80['q80_evi_'+nome_classe])
                
                df_evi_q20 = evi_dataset_classe_i.quantile(0.2, dim=['x','y']).to_dataframe()
                df_evi_q20 = df_evi_q20.reset_index()
                df_evi_q20 = df_evi_q20.rename(columns={'end_datetime':'data_referencia', 'evi':'q20_evi_'+nome_classe})
                df_evi_q20['q20_evi_'+nome_classe] = df_evi_q20['q20_evi_'+nome_classe].fillna(0)
                df_evi_q20['q20_evi_'+nome_classe] = df_evi_q20['q20_evi_'+nome_classe]/10000
                df_evi_q20 = df_evi_q20.set_index(pd.to_datetime(df_evi_q20.data_referencia))
                df_evi_q20 = pd.DataFrame(df_evi_q20['q20_evi_'+nome_classe])
                
                df_evi_max = evi_dataset_classe_i.max(dim=['x','y']).to_dataframe()
                df_evi_max = df_evi_max.reset_index()
                df_evi_max = df_evi_max.rename(columns={'end_datetime':'data_referencia', 'evi':'max_evi_'+nome_classe})
                df_evi_max['max_evi_'+nome_classe] = df_evi_max['max_evi_'+nome_classe].fillna(0)
                df_evi_max['max_evi_'+nome_classe] = df_evi_max['max_evi_'+nome_classe]/10000
                df_evi_max = df_evi_max.set_index(pd.to_datetime(df_evi_max.data_referencia))
                df_evi_max = pd.DataFrame(df_evi_max['max_evi_'+nome_classe])

                df_evi_min = evi_dataset_classe_i.min(dim=['x','y']).to_dataframe()
                df_evi_min = df_evi_min.reset_index()
                df_evi_min = df_evi_min.rename(columns={'end_datetime':'data_referencia', 'evi':'min_evi_'+nome_classe})
                df_evi_min['min_evi_'+nome_classe] = df_evi_min['min_evi_'+nome_classe].fillna(0)
                df_evi_min['min_evi_'+nome_classe] = df_evi_min['min_evi_'+nome_classe]/10000
                df_evi_min = df_evi_min.set_index(pd.to_datetime(df_evi_min.data_referencia))
                df_evi_min = pd.DataFrame(df_evi_min['min_evi_'+nome_classe])
                
                df_uniao = pd.concat([df_media_classe, df_evi_q80, df_evi_q20, df_evi_max, df_evi_min], axis=1)
                
                lista_dfs_classes.append(df_uniao)
                
                
            else:
                df_media_classe = evi_dataset_classe_i.mean(['x','y']).to_dataframe()
                df_media_classe = df_media_classe.reset_index()
                df_media_classe = df_media_classe.rename(columns={'end_datetime':'data_referencia'})
                df_media_classe['evi'] = df_media_classe.evi.fillna(0)
                df_media_classe['evi'] = df_media_classe['evi']/10000
                df_media_classe = df_media_classe.set_index(pd.to_datetime(df_media_classe.data_referencia))
                df_media_classe = df_media_classe[['hexagono','evi']]
                df_media_classe = df_media_classe.rename(columns={'hexagono':'id_hexagono', 'evi':'evi_'+nome_classe})  #{:02d}'.format(classe)
                # -3000 é o nodata do produto mod13q1v061, e os dados vão de -2000 a 10000
                df_media_classe['area_'+nome_classe] = (evi_dataset_classe_i.evi>-2500).sum(dim=['x','y']).values*250*250  ##{:02d}'.format(classe) 
                lista_dfs_classes.append(df_media_classe)
            
            
        concatenado = pd.concat(lista_dfs_classes, axis=1)
        concatenado = concatenado.loc[:,~concatenado.columns.duplicated()].copy()
        
        return concatenado


    ############################################################################################
    ########################## Itera entre os aois centrais de cada flor #######################
    ########################## cria um .parquet com a série (por calsse) da cada ###############
    ########################## aoi de cada flor, gera também a série de área de cada classe ####
    ########################## upa esses .parquets por aoi no blob #############################
    ############################################################################################

    print('######## Baixando dados do mapbiomas ##########')
    year = 2024
    da_mapbiomas = rioxarray.open_rasterio(f'https://storage.googleapis.com/mapbiomas-public/initiatives/brasil/collection_10/lulc/coverage/brazil_coverage_{year}.tif').squeeze(dim='band')
    mapbiomas_dataset = da_mapbiomas.to_dataset(name='classes')
    # da_mapbiomas = da_mapbiomas.to_dataset(name='classes')

        
    def processa_flor(geometry_df, mabiomas_dataset, aoi_central):
        
        hexagonos_processados_inprocess = []
        
        if geometry_df.shape[0]>1: #casp ainda tenha ao menos 2 hexagonos a serem processados, faz normal
            # try:
            #     dataset = retorna_evi(geometry_df)
            # except Exception as e:
            #     print(f"[erro] O dataset do hexagono central {aoi_central} não pôde ser recuperado na primeira tentativa, esperando 20s e tentando novamente...")
            #     time.sleep(30)
            #     dataset = retorna_evi(geometry_df)

            attempts = 0
            max_attempts = 5
            wait_time = 30
            while attempts < max_attempts:
                try:
                    dataset = retorna_evi(geometry_df)
                    break  # Exit the loop if successful
                except Exception as e:
                    print(f"    [erro] O dataset do hexagono central {aoi_central} não pôde ser recuperado na tentativa {attempts + 1}, esperando {wait_time}s e tentando novamente...")
                    attempts += 1
                    time.sleep(wait_time)
            if attempts == max_attempts:
                print(f"    [erro] Não foi possível obter o dataset do hexagono central {aoi_central} após {max_attempts} tentativas.")

            # corta e rescala mapbiomas
            dissolvido = geometry_df.dissolve().geometry[0]
            mapbiomas_dataset_flower = mapbiomas_dataset.sel(x=slice(dissolvido.bounds[0]-0.05, dissolvido.bounds[2]+0.05), y=slice(dissolvido.bounds[3]+0.05, dissolvido.bounds[1]-0.05))
            classificacao_rescaled = dataset[0,0,:,:].salem.lookup_transform(mapbiomas_dataset_flower, method=maximo) #rescala o mapbiomas baseado em 1 tempo/banda do modis

            lista_dados_hexagonos = []
            for position_hex in np.arange(0,dataset['hexagono'].shape[0]):
                hex_id = str(dataset['hexagono'][position_hex].values)
                array = dataset[:,position_hex,:,:]
                hexagonos_processados_inprocess.append(hex_id)
                lista_dados_hexagonos.append(cria_evi_series(classificacao_rescaled=classificacao_rescaled, evi_array=array))

            flor_concatenada = pd.concat(lista_dados_hexagonos, axis=0)
        
            flor_concatenada.reset_index().to_parquet(f'evi_{aoi_central}.parquet', index=False)

            upa_file_blob(local_filepath=f"evi_{aoi_central}.parquet",
                            blob_filepath=os.path.join(output_name_space, f"MODIS_EVI_HEXAGONO/evi_{aoi_central}.parquet"),
                            delete_original_file=True,blob_container='planetary-routines-output')
                
        
        elif geometry_df.shape[0]==1: #caso tenha só um, processa só ele
            # dataset = retorna_evi(geometry_df)
            attempts = 0
            max_attempts = 5
            wait_time = 30
            while attempts < max_attempts:
                try:
                    dataset = retorna_evi(geometry_df)
                    break  # Exit the loop if successful
                except Exception as e:
                    print(f"    [erro] O dataset do hexagono central {aoi_central} não pôde ser recuperado na tentativa {attempts + 1}, esperando {wait_time}s e tentando novamente...")
                    attempts += 1
                    time.sleep(wait_time)
            if attempts == max_attempts:
                print(f"    [erro] Não foi possível obter o dataset do hexagono central {aoi_central} após {max_attempts} tentativas.")

            # corta e rescala mapbiomas
            dissolvido = geometry_df.dissolve().geometry[0]
            mapbiomas_dataset_flower = mapbiomas_dataset.sel(x=slice(dissolvido.bounds[0]-0.05, dissolvido.bounds[2]+0.05), y=slice(dissolvido.bounds[3]+0.05, dissolvido.bounds[1]-0.05))
            classificacao_rescaled = dataset[0,0,:,:].salem.lookup_transform(mapbiomas_dataset_flower, method=maximo) #rescala o mapbiomas baseado em 1 tempo/banda do modis

            position_hex = 0
            hex_id = str(dataset['hexagono'].values)
            array = dataset[:,position_hex,:,:]
            hexagonos_processados_inprocess.append(hex_id)

            flor_unica = cria_evi_series(classificacao_rescaled=classificacao_rescaled, evi_array=array)
            
            flor_unica.reset_index().to_parquet(f'evi_{aoi_central}.parquet', index=False)

            upa_file_blob(local_filepath=f"evi_{aoi_central}.parquet",
                            blob_filepath=os.path.join(output_name_space, f"MODIS_EVI_HEXAGONO/evi_{aoi_central}.parquet"),
                            delete_original_file=True,blob_container='planetary-routines-output')
                
                
        else: #caso tenha nenhum exagono ainda não processado, não faz nada
            hexagonos_processados_inprocess.append([None])

        
        #bota a lista com os hexagonos processados nesse processo para fora dele
        result_queue.put(hexagonos_processados_inprocess)





    def maximo(x):
        """Return a scalar mode for array-like x, robust to scipy return shapes.

        Falls back to numpy/pandas methods when needed and returns np.nan if
        no valid value can be computed.
        """
        try:
            # scipy.stats.mode may return a scalar or array depending on version
            res = st.mode(x, nan_policy='omit')
            mode_val = res.mode
            # mode_val can be scalar or array-like
            if hasattr(mode_val, '__len__'):
                if len(mode_val) == 0:
                    raise ValueError("Empty mode result")
                return mode_val[0]
            else:
                return mode_val
        except Exception:
            arr = np.asarray(x).ravel()
            # remove nans
            if arr.size == 0:
                return np.nan
            arr = arr[~pd.isna(arr)]
            if arr.size == 0:
                return np.nan
            # If integer-like, use bincount for speed
            if np.issubdtype(arr.dtype, np.integer):
                vals, counts = np.unique(arr, return_counts=True)
                return vals[np.argmax(counts)]
            # else use pandas mode (handles floats/strings)
            mv = pd.Series(arr).mode()
            if mv.empty:
                return np.nan
            return mv.iat[0]


    def get_memory_usage():
        memory = psutil.virtual_memory()
        usage_mb = memory.used / (1024 * 1024)
        return usage_mb


    print('######## Processando as flores de hexagonos ##########')
    hexagonos_ja_processados = []
    lista_tempos_aois = []
    contador = 0

    max_attempts = 3
    current_attempt = 0

    for aoi_central in list(dict_aois.keys()):
        start_time_aoi_i = time.time()
        
        #arruma dataset evi
        geometry_df = dict_aois[aoi_central].query('hex != @hexagonos_ja_processados') #seleciona apenas os hexagonos ainda nao processados
        
        result_queue = Queue()
        
        while current_attempt < max_attempts:
            try:
                #função que processa//
                p = Process(target=processa_flor, args=(geometry_df, mapbiomas_dataset, aoi_central))
                p.start()
                p.join()
                p.close()

                #extrai os hexagonos processados nela
                output_hexagonos_processados_inprocess = result_queue.get(block=True, timeout=5)
                for id_hexagono_processado in output_hexagonos_processados_inprocess:
                    hexagonos_ja_processados.append(id_hexagono_processado)

                # concatena os tempos
                lista_tempos_aois.append((time.time() - start_time_aoi_i))
                contador = contador + 1
                print(f"[{aoi_central}] aoi {str(contador)} de {str(len(list(dict_aois.keys())))}, tempo médio por aoi até agora: {str(np.mean(lista_tempos_aois))}. Estimado faltante:{str((np.mean(lista_tempos_aois)*((len(list(dict_aois.keys()))+1)-contador))/60)}") 
                memory_usage = get_memory_usage()
                print(f"Memory Usage: {memory_usage} MB")
                
                # Sai do loop caso o processo seja concluído com sucesso
                break
            except Exception as e:
                current_attempt += 1
                print(f"Erro ao processar: {e}")
                print(f"Tentativa {current_attempt} de {max_attempts}")
        
        # Reinicia a variável de tentativa para o próximo aoi_central
        current_attempt = 0
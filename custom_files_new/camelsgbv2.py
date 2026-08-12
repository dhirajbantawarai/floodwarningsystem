from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from functools import lru_cache
import logging
import os

import numpy as np

import pandas as pd
import xarray
import yaml

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)

DEFAULT_GW_FALLBACK_MAX_DISTANCE_KM = 20.0


class CamelsGBV2(BaseDataset):
    """Dataset class for CAMELS_GB_V2 which includes groundwater timeseries in addition to hydromet.

    Same constructor contract as other datasets (see `BaseDataset`).
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        super(CamelsGBV2, self).__init__(cfg=cfg,
                                         is_train=is_train,
                                         period=period,
                                         basin=basin,
                                         additional_features=additional_features,
                                         id_to_int=id_to_int,
                                         scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load hydrometeorological and groundwater time series for a single basin.

        This loader looks for hydromet files under `timeseries/` with names containing
        the basin id and also optionally a groundwater timeseries file. If groundwater
        exists, its columns are appended (prefixed with `gw_` to avoid name clashes).
        """
        gw_data_frequency = self.cfg.gw_data_frequency
        train_start = self.cfg.train_start_date
        train_end = self.cfg.train_end_date
        return load_camels_gb_v2_timeseries(
            data_dir=self.cfg.data_dir,
            basin=basin,
            gw_data_frequency=gw_data_frequency,
            train_start_date=train_start,
            train_end_date=train_end
        )

    def _load_attributes(self) -> pd.DataFrame:
        return load_camels_gb_v2_attributes(self.cfg.data_dir, basins=self.basins)


def load_camels_gb_v2_attributes(data_dir: Path, basins: List[str] = []) -> pd.DataFrame:
    """Load CAMELS-GB-V2 attributes.

    The V2 attributes are expected under `attributes/` as CSV files ending with `_attributes.csv`,
    same as CAMELS-GB. This function mirrors `load_camels_gb_attributes` but is separated for clarity.
    """
    attributes_path = Path(data_dir) / 'attributes'

    if not attributes_path.exists():
        raise FileNotFoundError(f"Attribute folder not found at {attributes_path}")

    csv_files = attributes_path.glob('*_attributes.csv')

    dfs = []
    loaded_files: List[str] = []
    skipped_files: List[str] = []
    for csv_file in csv_files:
        df_temp = pd.read_csv(csv_file, sep=',', header=0)

        # Skip files that are not indexed by gauge_id (e.g., groundwater-well attributes indexed by gw_well_id).
        if 'gauge_id' not in df_temp.columns:
            skipped_files.append(csv_file.name)
            continue

        df_temp['gauge_id'] = df_temp['gauge_id'].astype(str)
        df_temp = df_temp.set_index('gauge_id')
        dfs.append(df_temp)
        loaded_files.append(csv_file.name)

    LOGGER.info(f"CAMELS_GB_V2 attribute files loaded ({len(loaded_files)}): {loaded_files}")
    if skipped_files:
        LOGGER.info(f"CAMELS_GB_V2 attribute files skipped (no gauge_id index, {len(skipped_files)}): {skipped_files}")

    if not dfs:
        raise FileNotFoundError(f'No attribute files found in {attributes_path}')

    df = pd.concat(dfs, axis=1)

    # Merge groundwater well attributes if available
    gw_attr_file = Path(data_dir) / 'attributes' / 'camels_gb_v2_groundwaterwell_attributes.csv'
    mapping_file = _find_wells_to_catchment_mapping_file(data_dir)
    if gw_attr_file.exists() and mapping_file is not None:
        gw_attr_df = pd.read_csv(gw_attr_file, sep=',', dtype={'gw_well_id': str})
        gw_attr_df['gw_well_id'] = gw_attr_df['gw_well_id'].apply(_normalize_well_id)
        
        # Encode categorical aquifer
        if 'aquifer' in gw_attr_df.columns:
            gw_attr_df['gw_aquifer_code'] = gw_attr_df['aquifer'].astype('category').cat.codes.astype(float)
        
        # Ensure all other numeric attributes are parsed properly
        for col in ['gw_well_depth', 'gw_well_datum', 'gw_well_easting', 'gw_well_northing']:
            if col in gw_attr_df.columns:
                gw_attr_df[col] = pd.to_numeric(gw_attr_df[col], errors='coerce')
        
        # Define cols to keep
        cols_to_keep = ['gw_well_id', 'gw_well_depth', 'gw_well_datum', 'gw_well_easting', 'gw_well_northing', 'gw_aquifer_code']
        cols_to_keep = [c for c in cols_to_keep if c in gw_attr_df.columns]
        gw_attr_df = gw_attr_df[cols_to_keep]
        
        # Load mapping file
        mapping_df = pd.read_csv(mapping_file, dtype={'ID_STRING': str, 'gw_well_id': str})
        mapping_df['ID_STRING'] = mapping_df['ID_STRING'].apply(_normalize_basin_id)
        mapping_df['gw_well_id'] = mapping_df['gw_well_id'].apply(_normalize_well_id)
        
        # Merge mapping with well attributes
        merged_gw = pd.merge(mapping_df, gw_attr_df, on='gw_well_id', how='inner')
        merged_gw = merged_gw.drop(columns=['gw_well_id']).drop_duplicates(subset=['ID_STRING']).set_index('ID_STRING')
        
        # Align with the main catchment attributes dataframe
        merged_gw = merged_gw.reindex(df.index)
        
        # Fill missing values (for basins with no mapped well) with the column mean
        for col in merged_gw.columns:
            col_mean = merged_gw[col].mean()
            if pd.isna(col_mean):
                col_mean = 0.0
            merged_gw[col] = merged_gw[col].fillna(col_mean)
            
        # Concatenate with the main attributes dataframe
        df = pd.concat([df, merged_gw], axis=1)

    if basins:
        if any(b not in df.index for b in basins):
            raise ValueError('Some basins are missing static attributes.')
        df = df.loc[basins]

    return df


def _normalize_well_id(well_id: str) -> str:
    return str(well_id).strip().upper()


def _normalize_basin_id(basin_id: str) -> str:
    return str(basin_id).strip()


def _get_gw_fallback_max_distance_km() -> float | None:
    raw_value = os.getenv('NH_GW_FALLBACK_MAX_DISTANCE_KM', str(DEFAULT_GW_FALLBACK_MAX_DISTANCE_KM)).strip()
    if raw_value.lower() in {'', 'none', 'null', 'nan', '-1'}:
        return None

    try:
        value = float(raw_value)
    except ValueError:
        LOGGER.warning(
            "Invalid NH_GW_FALLBACK_MAX_DISTANCE_KM=%s. Falling back to default %.1f km.",
            raw_value,
            DEFAULT_GW_FALLBACK_MAX_DISTANCE_KM,
        )
        return DEFAULT_GW_FALLBACK_MAX_DISTANCE_KM

    if value <= 0:
        return None
    return value


@lru_cache(maxsize=8)
def _load_groundwater_well_attributes(data_dir: Path) -> pd.DataFrame:
    gw_attr_file = Path(data_dir) / 'attributes' / 'camels_gb_v2_groundwaterwell_attributes.csv'
    if not gw_attr_file.exists():
        raise FileNotFoundError(f"Groundwater well attribute file not found at {gw_attr_file}")

    gw_df = pd.read_csv(gw_attr_file, sep=',', header=0, dtype={'gw_well_id': str})
    required_cols = {'gw_well_id', 'gw_well_easting', 'gw_well_northing'}
    missing = required_cols - set(gw_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in groundwater well attributes: {missing}")

    gw_df['gw_well_id'] = gw_df['gw_well_id'].apply(_normalize_well_id)
    gw_df['gw_well_easting'] = pd.to_numeric(gw_df['gw_well_easting'], errors='coerce')
    gw_df['gw_well_northing'] = pd.to_numeric(gw_df['gw_well_northing'], errors='coerce')
    gw_df = gw_df.dropna(subset=['gw_well_easting', 'gw_well_northing'])

    return gw_df[['gw_well_id', 'gw_well_easting', 'gw_well_northing']].drop_duplicates()


@lru_cache(maxsize=8)
def _find_groundwater_daily_dir(data_dir: Path) -> Path | None:
    candidate_dirs = [
        Path(data_dir) / 'Catchment_Timeseries' / 'groundwater' / 'daily',
        Path(data_dir) / 'groundwater' / 'daily',
        Path(data_dir) / 'timeseries' / 'groundwater' / 'daily'
    ]
    for directory in candidate_dirs:
        if directory.is_dir():
            return directory
    return None


@lru_cache(maxsize=8)
def _find_groundwater_monthly_dir(data_dir: Path) -> Path | None:
    candidate_dirs = [
        Path(data_dir) / 'Catchment_Timeseries' / 'groundwater' / 'monthly',
        Path(data_dir) / 'groundwater' / 'monthly',
        Path(data_dir) / 'timeseries' / 'groundwater' / 'monthly'
    ]
    for directory in candidate_dirs:
        if directory.is_dir():
            return directory
    return None


@lru_cache(maxsize=8)
def _find_wells_to_catchment_mapping_file(data_dir: Path) -> Path | None:
    candidates = [
        Path(data_dir) / 'wells_to_catchment_mapping.csv',
        Path(data_dir) / 'attributes' / 'wells_to_catchment_mapping.csv',
        # Common location used in this workspace.
        Path(__file__).resolve().parents[2] / 'examples' / '01-Introduction' / 'wells_to_catchment_mapping.csv'
    ]
    for file_path in candidates:
        if file_path.is_file():
            return file_path
    return None


@lru_cache(maxsize=8)
def _get_available_groundwater_well_files(daily_dir: Path) -> Dict[str, Path]:
    # Expected name pattern:
    # camels_gb_v2_groundwater_daily_timeseries_<well_id>_<start>-<end>.csv
    file_map: Dict[str, Path] = {}
    for file_path in daily_dir.glob('camels_gb_v2_groundwater_daily_timeseries_*.csv'):
        stem = file_path.stem
        marker = 'camels_gb_v2_groundwater_daily_timeseries_'
        if not stem.startswith(marker):
            continue
        remainder = stem[len(marker):]
        # Split from the right once to keep well IDs that already contain underscores.
        try:
            well_id_raw, _ = remainder.rsplit('_', 1)
        except ValueError:
            continue
        file_map[_normalize_well_id(well_id_raw)] = file_path
    return file_map


@lru_cache(maxsize=8)
def _get_available_groundwater_monthly_files(monthly_dir: Path) -> Dict[str, Path]:
    # Expected name pattern:
    # camels_gb_v2_groundwater_monthly_timeseries_<well_id>_<start>-<end>.csv
    file_map: Dict[str, Path] = {}
    for file_path in monthly_dir.glob('camels_gb_v2_groundwater_monthly_timeseries_*.csv'):
        stem = file_path.stem
        marker = 'camels_gb_v2_groundwater_monthly_timeseries_'
        if not stem.startswith(marker):
            continue
        remainder = stem[len(marker):]
        try:
            well_id_raw, _ = remainder.rsplit('_', 1)
        except ValueError:
            continue
        file_map[_normalize_well_id(well_id_raw)] = file_path
    return file_map


@lru_cache(maxsize=16)
def _build_basin_to_well_map(data_dir: Path, fallback_max_distance_km: float | None, gw_data_frequency: str = 'daily') -> Dict[str, str]:
    topographic_file = Path(data_dir) / 'attributes' / 'camels_gb_v2_topographic_attributes.csv'
    if not topographic_file.exists():
        raise FileNotFoundError(f"Topographic attributes not found at {topographic_file}")

    basin_df = pd.read_csv(topographic_file, sep=',', header=0, dtype={'gauge_id': str})
    required_basin_cols = {'gauge_id', 'gauge_easting', 'gauge_northing'}
    missing_basin_cols = required_basin_cols - set(basin_df.columns)
    if missing_basin_cols:
        raise ValueError(f"Missing required basin columns: {missing_basin_cols}")

    basin_df['gauge_easting'] = pd.to_numeric(basin_df['gauge_easting'], errors='coerce')
    basin_df['gauge_northing'] = pd.to_numeric(basin_df['gauge_northing'], errors='coerce')
    basin_df = basin_df.dropna(subset=['gauge_easting', 'gauge_northing'])

    gw_df = _load_groundwater_well_attributes(data_dir)

    # Determine well mapping depending on preferred frequency
    if gw_data_frequency == 'daily':
        daily_dir = _find_groundwater_daily_dir(data_dir)
        if daily_dir is None:
            return {}
        well_to_file = _get_available_groundwater_well_files(daily_dir)
    elif gw_data_frequency == 'monthly':
        monthly_dir = _find_groundwater_monthly_dir(data_dir)
        if monthly_dir is None:
            return {}
        well_to_file = _get_available_groundwater_monthly_files(monthly_dir)
    else:
        # 'any': prefer daily, fall back to monthly if daily not available
        daily_dir = _find_groundwater_daily_dir(data_dir)
        daily_wells = _get_available_groundwater_well_files(daily_dir) if daily_dir is not None else {}
        monthly_dir = _find_groundwater_monthly_dir(data_dir)
        monthly_wells = _get_available_groundwater_monthly_files(monthly_dir) if monthly_dir is not None else {}
        well_to_file = {**monthly_wells, **daily_wells}

    if not well_to_file:
        return {}

    # Restrict mapping to wells for which timeseries files exist.
    gw_df = gw_df[gw_df['gw_well_id'].isin(well_to_file.keys())]
    if gw_df.empty:
        return {}

    basin_to_well: Dict[str, str] = {}

    # 1) Prefer explicit point-in-polygon mapping if available.
    mapping_file = _find_wells_to_catchment_mapping_file(data_dir)
    if mapping_file is not None:
        mapping_df = pd.read_csv(mapping_file, dtype={'ID_STRING': str, 'gw_well_id': str})
        if {'ID_STRING', 'gw_well_id'}.issubset(mapping_df.columns):
            useful_cols = ['ID_STRING', 'gw_well_id'] + [
                c for c in ['gw_well_easting', 'gw_well_northing', 'daily_gwlevel_perc_complete'] if c in mapping_df.columns
            ]
            mapping_df = mapping_df[useful_cols].copy()
            mapping_df['ID_STRING'] = mapping_df['ID_STRING'].apply(_normalize_basin_id)
            mapping_df['gw_well_id'] = mapping_df['gw_well_id'].apply(_normalize_well_id)

            # Keep only wells with available files.
            mapping_df = mapping_df[mapping_df['gw_well_id'].isin(well_to_file.keys())]

            if not mapping_df.empty:
                basin_lookup = basin_df.set_index('gauge_id')[['gauge_easting', 'gauge_northing']]
                has_well_xy = {'gw_well_easting', 'gw_well_northing'}.issubset(mapping_df.columns)

                if has_well_xy:
                    mapping_df['gw_well_easting'] = pd.to_numeric(mapping_df['gw_well_easting'], errors='coerce')
                    mapping_df['gw_well_northing'] = pd.to_numeric(mapping_df['gw_well_northing'], errors='coerce')
                if 'daily_gwlevel_perc_complete' in mapping_df.columns:
                    mapping_df['daily_gwlevel_perc_complete'] = pd.to_numeric(
                        mapping_df['daily_gwlevel_perc_complete'], errors='coerce')

                for basin_id, group in mapping_df.groupby('ID_STRING'):
                    if basin_id not in basin_lookup.index:
                        continue

                    if has_well_xy and group[['gw_well_easting', 'gw_well_northing']].notna().all(axis=1).any():
                        bx = basin_lookup.at[basin_id, 'gauge_easting']
                        by = basin_lookup.at[basin_id, 'gauge_northing']
                        valid = group.dropna(subset=['gw_well_easting', 'gw_well_northing'])
                        d2_local = (valid['gw_well_easting'].to_numpy() - bx) ** 2 + (valid['gw_well_northing'].to_numpy() - by) ** 2
                        chosen_well = valid.iloc[int(np.argmin(d2_local))]['gw_well_id']
                    elif 'daily_gwlevel_perc_complete' in group.columns and group['daily_gwlevel_perc_complete'].notna().any():
                        chosen_well = group.sort_values('daily_gwlevel_perc_complete', ascending=False).iloc[0]['gw_well_id']
                    else:
                        chosen_well = group.sort_values('gw_well_id').iloc[0]['gw_well_id']

                    basin_to_well[basin_id] = chosen_well

                LOGGER.info(f"CAMELS_GB_V2 groundwater mapping from wells_to_catchment CSV: {len(basin_to_well)} basins ({gw_data_frequency})")

    return basin_to_well


def get_groundwater_mapping_diagnostics(data_dir: Path,
                                        basins: List[str],
                                        fallback_max_distance_km: float | None = None,
                                        gw_data_frequency: str = 'daily') -> pd.DataFrame:
    """Return basin-level groundwater mapping diagnostics.

    Output columns:
    - basin
    - mapped_well
    - source: one of {csv, fallback, csv_missing_daily, csv_unmapped, none}
    - distance_km
    """
    if fallback_max_distance_km is None:
        fallback_max_distance_km = _get_gw_fallback_max_distance_km()

    basin_to_well = _build_basin_to_well_map(data_dir, fallback_max_distance_km, gw_data_frequency)

    mapping_file = _find_wells_to_catchment_mapping_file(data_dir)
    
    if gw_data_frequency == 'daily':
        daily_dir = _find_groundwater_daily_dir(data_dir)
        well_file_map = _get_available_groundwater_well_files(daily_dir) if daily_dir is not None else {}
    elif gw_data_frequency == 'monthly':
        monthly_dir = _find_groundwater_monthly_dir(data_dir)
        well_file_map = _get_available_groundwater_monthly_files(monthly_dir) if monthly_dir is not None else {}
    else:
        daily_dir = _find_groundwater_daily_dir(data_dir)
        daily_wells = _get_available_groundwater_well_files(daily_dir) if daily_dir is not None else {}
        monthly_dir = _find_groundwater_monthly_dir(data_dir)
        monthly_wells = _get_available_groundwater_monthly_files(monthly_dir) if monthly_dir is not None else {}
        well_file_map = {**monthly_wells, **daily_wells}

    csv_basin_set = set()
    csv_raw_basin_set = set()
    csv_missing_daily_basin_set = set()
    if mapping_file is not None:
        mdf = pd.read_csv(mapping_file, dtype={'ID_STRING': str, 'gw_well_id': str})
        if {'ID_STRING', 'gw_well_id'}.issubset(mdf.columns):
            mdf['ID_STRING'] = mdf['ID_STRING'].astype(str).str.strip()
            mdf['gw_well_id'] = mdf['gw_well_id'].astype(str).str.strip().str.upper()
            csv_raw_basin_set = set(mdf['ID_STRING'].unique())

            mdf_has_daily = mdf[mdf['gw_well_id'].isin(well_file_map.keys())]
            csv_basin_set = set(mdf_has_daily['ID_STRING'].unique())

            csv_missing_daily_basin_set = csv_raw_basin_set - csv_basin_set

    topo = pd.read_csv(Path(data_dir) / 'attributes' / 'camels_gb_v2_topographic_attributes.csv', dtype={'gauge_id': str})
    wells = pd.read_csv(Path(data_dir) / 'attributes' / 'camels_gb_v2_groundwaterwell_attributes.csv', dtype={'gw_well_id': str})

    topo = topo[['gauge_id', 'gauge_easting', 'gauge_northing']].copy()
    topo['gauge_easting'] = pd.to_numeric(topo['gauge_easting'], errors='coerce')
    topo['gauge_northing'] = pd.to_numeric(topo['gauge_northing'], errors='coerce')
    topo = topo.dropna(subset=['gauge_easting', 'gauge_northing'])

    wells['gw_well_id'] = wells['gw_well_id'].astype(str).str.strip().str.upper()
    wells = wells[['gw_well_id', 'gw_well_easting', 'gw_well_northing']].copy()
    wells['gw_well_easting'] = pd.to_numeric(wells['gw_well_easting'], errors='coerce')
    wells['gw_well_northing'] = pd.to_numeric(wells['gw_well_northing'], errors='coerce')
    wells = wells.dropna(subset=['gw_well_easting', 'gw_well_northing'])

    topo_lookup = topo.set_index('gauge_id')[['gauge_easting', 'gauge_northing']].to_dict('index')
    well_lookup = wells.set_index('gw_well_id')[['gw_well_easting', 'gw_well_northing']].to_dict('index')

    diag_rows: List[Dict[str, Any]] = []
    for basin in basins:
        mapped_well = basin_to_well.get(basin)
        if mapped_well is None:
            if basin in csv_missing_daily_basin_set:
                source = 'csv_missing_daily'
            elif basin in csv_raw_basin_set:
                source = 'csv_unmapped'
            else:
                source = 'none'
            dist_km = np.nan
        else:
            source = 'csv' if basin in csv_basin_set else 'fallback'
            if basin in topo_lookup and mapped_well in well_lookup:
                bx, by = topo_lookup[basin]['gauge_easting'], topo_lookup[basin]['gauge_northing']
                wx, wy = well_lookup[mapped_well]['gw_well_easting'], well_lookup[mapped_well]['gw_well_northing']
                dist_km = float(np.sqrt((bx - wx) ** 2 + (by - wy) ** 2) / 1000.0)
            else:
                dist_km = np.nan

        diag_rows.append({
            'basin': basin,
            'mapped_well': mapped_well if mapped_well is not None else 'NONE',
            'source': source,
            'distance_km': dist_km,
        })

    return pd.DataFrame(diag_rows)


def _read_basin_ids_file(file_path: Path) -> List[str]:
    with Path(file_path).open('r') as fp:
        return [line.strip() for line in fp if line.strip()]


def _contains_groundwater_feature(dynamic_inputs: Any) -> bool:
    if isinstance(dynamic_inputs, dict):
        return any(_contains_groundwater_feature(v) for v in dynamic_inputs.values())
    if isinstance(dynamic_inputs, (list, tuple)):
        return any(_contains_groundwater_feature(v) for v in dynamic_inputs)
    return str(dynamic_inputs) == 'gw_groundwater_level'


def _remove_groundwater_feature(dynamic_inputs: Any) -> Any:
    if isinstance(dynamic_inputs, dict):
        cleaned: Dict[str, Any] = {}
        for key, value in dynamic_inputs.items():
            updated = _remove_groundwater_feature(value)
            if isinstance(updated, list) and len(updated) == 0:
                continue
            cleaned[key] = updated
        return cleaned

    if isinstance(dynamic_inputs, list):
        cleaned_list: List[Any] = []
        for item in dynamic_inputs:
            if isinstance(item, list):
                group = [v for v in item if str(v) != 'gw_groundwater_level']
                if group:
                    cleaned_list.append(group)
            elif str(item) != 'gw_groundwater_level':
                cleaned_list.append(item)
        return cleaned_list

    return dynamic_inputs


def _create_no_groundwater_fallback_config(config_file: Path) -> Path:
    with Path(config_file).open('r') as fp:
        cfg_dict = yaml.safe_load(fp)

    if not isinstance(cfg_dict, dict):
        raise RuntimeError(f"Could not parse config file as YAML dictionary: {config_file}")

    updated_dynamic_inputs = _remove_groundwater_feature(cfg_dict.get('dynamic_inputs', []))
    if isinstance(updated_dynamic_inputs, dict) and len(updated_dynamic_inputs) == 0:
        raise RuntimeError("Automatic no-GW fallback produced empty dynamic_inputs.")
    if isinstance(updated_dynamic_inputs, list) and len(updated_dynamic_inputs) == 0:
        raise RuntimeError("Automatic no-GW fallback produced empty dynamic_inputs.")

    cfg_dict['dynamic_inputs'] = updated_dynamic_inputs

    exp_name = str(cfg_dict.get('experiment_name', Path(config_file).stem))
    if not exp_name.endswith('_auto_nogw'):
        cfg_dict['experiment_name'] = f"{exp_name}_auto_nogw"

    fallback_config = Path(config_file).with_name(f"{Path(config_file).stem}_auto_nogw.yml")
    with fallback_config.open('w') as fp:
        yaml.safe_dump(cfg_dict, fp, sort_keys=False)

    return fallback_config


def prepare_groundwater_training_context(config_file: Path,
                                         runs_dir: Path = Path('runs'),
                                         print_diagnostics: bool = True,
                                         auto_disable_gw_on_empty_train: bool = False) -> Dict[str, Any]:
    """Build groundwater precheck context for training notebooks.

    This helper centralizes non-training logic (basin list loading, groundwater
    mapping diagnostics, train/val/test coverage checks, and latest run lookup)
    so notebook cells can remain compact.
    """
    cfg_gw = Config(config_file)
    run_pattern = f"{cfg_gw.experiment_name}_*"

    uses_gw_feature = _contains_groundwater_feature(cfg_gw.dynamic_inputs)
    gw_data_frequency = cfg_gw.gw_data_frequency

    train_basins = _read_basin_ids_file(Path(cfg_gw.train_basin_file))
    val_basins = _read_basin_ids_file(Path(cfg_gw.validation_basin_file))
    test_basins = _read_basin_ids_file(Path(cfg_gw.test_basin_file))

    run_candidates = sorted(Path(runs_dir).glob(run_pattern))
    latest_run_dir = run_candidates[-1] if run_candidates else None

    if not uses_gw_feature:
        return {
            'config': cfg_gw,
            'config_file': Path(config_file),
            'run_pattern': run_pattern,
            'fallback_km': None,
            'train_basins': train_basins,
            'val_basins': val_basins,
            'test_basins': test_basins,
            'train_diag': pd.DataFrame(),
            'val_diag': pd.DataFrame(),
            'test_diag': pd.DataFrame(),
            'train_with_gw': 0,
            'val_with_gw': 0,
            'test_with_gw': 0,
            'train_gw_coverage': pd.DataFrame(),
            'train_with_gw_data': 0,
            'gw_auto_disabled': False,
            'run_dir': latest_run_dir,
        }

    fallback_km = _get_gw_fallback_max_distance_km()
    train_diag = get_groundwater_mapping_diagnostics(cfg_gw.data_dir, train_basins, fallback_km, gw_data_frequency)
    val_diag = get_groundwater_mapping_diagnostics(cfg_gw.data_dir, val_basins, fallback_km, gw_data_frequency)
    test_diag = get_groundwater_mapping_diagnostics(cfg_gw.data_dir, test_basins, fallback_km, gw_data_frequency)

    train_with_gw = int((train_diag['mapped_well'] != 'NONE').sum())
    val_with_gw = int((val_diag['mapped_well'] != 'NONE').sum())
    test_with_gw = int((test_diag['mapped_well'] != 'NONE').sum())

    train_start = pd.to_datetime(cfg_gw.train_start_date, format='%d/%m/%Y')
    train_end = pd.to_datetime(cfg_gw.train_end_date, format='%d/%m/%Y')

    gw_train_rows: List[Dict[str, Any]] = []
    for basin in train_basins:
        basin_df = load_camels_gb_v2_timeseries(
            cfg_gw.data_dir,
            basin,
            gw_data_frequency=gw_data_frequency,
            train_start_date=cfg_gw.train_start_date,
            train_end_date=cfg_gw.train_end_date
        )
        basin_df = basin_df.loc[(basin_df.index >= train_start) & (basin_df.index <= train_end)]
        gw_non_nan = int(basin_df['gw_groundwater_level'].notna().sum()) if 'gw_groundwater_level' in basin_df.columns else 0
        gw_train_rows.append({'basin': basin, 'gw_non_nan_train': gw_non_nan})

    train_gw_coverage_df = pd.DataFrame(gw_train_rows)
    train_with_gw_data = int((train_gw_coverage_df['gw_non_nan_train'] > 0).sum())

    if print_diagnostics:
        print(f"GW precheck (fallback_km={fallback_km}, frequency={gw_data_frequency}):")
        print(f"  train: {train_with_gw}/{len(train_basins)} basins with mapped groundwater")
        print(f"  valid: {val_with_gw}/{len(val_basins)} basins with mapped groundwater")
        print(f"  test : {test_with_gw}/{len(test_basins)} basins with mapped groundwater")
        print("\nTrain-basin groundwater mapping diagnostics:")
        print(train_diag.to_string(index=False))
        print("\nTrain-period groundwater data availability (non-NaN rows):")
        print(train_gw_coverage_df.to_string(index=False))

    if train_with_gw == 0 and not auto_disable_gw_on_empty_train:
        raise RuntimeError(
            "No train basins have mapped groundwater at current fallback threshold. "
            "If source shows 'csv_missing_daily', basin is in mapping CSV but mapped well has no daily GW file.")

    effective_config_file = Path(config_file)
    effective_cfg = cfg_gw
    effective_run_pattern = run_pattern
    gw_auto_disabled = False

    if train_with_gw == 0 or train_with_gw_data == 0:
        if not auto_disable_gw_on_empty_train:
            raise RuntimeError(
                "Groundwater is mapped but unavailable in the configured training period for all train basins "
                f"({cfg_gw.train_start_date} to {cfg_gw.train_end_date}). "
                "Use train basins with GW observations in this period, adjust train dates, "
                "or remove 'gw_groundwater_level' from dynamic_inputs.")

        effective_config_file = _create_no_groundwater_fallback_config(Path(config_file))
        effective_cfg = Config(effective_config_file)
        effective_run_pattern = f"{effective_cfg.experiment_name}_*"
        run_candidates = sorted(Path(runs_dir).glob(effective_run_pattern))
        latest_run_dir = run_candidates[-1] if run_candidates else None
        gw_auto_disabled = True

        if print_diagnostics:
            print(
                "\nNo train-period GW observations detected; auto-disabled 'gw_groundwater_level' "
                f"for this run via config: {effective_config_file}")

    return {
        'config': effective_cfg,
        'config_file': effective_config_file,
        'run_pattern': effective_run_pattern,
        'fallback_km': fallback_km,
        'train_basins': train_basins,
        'val_basins': val_basins,
        'test_basins': test_basins,
        'train_diag': train_diag,
        'val_diag': val_diag,
        'test_diag': test_diag,
        'train_with_gw': train_with_gw,
        'val_with_gw': val_with_gw,
        'test_with_gw': test_with_gw,
        'train_gw_coverage': train_gw_coverage_df,
        'train_with_gw_data': train_with_gw_data,
        'gw_auto_disabled': gw_auto_disabled,
        'run_dir': latest_run_dir,
    }


def load_camels_gb_v2_timeseries(data_dir: Path,
                                 basin: str,
                                 gw_data_frequency: str = 'daily',
                                 train_start_date: str = None,
                                 train_end_date: str = None) -> pd.DataFrame:
    """Load hydromet and optional groundwater timeseries for a basin of CAMELS_GB_V2.

    Expectations:
    - `data_dir/timeseries/` contains hydromet CSV files whose filenames include the basin id.
    - Groundwater files, if present, should include 'groundwater' or 'gw' in the filename and the basin id.
    - Hydromet files must contain a `date` column parseable with `%Y-%m-%d`.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame containing combined forcings, targets and groundwater variables (prefixed `gw_`).
    """
    forcing_path = Path(data_dir) / 'timeseries'
    if not forcing_path.is_dir():
        raise OSError(f"{forcing_path} does not exist")

    # find hydromet file for basin
    files = list(forcing_path.glob('**/*'))
    hydromet_candidates = [
        f for f in files
        if f.is_file() and 'hydromet_daily_timeseries' in f.name and f"_{basin}_" in f.name
    ]

    if hydromet_candidates:
        hydromet_file = hydromet_candidates[0]
    else:
        # fallback: any file containing basin id
        other_candidates = [f for f in files if f.is_file() and f"_{basin}_" in f.name]
        if other_candidates:
            hydromet_file = other_candidates[0]
        else:
            raise FileNotFoundError(f'No hydromet file found for Basin {basin} in {forcing_path}')

    df = pd.read_csv(hydromet_file, sep=',', header=0, dtype={'date': str})
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df = df.set_index('date')

    # map basin -> nearest groundwater well via easting/northing and merge groundwater timeseries if available
    fallback_max_distance_km = _get_gw_fallback_max_distance_km()
    basin_to_well = _build_basin_to_well_map(data_dir, fallback_max_distance_km, gw_data_frequency)
    mapped_well = basin_to_well.get(str(basin))
    if mapped_well is not None:
        daily_dir = _find_groundwater_daily_dir(data_dir)
        monthly_dir = _find_groundwater_monthly_dir(data_dir)
        
        gw_file = None
        is_monthly = False
        
        # Check daily first if allowed
        if gw_data_frequency in ['daily', 'any']:
            if daily_dir is not None:
                gw_file_map = _get_available_groundwater_well_files(daily_dir)
                gw_file = gw_file_map.get(_normalize_well_id(mapped_well))
        
        # If daily not found/allowed, check monthly
        if gw_file is None and gw_data_frequency in ['monthly', 'any']:
            if monthly_dir is not None:
                gw_file_map = _get_available_groundwater_monthly_files(monthly_dir)
                gw_file = gw_file_map.get(_normalize_well_id(mapped_well))
                if gw_file is not None:
                    is_monthly = True

        if gw_file is not None and gw_file.exists():
            gw_df = pd.read_csv(gw_file, sep=',', header=0, dtype={'date': str})
            if 'date' in gw_df.columns:
                gw_df['date'] = pd.to_datetime(gw_df['date'], format='%Y-%m-%d')
                gw_df = gw_df.set_index('date')

                # Dynamic column name detection: rename values to 'groundwater_level'
                for col in ['value', 'groundwater_level', 'lh_val']:
                    if col in gw_df.columns:
                        gw_df = gw_df.rename(columns={col: 'groundwater_level'})
                        break

                # If monthly data, upsample to daily using linear interpolation
                if is_monthly:
                    gw_df = gw_df.resample('D').interpolate(method='linear')

                # Local Z-Score normalization (standardized anomaly) per well
                if 'groundwater_level' in gw_df.columns:
                    if train_start_date and train_end_date:
                        t_start = pd.to_datetime(train_start_date, format='%d/%m/%Y')
                        t_end = pd.to_datetime(train_end_date, format='%d/%m/%Y')
                        train_window = gw_df.loc[(gw_df.index >= t_start) & (gw_df.index <= t_end), 'groundwater_level']
                    else:
                        train_window = gw_df['groundwater_level']

                    mean_val = train_window.mean()
                    std_val = train_window.std()
                    if pd.notna(std_val) and std_val > 0:
                        gw_df['groundwater_level'] = (gw_df['groundwater_level'] - mean_val) / std_val
                    elif std_val == 0:
                        gw_df['groundwater_level'] = 0.0
                    # If std_val is NaN (no data or 1 data point in train period), leave as NaN.

                # Keep only groundwater_level column
                cols_to_keep = [col for col in ['groundwater_level'] if col in gw_df.columns]
                gw_df = gw_df[cols_to_keep]

                # prefix groundwater column names to avoid clashes
                gw_df = gw_df.add_prefix('gw_')

                # align to hydromet date index to avoid extending date range unexpectedly
                gw_df = gw_df.reindex(df.index)

                # combine on date index
                df = pd.concat([df, gw_df], axis=1)

    # Keep groundwater dynamic inputs available even when a basin has no matched well
    # (e.g., filtered by max fallback distance).
    if 'gw_groundwater_level' not in df.columns:
        df['gw_groundwater_level'] = np.nan

    return df

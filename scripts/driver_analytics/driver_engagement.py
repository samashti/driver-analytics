#!/usr/bin/env python
# -*- coding: utf-8 -*-

####################################################################################################
"""
driver_engagement.py: ...
"""

__authors__ = (
    "Nikhil S Hubballi (nikhil.hubballi@gmail.com)",
)
__copyright__ = "Copyright 2021, Nikhil Hubballi"
__credits__ = [
    "Nikhil S Hubballi",
]
__license__ = ""
__version__ = "0.1.0"
__first_release__ = "Dec 13, 2021"
__version_date__ = "Dec 13, 2021"
__maintainer__ = ("Nikhil S Hubballi")
__email__ = ("nikhil.hubballi@gmail.com")
__status__ = "Development"
__sheet__ = __name__

####################################################################################################

import os
from pathlib import Path
from argparse import Namespace

import pandas as pd
from tqdm import tqdm

from driver_analytics.generate_data import GenerateDriverData

tqdm.pandas()


def driver_dates_master(df: pd.DataFrame):
    """
    Args:
        df (pd.DataFrame): Dataframe with simulared driver log records

    Returns:
        keys[pd.DataFrame]: Master dataframe of driver_id and dates
    """

    df['timestamp'] = pd.to_datetime(df['timestamp'].apply(
        lambda x: x.strftime('%Y-%m-%d')
    ))

    keys = list(df.groupby(['driver_id', 'timestamp']).groups.keys())
    keys = pd.DataFrame(keys)
    keys = keys.drop_duplicates()
    keys.columns = ['driver_id', 'date']

    return keys


def driver_engage_analysis(df: pd.DataFrame,
                           gen_data: GenerateDriverData,
                           datapath: Path):
    """Perform preprocessing and anlysis on the simulated driver log records,
    to generate reports on breaks and deliveries by drivers in the analysis 
    period.

    Args:
        df (pd.DataFrame): Dataframe with simulared driver log records

        gen_data (GenerateDriverData): Class used for simulating driver logs

        datapath (Path): folder path where output files are exported.

    Returns:
        [pd.DataFrame]: delivery log records report for each date
    """

    # Since each trip has 2 associated records,
    # start time and end time, we filter only the record
    # associated with end time for delivery and break end
    # for the analysis
    trips_df = df[(df['flag'] == 'delivered') | (df['flag'] == 'b-end')]

    # Calculate Distance travelled for each record based
    # on start and end latlon pair
    def get_dist(row):
        start = (row['start_lon'], row['start_lat'])
        end = (row['end_lon'], row['end_lat'])
        return gen_data.get_distance_time(start, end)[0]

    trips_df['distance'] = trips_df.progress_apply(get_dist, axis=1)

    # date column for aggregation later
    trips_df['date'] = trips_df.timestamp.progress_apply(
        lambda x: x.strftime('%Y-%m-%d')
    )

    # aggregate time spent, distance travelled and number of instances
    # of deliveries and breaks for the day by each date and driver
    report = trips_df.groupby(['driver_id', 'date', 'flag']).agg(
        {'timestamp': 'count', 'time_spent': 'sum', 'distance': 'sum'}
    ).reset_index()

    report.rename(columns={'timestamp': 'trips'}, inplace=True)
    report['time_spent'] = report['time_spent'].apply(
        lambda x: round(x/60, 3)  # in minutes
    )
    report['distance'] = report['distance'].apply(
        lambda x: round(x/1000, 2)  # in km
    )
    report['date'] = pd.to_datetime(report['date'])

    # Since, each break corresponds to no delivery, update
    # trips column corresponding to 'b-end' as zero
    report.loc[report['flag'] != 'delivered', 'trips'] = 0

    # Generate Key Master data with drivers & dates
    keys_df = driver_dates_master(df)

    # delivery dataframe for the entire period for all drivers
    report_del = pd.merge(
        keys_df, report[report['flag'] == 'delivered'],
        on=['driver_id', 'date'], how='left'
    )
    report_del.loc[report_del['flag'].isna(), 'flag'] = 'delivered'
    report_del = report_del.fillna(0)

    # breaks dataframe for the entire period for all drivers
    report_brk = pd.merge(
        keys_df, report[report['flag'] == 'b-end'],
        on=['driver_id', 'date'], how='left'
    )

    # Export delivery and breaks by date dataframes
    report_del.to_feather(
        datapath.joinpath('drivers_delivery_by_date.feather')
    )
    report_brk.to_feather(
        datapath.joinpath('drivers_breaks_by_date.feather')
    )
    return report_del


def driver_cohort_report(report_del: pd.DataFrame, datapath: Path):
    """Using the delivery log records report for each date, create a report
    of driver's cohort (PD, AD, EAD, UAD) change updates through out the 
    analysis period on each day with 30 day sliding window.

    Args:
        report_del (pd.DataFrame): delivery log records report for each date

        datapath (Path): folder path where output files are exported.

    Returns:
        [pd.DataFrame]: driver's engagement cohort changes report
    """

    driver_clubs = {'PD': 1, 'AD': 2, 'EAD': 3, 'UAD': 4}

    # 30-day rolling aggregates for Driver engagement data
    def groupby_rolling(grp_df):
        grp_df = grp_df.sort_values(by='date')
        df = grp_df.set_index("date")
        df['trips'] = df.rolling("30D")["trips"].sum()
        df['time_spent'] = df.rolling("30D")["time_spent"].sum()
        df['distance'] = df.rolling("30D")["distance"].sum()
        return df.reset_index()

    rdf = report_del.groupby('driver_id').apply(groupby_rolling)
    rdf = rdf.drop('driver_id', axis=1).reset_index()
    rdf = rdf.drop('level_1', axis=1)

    # Conditions for driver engagement club classification
    # ! trips-3.0, time-20.73, distance-12.09 (not covered under suggested filter)
    # -> for PD, distance filter changed from <10km to <20km
    # ! trips-4.0, time-27.41, distance-15.99 (not covered under suggested filter)
    # -> for PD, trips/delivery number filter changed from <4 to <=4
    # ! trips-6.0, time-27.52, distance-16.27 (not covered under suggested filter)
    # -> added extra condition for AD, trips/delivery number >=4
    # & dirance filter <20 km
    pd_condition = (rdf['trips'] <= 4) & (rdf['time_spent'] < 30) & \
                   (rdf['distance'] < 20)
    ad_condition = ((rdf['time_spent'] >= 30) & (rdf['distance'] > 10)) | \
                   ((rdf['trips'] >= 4) & (rdf['distance'] < 20))
    ead_condition = (rdf['trips'] >= 4) & (rdf['distance'] >= 30)
    uad_condition = (rdf['trips'] >= 10) & (rdf['distance'] >= 40)

    # Apply filters
    rdf.loc[pd_condition, 'category'] = 'PD'
    rdf.loc[ad_condition, 'category'] = 'AD'
    rdf.loc[ead_condition, 'category'] = 'EAD'
    rdf.loc[uad_condition, 'category'] = 'UAD'
    print('Category Counts: \n', rdf.category.value_counts())
    print('NA Check for Category: ', (len(rdf[rdf.category.isna()]) == 0))

    # Create Cohorts based on Upgrade/Downgrade
    rdf['change'] = rdf['category'].apply(lambda x: driver_clubs[x])

    def get_flags(_df):
        _df = _df.sort_values(by='date')
        _df['change'] = _df['change'].diff()
        _df.loc[_df['change'] < 0, 'change_flag'] = 'Downgraded'
        _df.loc[_df['change'] == 0, 'change_flag'] = 'NoChange'
        _df.loc[_df['change'] > 0, 'change_flag'] = 'Upgraded'
        return _df

    rdf = rdf.groupby('driver_id').apply(get_flags)
    rdf = rdf.round(3)

    rdf.to_feather(
        datapath.joinpath('drivers_engagement_clubs.feather')
    )
    return rdf


def driver_clubs_report(rdf: pd.DataFrame, datapath: Path):
    """Using the driver's engagement cohort changes report, each cohort's
    driver distribution report is generated for each date.

    Args:
        rdf (pd.DataFrame): driver's engagement cohort changes report

        datapath (Path): folder path where output files are exported.

    Returns:
        [pd.DataFrame]: report for the distribution among 4 categorie
                    (PD, AD, EAD, UAD)
    """

    dist_df = rdf.groupby(['date', 'category']).agg(
        {'trips': 'count'}).reset_index()

    dist_df = pd.pivot_table(dist_df, 'trips', index='date',
                             columns='category').fillna(0).reset_index()

    dtypes = {'PD': 'int', 'AD': 'int', 'EAD': 'int', 'UAD': 'int'}
    dist_df = dist_df.astype(dtypes)

    dist_df.to_feather(
        datapath.joinpath('driver_clubs_report_by_date.feather')
    )

    return dist_df


def main():
    """
    1. Simulate and export driver log data for the analysis period

    2. Run Driver engagement Analysis

    3. Generate report for driver cohort changes

    4. Generate driver clubs report for each date
    """

    print('Generating Data ...')
    # Generatee Simulation Data for Drivers
    # for the Given time period and ROI
    gen_data = GenerateDriverData(
        args.num_drivers, args.start_date, args.end_date,
        args.lat, args.lon, args.radius, args.distance_method
    )

    datapath = args.filepath.joinpath('data')
    gen_data.run(datapath)

    # drivers log dataframe
    dlog_df = gen_data.log_df.copy()
    print('The simulated dataframe for driver data shape: ', dlog_df.shape)

    print('Running Driver Engagement Analysis ...')
    # Run Engagement Analysis
    report_del = driver_engage_analysis(dlog_df, gen_data, datapath)

    print('Generating Report for Cohort Changes ...')
    # Run Driver Cohort Upgrade/Downgrade Analysis Report
    rdf = driver_cohort_report(report_del, datapath)

    print('Generating Driver Clubs Report for each Date ...')
    # Run Drivers Clubs Report for each Date
    _ = driver_clubs_report(rdf, datapath)

    print('All Data Exported to folder...')
    print('Done.')

    pass


if __name__ == '__main__':

    # ! ROI - Bangalore area: lat=12.975118, lon=77.592690, radius ~25000m
    args = Namespace(
        num_drivers=1000,
        start_date='2021-08-01',
        end_date='2021-11-30',
        lat=12.975118,
        lon=77.5926901,
        radius=25000.0,
        distance_method='haversine',
        filepath=Path(__file__).parent.parent.parent
    )

    main()

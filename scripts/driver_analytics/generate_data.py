#!/usr/bin/env python
# -*- coding: utf-8 -*-

####################################################################################################
"""
generate_data.py: ...
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

import uuid
import math

import names
import random
import requests
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from rich.progress import track

# from tqdm import tqdm
# tqdm.pandas()

# from pandarallel import pandarallel
# pandarallel.initialize(nb_workers=2, progress_bar=True)


class GenerateDriverData:

    def __init__(self, num_drivers: int, start_date: str, end_date: str,
                 lat: float, lon: float, radius: float,
                 distance_method: str = 'haversine') -> None:
        """ Initialise the Simulation to generate Driver data with the
        arguments required.

        Args:
            num_drivers (int): Number of drivers to be used for simulation

            start_date (str): Start date for simulation

            end_date (str): End date for simulation

            lat (float): Central latitude of the ROI

            lon (float): Central longitude of the ROI

            radius (float): Radius to be used to cover the ROI

            distance_method (str, optional): Distance method to be used to 
                        calculate distance between two latlon pair.
                        Takes either of 'haversine', 'osrm_api' as inputs.

                        1. Haversine uses the Haversine distance formula to
                            calculate distance between two points. This is a
                            geographic distance

                        2. Using OSRM API provides the distance between two
                            points based on a route via road network using
                            OSRM's demo server. 
                            PS: Adds delay to simulation as requests take longer

                        Defaults to 'haversine'.
        """

        self.num_drivers = num_drivers
        self.start_date = start_date
        self.end_date = end_date
        self.roi_lat = lat
        self.roi_lon = lon
        self.roi_radius = radius
        self.distance_method = distance_method

        pass

    def generate_latlon(self, lat: float, lon: float, radius: float) -> tuple[float]:
        """Generate a random pair of latitude and longitude within a fixed
        radius of given latitude, longitude values.

        Args:
            lat (float): Latitude of input point

            lon (float): Longitude of input point

            radius (float): Radius bound to generate a random point

        Returns:
            tuple[float]: random point with lat, lon as tuple of float values
        """

        radius_deg = radius/111300  # radius in degree

        u = float(random.uniform(0.0, 1.0))
        v = float(random.uniform(0.0, 1.0))

        w = radius_deg * math.sqrt(u)
        t = 2 * math.pi * v
        y = w * math.cos(t)
        x = w * math.sin(t)

        x1 = x + lon
        y1 = y + lat
        # tuple(lon, lat)
        return (x1, y1)

    def create_driver_profile(self, depot_locations: list[tuple]) -> dict:
        """Creates a random driver profile with driver id, name, vehicle id
        by assigning him to any of the given depot locations randomly.

        Args:
            depot_locations (list[tuple]): list of latlon points on earth
                        that corresponds to depot locations

        Returns:
            dict: dictionary of driver profile
        """

        driver = dict()
        driver['driver_id'] = str(uuid.uuid4()).split('-')[-1]
        driver['first_name'] = names.get_first_name()
        driver['last_name'] = names.get_last_name()
        driver['vehicle_id'] = str(uuid.uuid4()).split('-')[-1]
        driver['depot'] = random.choice(depot_locations)

        return driver

    def pickup_delivery_latlon(self, depot: tuple[float]) -> tuple:
        """Creates a pickup point and delivery point with respect to the
        input depot location.

        ! Assumption: Each starting point is randomly chosen within a 10km
                    radius of depot

        ! Assumption: Each associated end point is randomly chosen within
                    a radius of 5km from the starting point

        Args:
            depot (tuple[float]): depot location's latlon as tuple

        Returns:
            tuple: tuple of starting point and end point for pickup & delivery
        """

        # within 10km radius of depot
        start_point = self.generate_latlon(depot[1], depot[0], 10000)
        # within 5km radius of starting point
        end_point = self.generate_latlon(start_point[1], start_point[0], 5000)

        return start_point, end_point

    def osrm_routing_api(self, start: tuple[float], end: tuple[float]) -> tuple[float]:
        """Using the OSRM API hosted on a demo server at router.project-osrm.org
        the distance between the start and end point can be calculated based
        on route via road network between two points, along with an estimated
        time taken

        Args:
            start (tuple[float]): latlon pair as tuple for start point

            end (tuple[float]): latlon pair as tuple for end point

        Returns:
            tuple[float]: distance and time taken for travel as tuple
        """

        BASE_URL = "https://router.project-osrm.org/route/v1/driving"
        start_coord = f'{start[0]},{start[1]}'
        end_coord = f'{end[0]},{end[1]}'
        ENDPOINT = f"{BASE_URL}/{start_coord};{end_coord}"
        r = requests.get(ENDPOINT)
        res = r.json()
        distance = res['routes'][0]['distance']
        # Assuming a city avg traffic speed of ~35km/h
        duration = (res['routes'][0]['duration'] * 2)

        return distance, duration

    def haversine_distance(self, start: tuple[float], end: tuple[float]) -> tuple[float]:
        """Haversine uses the Haversine distance formula to calculate distance
        between two points. This is a geographic distance between 2 points.

        Args:
            start (tuple[float]): latlon pair as tuple for start point

            end (tuple[float]): latlon pair as tuple for end point

        Returns:
            tuple[float]: distance and time taken for travel as tuple
        """

        lon1, lat1 = start
        lon2, lat2 = end

        r = 6371*1000  # radius of earth (in meter)
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)

        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)

        a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * \
            np.cos(phi2) * np.sin(delta_lambda / 2)**2
        res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

        distance = np.round(res, 2)
        # Assuming a city avg traffic speed of 35km/h
        duration = ((distance / 1000) / 35) * 3600
        return distance, duration

    def deliveries_in_day(self, num_delivery: int, depot: tuple[float]) -> pd.DataFrame:
        """Based on the number of deliveries, and the starting depot location,
        this method creates multiple pickup-delivery location pairs along with
        an unique id, distance between locations and time taken for delivery.

        Args:
            depot (tuple[float]): Depot location's latlon pair as tuple

        Returns:
            pd.DataFrame: dataframe consisting of pickup-delivery info for N deliveries
        """

        deliveries = list()
        for n in range(num_delivery):
            delivery = dict()
            delivery['id'] = n+1

            sp, ep = self.pickup_delivery_latlon(depot)
            delivery['start_point'] = sp
            delivery['end_point'] = ep

            distance, duration = self.get_distance_time(sp, ep)
            delivery['distance_m'] = distance
            delivery['duration_s'] = duration
            deliveries.append(delivery)

        return pd.DataFrame(deliveries)

    def calculate_delivery_order(self, ddf: pd.DataFrame,
                                 depot: tuple[float]) -> list[int]:
        """For deliveries assigned in the day to driver from a depot location,
        the deliveries ordered so that the driver moves to the first closest
        pickup point first and deliver it. Next, he would move to the next 
        closest pickup point for the subsequent delivery and so on, until he
        completes all deliveries for the day. This method generates the 
        delivery order list for a given number of pickup-delivery location pairs
        in a day.

        Args:
            ddf (pd.DataFrame): delivery dataframe with pickup-deliver locations

            depot (tuple[float]): depot location from which the driver starts

        Returns:
            list[int]: delivery order numbers as list
        """

        delivery_order = list()

        # Get First Order - Depot to First Pickup
        ddf['dist_depot_sp'] = ddf['start_point'].apply(
            lambda loc: self.get_distance_time(depot, loc)[0]
        )
        first_order = ddf[ddf['dist_depot_sp'] ==
                          ddf['dist_depot_sp'].min()].iloc[0]
        delivery_order.append(first_order['id'])

        # Get Subsequent Orders - After First Drop-off
        _df = ddf[ddf['id'] != first_order['id']].copy()
        end_point = first_order['end_point']

        while len(_df) > 0:

            _df['dist_epx_spy'] = _df['start_point'].apply(
                lambda loc: self.get_distance_time(end_point, loc)[0]
            )
            order = _df[_df['dist_epx_spy'] ==
                        _df['dist_epx_spy'].min()].iloc[0]
            delivery_order.append(order['id'])

            _df = _df[_df['id'] != order['id']].copy()
            end_point = order['end_point']

        return delivery_order

    def create_log(self, driver_id: str, timestamp: pd.Timestamp,
                   flag: str, sp: tuple[float], ep: tuple[float],
                   time_spent: float) -> dict:
        """Based on the inputs, this method creates a dictionary of log,
        for the driver.

        Args:
            driver_id (str): Unique driver id

            timestamp (pd.Timestamp): Logging time

            flag (str): Flag of the task in progress 
                        (b-start, b-end, picked-up, delivered)

            sp (tuple[float]): latlon of start point as tuple

            ep (tuple[float]): latlon of end point as tuple

        Returns:
            dict: Log as dictionary object
        """

        dlog = {
            'driver_id': driver_id,
            'timestamp': timestamp,
            'flag': flag,
            'start_lon': sp[0],
            'start_lat': sp[1],
            'end_lon': ep[0],
            'end_lat': ep[1],
            'time_spent': time_spent
        }

        return dlog

    def create_driver_log(self, driver_id: str, date: pd.Timestamp,
                          depot: tuple[float], ddf: pd.DataFrame) -> pd.DataFrame:
        """Simulate a driver's day from the start of his shift at 9AM with day's
        deliveries in pipeline. In between each delivery, break periods are
        logged as well. Each log contains, the driver id, timestamp, flag for the
        task in progress (break, delivery) along with start and end points.

        This method covers all the assumptions mentioned in readme.

        Args:
            driver_id (str): Unique Driver ID

            date (pd.Timestamp): Date on which simulation of logging is done

            depot (tuple[float]): Driver's depot location

            ddf (pd.DataFrame): deliveries info for the day

        Returns:
            pd.DataFrame: logs for the day as dataframe
        """

        # time for all deliveries in seconds
        time_delivery = ddf['duration_s'].sum()
        time_working = 10*60*60  # 9AM-7PM, in seconds
        time_last_delivery = ddf.iloc[-1]['duration_s']
        # to accommodate the last delivery before 7PM
        time_breaks = time_working - time_delivery - time_last_delivery

        driver_logs = list()

        # First Log of the Day - from Depot
        start_point = ddf['start_point'].iloc[0]
        driver_logs.append(
            self.create_log(driver_id, date, 'b-start',
                            depot, start_point, np.NaN)
        )

        # break time should include time for next commute to pickup
        min_break = self.get_distance_time(depot, start_point)[-1]
        break_interval = random.uniform(min_break, time_breaks)
        # update remaining time for breaks in the day
        time_breaks -= break_interval
        # break ends at
        timestamp = date + pd.Timedelta(seconds=break_interval)

        driver_logs.append(
            self.create_log(driver_id, timestamp, 'b-end',
                            depot, start_point, break_interval)
        )

        # Loop through delivery assignments for the day
        for order in range(len(ddf)):

            start_point = ddf['start_point'].iloc[order]
            end_point = ddf['end_point'].iloc[order]

            driver_logs.append(
                self.create_log(driver_id, timestamp, 'picked-up',
                                start_point, end_point, np.NaN)
            )

            # update delivery time based on time for traveling to destination
            time_taken = ddf['duration_s'].iloc[order]
            timestamp += pd.Timedelta(seconds=time_taken)

            driver_logs.append(
                self.create_log(driver_id, timestamp, 'delivered',
                                start_point, end_point, time_taken)
            )

            # check if there is time remaining for breaks in the day
            if time_breaks > 0:

                # check if there is another delivery in pipeline
                next_order = (order < len(ddf)-1)

                # CASE - More deliveries left in the day
                if next_order:
                    next_start = ddf['start_point'].iloc[order+1]
                    driver_logs.append(
                        self.create_log(driver_id, timestamp, 'b-start',
                                        end_point, next_start, np.NaN)
                    )

                    # next commute
                    min_break = self.get_distance_time(
                        end_point, next_start)[-1]
                    break_interval = random.uniform(min_break, time_breaks)

                    time_breaks -= break_interval
                    timestamp += pd.Timedelta(seconds=break_interval)

                    driver_logs.append(
                        self.create_log(driver_id, timestamp, 'b-end',
                                        end_point, next_start, break_interval)
                    )
                # CASE - All deliveries done for the day
                else:

                    driver_logs.append(
                        self.create_log(driver_id, timestamp, 'b-start',
                                        end_point, depot, np.NaN)
                    )
                    ts = timestamp
                    timestamp = date + pd.Timedelta(seconds=time_working)
                    break_interval = (timestamp - ts).seconds

                    driver_logs.append(
                        self.create_log(driver_id, timestamp, 'b-end',
                                        end_point, depot, break_interval)
                    )
                # END IF - check remaining deliveries
            # END IF - check remaining break time
        # END FOR LOOP - deliveries for the day

        log_df = pd.DataFrame(driver_logs)
        return log_df

    def log_free_day(self, driver_id: str,
                     date: pd.Timestamp, depot: tuple[float]) -> pd.DataFrame:
        """Simulate a driver's free day from the start of his shift at 9AM with 
        when there are no deliveries planned for the day. The entire day would
        be logged under break period. Each log contains, the driver id, timestamp, 
        flag for the task in progress (break) along with depot as both start 
        and end points.

        This method covers all the assumptions mentioned in readme.

        Args:
            driver_id (str): Unique Driver ID

            date (pd.Timestamp): Date on which simulation of logging is done

            depot (tuple[float]): Driver's depot location

        Returns:
            pd.DataFrame: logs for the day as dataframe
        """
        driver_logs = list()

        # First Log of the Day - from Depot
        driver_logs.append(
            self.create_log(driver_id, date, 'b-start',
                            depot, depot, np.NaN)
        )

        time_working = 10*60*60  # 9AM-7PM, in seconds
        timestamp = date + pd.Timedelta(seconds=time_working)

        driver_logs.append(
            self.create_log(driver_id, timestamp, 'b-end',
                            depot, depot, time_working)
        )

        log_df = pd.DataFrame(driver_logs)
        return log_df

    def loop_drivers(self, row: pd.Series, date: pd.Timestamp) -> pd.DataFrame:
        """Simulate time logging of breaks and deliveries for all drivers in 
        consideration using their driver profile on the input date.

        Args:
            row (pd.Series): driver profile data

            date (pd.Timestamp): Date on which simulation of logging is done

        Returns:
            pd.DataFrame: logs for the day for all drivers as dataframe
        """

        driver_id = row['driver_id']
        depot = row['depot']
        num_delivery = row['num_delivery']

        # Genearate dataframe of Delivery Assignments for day
        if num_delivery > 0:
            delivery_df = self.deliveries_in_day(num_delivery, depot)
        else:
            delivery_df = pd.DataFrame()

        # exit if there are no deliveries to make for the date
        if delivery_df.empty:
            log_df = self.log_free_day(driver_id, date, depot)
            return log_df

        # Identify the order of delivery assignments based on travel distance
        delivery_order = self.calculate_delivery_order(
            delivery_df.copy(), depot
        )
        delivery_df['delivery_order'] = delivery_order
        delivery_df.sort_values(by='delivery_order', inplace=True)

        # Simulate Driver Logs for the day based on deliveries & break time
        log_df = self.create_driver_log(
            driver_id, date, depot, delivery_df.copy()
        )

        return log_df

    def loop_dates(self, date: pd.Timestamp, filepath: Path) -> None:
        """Simulate time logging of breaks and deliveries for drivers
        on the input date.

        Args:
            date (pd.Timestamp): Date on which simulation of logging is done

            filepath (Path): folder path where the output file containing logs
                        for each day is exported.
        """

        ddf = self.drivers_df.copy()
        ddf['num_delivery'] = random.choices(
            [0, 1, 2, 3], weights=[70, 20, 5, 5], k=self.num_drivers
        )

        log_dflist = ddf.apply(
            lambda row: self.loop_drivers(row, date),
            axis=1
        )
        # log_dflist = list()
        # for ri in track(range(len(self.drivers_df)), description='Looping through Drivers : ', total=len(self.drivers_df)):
        #     log_dflist.append(self.loop_drivers(
        #         self.drivers_df.iloc[ri], date))

        log_df = pd.concat(log_dflist.tolist())
        log_df.reset_index(drop=True, inplace=True)
        log_df.to_feather(
            filepath.joinpath(f'{date.strftime("%Y-%m-%d")}.feather')
        )
        pass

    def run(self, filepath: str) -> None:
        """Runs Simulation of generating random drivers profile, assigns depot
        locations, number of deliveries for all drivers on a given date and
        logs the times for each day (breaks and deliveries).

        Args:
            filepath (str): folder path where the output files are exported.
        """

        # Filepath
        filepath = Path(filepath)
        wip_path = filepath.joinpath('wip')

        # ! Assumption: There are 5 Depot locations serving around the AOI
        # Generate 5 Random locations for depot within ROI
        self.depot_list = [
            self.generate_latlon(
                self.roi_lat, self.roi_lon, self.roi_radius
            ) for loc in range(5)
        ]

        # Generate a random dataframe for drivers
        self.drivers_df = pd.DataFrame([self.create_driver_profile(
            self.depot_list) for r in range(self.num_drivers)])

        # Generate list of working-day dates between start & end date
        # ! Assumption: Deliveries are done only on working/week days (Mon-Fri)
        # ! Assumption: Driver Work in a 9AM-7PM shift on work day
        self.dates_list = [self.start_date] if np.is_busday(
            self.start_date) else []
        workdays = np.busday_count(self.start_date, self.end_date)

        # Loop through identified number of workdays
        for wi in range(0, workdays+1):
            date = np.busday_offset(self.start_date, wi, roll='forward')
            date = pd.to_datetime(str(date)) + pd.Timedelta(hours=9)
            self.dates_list.append(date)
        # END FOR LOOP - workdays

        # Choose Distance & Time calculation method
        distance_dict = {
            'haversine': self.haversine_distance,
            'osrm_api': self.osrm_routing_api,
        }
        self.get_distance_time = distance_dict[self.distance_method]

        # Run Simulation
        _ = [self.loop_dates(date, wip_path) for date in track(
            self.dates_list, description='Looping through Dates ...', total=len(self.dates_list))]

        # Export Drivers Master
        self.drivers_df['depot'] = self.drivers_df['depot'].astype(str)
        self.drivers_df.to_feather(filepath.joinpath('drivers_master.feather'))

        # Export Simulated Data
        flist = glob(f'{wip_path.as_posix()}/*.feather')
        self.log_df = pd.concat([pd.read_feather(f) for f in track(
            flist, description='Reading Date-wise driver logs ...', total=len(flist))])
        self.log_df.sort_values(by='timestamp', inplace=True)
        self.log_df.reset_index(drop=True, inplace=True)
        self.log_df.to_feather(
            filepath.joinpath('drivers_delivery_log.feather')
        )
        pass

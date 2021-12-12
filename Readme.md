# Driver Engagement Analytics

Derive driver clubs (driver engagement categories/cohorts) for each driver based on their trips dataset from last 4 months, by generating simulated data.

---

## TASK

- Number of drivers - 1000

- Generate random timestamps (start or end) for each driver (past 4 months) - this end time stamp is a time at which the delivery happened; start is a time stamp at which the parcel was picked up. Can also have a record indicating break-starts and break-ends. Options;

  - trip_start

  - trip_end

  - break_start

  - break_end

- Flags possible for each record

  - picked-up : trip started

  - delivered : trip ended

  - b-start : break started

  - b-end : break ended

- For each trip, generate the start and end latitude & longitude

- For a given driver, for a given delivery, start is always less than end for both deliveries and break.

- columns - `driver_id`, `timestamp`, `flag`, `start_lon`, `start_lat`, `end_lon`, `end_lat`

- Midnight splits the day and deliveries spilling over 12.00am can be considered to be part of a new day.

- Subtract the break time from the final time spent.

- However, consider break time as final on-duty time.

- Categorize drivers in 4 club - (Maintain a SLIDING WINDOW of past 30 day)

  - PD (passive drivers): drivers with < 4 deliveries, but < 30 mins as total time spent (By default a driver is PD) and total distance travelled < 10

  - AD (active drivers): drivers with >= 30 mins time spent (drivers may have < 4 deliveries, but total time spent >= 30 mins) and total distance travelled >= 10

  - EAD (extra active drivers): drivers with >= 4 deliveries on in last 30 days and total distance travelled >= 30

  - UAD (ultra active drivers): drivers with >= 10 deliveries in last 30 days and total distance travelled >= 40

- **Results**:

  - For each driver, Analyse day-by-day usage clubs standings. For ex:

    - on 2018-06-01 --- user was EAD

    - on 2018-06-10 --- user became UAD

    - on 2018-06-11 --- user became PD ----> Usage cohort DOWNGRADED

  - And, What is the distribution among 4 categories on 2018-05-11?

    - Ex, PD: 40, AD: 20, EAD: 15, UAD: 5

---

## Assumptions:

### Time Period

- The time period starts from Aug 1, 2021 til Nov 30, 2021.

- Deliveries are done only on working/week days (Mon-Fri) - Driver Workdays.

- Driver Work in a 9AM-7PM shift on each work day.

### Break Periods

- Driver starts the day with the break period starting at reporting location and ending at first pickup location.

- The commute time from delivery at one location to the pickup at next location is integrated within the break period itself.

- Break starts as soon as a delivery is completed and lasts til next pickup.

- After each delivery, the break period is at least more than or equal to the commute time to the next pickup location.

- If the driver finishes all his deliveries for the day before 7PM, he will log the remaining time til end of shift as break period.

- In a given day, if there are no deliveries to make, the driver logs the entire day (9AM-7PM) as break period. For this start and end point remain the same (driver's depot).

### Geography

- pickup & delivery points are considered only in Bangalore area with lat=12.975118, lon=77.592690 at center and a radius ~25000 meter, for the simulation.

- A total of 5 depot locations are randomly generated within the ROI, and each driver is assigned any of the depot location randomly, which is constant throughout the time period.

- Each driver starts his day by reporting at his corresponding depot at 9AM, before moving to his first pickup of the day.

- Each starting point is randomly chosen within a 10km radius of depot of each driver.

- Each associated end point is randomly chosen within a radius of 5km from the starting point (pickup).

- Driver moves to the first closest pickup location from his depot, and after associated delivery at delivery location is completed, he moves to the next closest pickup location and so on, till he completes all his deliveries for the day.

- The routing from pickup to delivery location is based on either [OSRM](https://github.com/Project-OSRM/osrm-backend) routing api from [OSRM demo server](https://github.com/Project-OSRM/osrm-backend/wiki/Demo-server) or using the haversine distance formulae for the Bangalore region. User can choose either before running simulation.

- For each route, the api or the distance formula method provides the distance and time taken from start to end point. This is calculated for pickup-delivery location pair as well as, from delivery location of previous delivery to pickup location of next delivery.

- This helps in incorporating the commute time also in the break periods. Commute time is not reflected separately.

### Deliveries

- Each driver is assigned an 'N' (>=0) number of pickups for the day as the work starts

- Delivery of a consignment happens in pair, only after the current one is delivered, the driver can pickup next item.

- To adjust for the driver clubs, max number of deliveries in a day can be 3. So, the deliveries in a day range from 0 to 3 (due to simulation constraints).

- Each delivery number for the day for all drivers is randomly generated based on weights of 70, 20, 5 and 5 for 0, 1, 2, 3 deliveries respectively.

- Other parameters such as vehicle capacity, delivery demand and other complex metrics are not considered for simplicity.

---

## Folder structure of the project:

- Follow the `Readme.md` file in the root directory of the project folder.

- `environment.txt` file helps with setting up the project.

- `data/wip` folder will contain all the intermediate driver log files for each date for all drivers.

- And the reports would be saved in the `data` folder.

- Task related scripts are within the `scripts/driver_analytics` folder.

- And, the Jupyter Notebook for `EDA.ipynb` can be found in the `scripts` folder.

```bash
$ tree -h
.
├── Readme.md
├── data
│   ├── driver_clubs_report_by_date.feather
│   ├── drivers_breaks_by_date.feather
│   ├── drivers_delivery_by_date.feather
│   ├── drivers_delivery_log.feather
│   ├── drivers_engagement_clubs.feather
│   ├── drivers_master.feather
│   └── wip
│       ├── 2021-08-02.feather
│       ├── 2021-08-03.feather
│       ├── .....
│       ├── .....
│       ├── .....
│       ├── .....
│       ├── 2021-11-26.feather
│       ├── 2021-11-29.feather
│       └── 2021-11-30.feather
├── environment.txt
└── scripts
    ├── EDA.ipynb
    └── driver_analytics
        ├── __pycache__
        │   └── generate_data.cpython-39.pyc
        ├── driver_engagement.py
        └── generate_data.py

5 directories, 99 files

5 directories, 6 files
```

---

## Dependencies

Set up the code dependencies by running the following the terminal command.

```bash
conda create -n driver_analytics python=3.9
conda activate driver_analytics
pip install -r environment.txt
```

---

## Running the Code

To run the simulation and generate the driver analytics engagement report, follow below instructions:

- Open your terminal and change directory to project root directory.

```bash
cd driver_analytics
```

- From within the directory, just run the python script `driver_engagement.py`, and all the simulations
  and reports are generated.

```bash
python ./scripts/driver_analytics/driver_engagement.py
```

---

## Author

Nikhil S Hubballi

[Mail](mailto:nikhil.hubballi@gmail.com) | [LinkedIn](https://www.linkedin.com/in/nikhilhubballi/) | [Twitter](https://twitter.com/samashti_)

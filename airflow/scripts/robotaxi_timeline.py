import pandas as pd

robotaxi_data = [
    ["ARIZONA", "Phoenix", 1650070, "2022-06-26", "2022-09-19", None, None, "2022-09-17", None, "2025-10-12"],
    ["ARIZONA", "Mesa/Tempe", 989000, "2022-06-26", "2022-09-19", "2022-07-18", None, "2022-09-10", None, "2025-10-12"],
    ["CALIFORNIA", "Los Angeles", 3878704, "2024-03-18", "2024-07-02", None, "2024-07-30", "2024-09-04", "GeoFence", "2025-10-09"],
    ["CALIFORNIA", "San Diego", 1388320, "2024-03-18", "2024-07-02", None, "2024-07-30", "2024-09-04", "GeoFence", "2025-10-09"],
    ["CALIFORNIA", "San Francisco", 808988, "2019-03-18", "2019-07-18", None, "2024-07-30", "2024-09-04", "GeoFence", "2025-10-09"],
    ["CALIFORNIA", "Oakland", 433544, "2019-03-18", "2019-07-18", None, "2024-07-30", "2024-09-04", "GeoFence", "2025-10-09"],
    ["CALIFORNIA", "San Jose", 969655, "2019-03-18", "2019-07-18", None, "2024-07-30", "2024-09-04", "GeoFence", "2025-10-09"],
    ["COLORADO", "Denver", 715891, "2022-10-13", None, None, None, None, None, None],
    ["FLORIDA", "Jacksonville", 1009833, None, None, None, None, "2024-08-05", None, None],
    ["FLORIDA", "Miami", 464655, None, None, None, None, "2024-08-05", None, None],
    ["FLORIDA", "Orlando", 325044, None, None, None, None, "2024-08-05", None, None],
    ["FLORIDA", "Tampa", 413657, None, None, None, None, "2024-08-05", None, None],
    ["ILLINOIS", "Chicago", 2661089, None, None, None, None, "2024-07-31", None, None],
    ["NEVADA", "Las Vegas", 670352, "2022-09-03", "2022-09-11", "2022-08-25", None, None, None, "2025-10-12"],
    ["NEW YORK", "Brooklyn", 2648943, None, None, None, None, "2024-08-05", None, None],
    ["NEW YORK", "Queens", 2405027, None, None, None, None, "2024-08-05", None, None],
    ["TEXAS", "Austin", 989252, "2021-05-27", "2021-08-06", "2021-06-22", None, "2021-07-14", None, "2025-10-02"],
    ["TEXAS", "Dallas", 1302868, "2021-05-27", "2021-08-06", "2021-09-23", None, None, None, "2025-10-02"],
    ["TEXAS", "Houston", 2324082, "2021-05-27", "2021-08-06", "2021-07-30", None, None, None, "2025-10-02"],
    ["TEXAS", "San Antonio", 1526697, "2021-05-27", "2021-08-06", "2021-07-30", None, None, None, "2025-10-02"]
]

df_robotaxi = pd.DataFrame(robotaxi_data, columns=[
    "State", "City", "Population", "Tesla_Insurance_Available",
    "Permit_Applied", "Permit_Received", "Vehicle_Operated_Ads",
    "Public_Test", "GeoFence", "Regulatory_Approval"
])

df_robotaxi["scrape_timestamp"] = pd.Timestamp.utcnow()
df_robotaxi.to_csv('robotaxi_timeline.csv')

import mysql.connector
import requests
import pandas as pd
import numpy as np
from faker import Faker
import gower
import kmedoids
import ast
import json
from datetime import datetime



db_config = {
    'user': 'pentagon',
    'password': 'pentagon',
    'host': 'cathaytest.cdkkwaamk0zl.ap-southeast-1.rds.amazonaws.com',
    'database': 'Cathay-pentagon'
}


def table_exists(cursor, table_name):
    query = "SHOW TABLES LIKE %s"
    cursor.execute(query, (table_name,))
    result = cursor.fetchone()
    return bool(result)

gemini_instructions = """
RULES:
1. Only generate the requested item. Do not say anything else. 
2. Be concise. 
3. Do not hallucinate and give untrue information.\n
"""
gemini_link = "https://developers.cathaypacific.com/hackathon-apigw/hackathon-middleware/v1/vertex-ai/google-gemini"
api_key = "0Ws2MAmAseTl39JZLohswZZgWLCxpZ1K"
gemini_header = {
    'apiKey': api_key,
    'Content-Type': 'application/json',
    'User-Agent': 'PostmanRuntime/7.42.0',
    'Accept': '*/*',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
}


def get_gemini_response(query: str) -> dict:
    gemini_body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": gemini_instructions + query
                    }
                ]
            }
        ]
    }
    gemini_response = requests.post(gemini_link, 
                                    headers=gemini_header,
                                    json=gemini_body)
    if gemini_response.status_code == 200:
        return gemini_response.json()
    return None


def generate_notification(activity: dict, cust_segment: dict, extra_criteria: dict) -> str:
    # generate a notification promoting the given activity to the given customer segment
    notif_instruction = (f"Generate 5 different personalized notifications promoting the given activity to the given customer segment.\n" +
                         f"Our goal is to make the notifications engaging and relevant to the customer segment with the given properties.\n" +
                         f"Be creative! Add emojis to make the notifications more engaging.\n" +
                         f"Each notification should have a title and a description with in total less than 178 characters each.\n" +
                         f"Generate the notification in the following format:\n" +
                         f"1.\n"
                         f"Title: [title]\n" +
                         f"Description: [description]\n" +
                         f"2.\n" +
                         f"Title: [title]\n" +
                         f"Description: [description]\n...")
    extra_criteria_str = "".join([f"{str.upper(k)}: {str(extra_criteria[k])}\n" for k in extra_criteria.keys()])
    notif_query = (f"ACTIVITY: \n{activity}\n\n" + 
                   f"CUSTOMER SEGMENT: \n{cust_segment}\n" +
                   extra_criteria_str + "\n" +
                   f"INSTRUCTION: \n{notif_instruction}")
    # print(notif_query)
    gemini_response = get_gemini_response(notif_query)
    if gemini_response:
        return gemini_response['candidates'][0]['content']['parts'][0]['text']
    return None


def regenerate_seg2act_mapping():
    # generate a mapping of customer segments to activities
    # for each activity, we go through all customer segments and decide if the activity is relevant to the segment
    db = mysql.connector.connect(**db_config) 
    cursor = db.cursor()
    cursor.execute("SELECT * FROM tbl_activities JOIN tbl_activities_tags ON tbl_activities.act = tbl_activities_tags.activity_id")


def create_tbl_customers():
    db = mysql.connector.connect(**db_config)
    cursor = db.cursor()
    if table_exists(cursor, "tbl_customers"):
        print("Table tbl_customers already exists.")
        return
    sql = """CREATE TABLE `tbl_customers` (
  `cust_id` int NOT NULL AUTO_INCREMENT,
  `cust_first_name` varchar(100) DEFAULT NULL,
  `cust_last_name` varchar(100) DEFAULT NULL,
  `cust_date_of_birth` date DEFAULT NULL,
  `cust_age_group` varchar(45) DEFAULT NULL,
  `cust_member_tier` int DEFAULT NULL,
  `cust_language` varchar(45) DEFAULT NULL,
  `cust_travel_spending` int DEFAULT NULL COMMENT 'Money spent on travelling last year',
  `cust_travel_group` enum('Single','Couple','Family','Friends') DEFAULT NULL,
  `cust_income_level` int DEFAULT NULL,
  `cust_location` varchar(45) DEFAULT NULL,
  `cust_flying_frequency` varchar(45) DEFAULT NULL,
  `cust_miles_spending_on_food` int DEFAULT NULL COMMENT 'Miles spent last year',
  `cust_miles_spending_on_travelling` int DEFAULT NULL,
  `cust_miles_spending_on_shopping` int DEFAULT NULL,
  PRIMARY KEY (`cust_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
"""
    cursor.execute(sql)
    db.commit()
    print("Table tbl_customers created.")
    cursor.close()
    db.close()
    return


def drop_tbl_customers():
    db = mysql.connector.connect(**db_config)
    cursor = db.cursor()
    if not table_exists(cursor, "tbl_customers"):
        print("Table tbl_customers does not exist.")
        return
    sql = "DROP TABLE tbl_customers"
    cursor.execute(sql)
    db.commit()
    print("Table tbl_customers dropped.")
    cursor.close()
    db.close()
    return


# A function that generates many random customers and inserts them into the tbl_customers table
# This is hard coded to follow the current tbl_customers schema for now. 
# If the schema changes, this function will need to be updated.

def repopulate_tbl_customers(count: int):
    # Constants:
    age_groups = ["18-25", "26-35", "36-50", "51+"]
    languages = ["English", "Chinese"]
    locations = ["Hong Kong", "Taiwan", "Mainland China"]
    flying_frequencies = ["Less than once a year", "Once a year", "Twice a year", "More than twice a year"]
    travel_groups = ["Single", "Couple", "Family", "Friends"]
    # Begin repopulation
    df = pd.DataFrame()
    fake = Faker('zh_CN')
    firstnames = []
    lastnames = []
    for _ in range(count):
        fakename = fake.name()
        lastnames.append(fakename[0])
        firstnames.append(fakename[1:])
    df['cust_first_name'] = firstnames
    df['cust_last_name'] = lastnames
    df['cust_age_group'] = np.random.choice(age_groups, size=count)
    df['cust_member_tier'] = np.random.randint(1, 5, size=count)
    df['cust_language'] = np.random.choice(languages, size=count)
    df['cust_travel_spending'] = np.random.randint(100, 10000, size=count)
    df['cust_travel_group'] = np.random.choice(travel_groups, size=count)
    df['cust_income_level'] = np.random.randint(1, 5, size=count)
    df['cust_location'] = np.random.choice(locations, size=count)
    df['cust_flying_frequency'] = np.random.choice(flying_frequencies, size=count)
    df['cust_miles_spending_on_food'] = np.random.randint(0, 10000, size=count)
    df['cust_miles_spending_on_travelling'] = np.random.randint(0, 10000, size=count)
    df['cust_miles_spending_on_shopping'] = np.random.randint(0, 10000, size=count)
    df = df.astype({
        'cust_age_group': str,
        'cust_member_tier': int,
        'cust_language': str,
        'cust_travel_spending': int,
        'cust_travel_group': str,
        'cust_income_level': int,
        'cust_location': str,
        'cust_flying_frequency': str,
        'cust_miles_spending_on_food': int,
        'cust_miles_spending_on_travelling': int,
        'cust_miles_spending_on_shopping': int
        })
    db = mysql.connector.connect(**db_config)
    cursor = db.cursor()
    sql = f"""INSERT INTO tbl_customers (cust_first_name, cust_last_name, cust_age_group, cust_member_tier, cust_language, cust_travel_spending, cust_travel_group, cust_income_level, cust_location, cust_flying_frequency, cust_miles_spending_on_food, cust_miles_spending_on_travelling, cust_miles_spending_on_shopping) 
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    data = [tuple(map(lambda x: int(x) if isinstance(x, np.int64) else x, df.iloc[i])) for i in range(count)]
    cursor.executemany(sql, data)
    db.commit()
    cursor.close()
    db.close()
    return


# # Repopulate the customers table with 4000 random customers
# drop_tbl_customers()
# create_tbl_customers()
# repopulate_tbl_customers(4000)


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import DistanceMetric
def calculate_gower_distance(df):
  
    '''
    Takes a dataframe as an input and returns a gower distance matrix.
    code taken from https://datascience.stackexchange.com/questions/8681/clustering-for-mixed-numeric-and-nominal-discrete-data
    '''

    variable_distances = []
    for col in range(df.shape[1]):
        feature = df.iloc[:,[col]]
        if feature.dtypes.values == object:
            feature_dist = DistanceMetric.get_metric('dice').pairwise(pd.get_dummies(feature, drop_first=True))
    else:
        feature_dist = DistanceMetric.get_metric('manhattan').pairwise(feature) / max(np.ptp(feature.values),1)
        variable_distances.append(feature_dist)
    return np.array(variable_distances).mean(0)


def create_tbl_cust2seg_mapping():
    sql = """CREATE TABLE `tbl_cust2seg_mapping` (
  `cust_id` int NOT NULL,
  `seg_id` int NOT NULL,
  PRIMARY KEY (`cust_id`,`seg_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
"""
    db = mysql.connector.connect(**db_config)
    cursor = db.cursor()
    if table_exists(cursor, "tbl_cust2seg_mapping"):
        print("Table tbl_cust2seg_mapping already exists.")
        return
    print("TTEST")
    cursor.execute(sql)
    db.commit()
    cursor.close()
    db.close()
    return

def truncate_tbl_cust2seg_mapping():
    db = mysql.connector.connect(**db_config)
    cursor = db.cursor()
    if not table_exists(cursor, "tbl_cust2seg_mapping"):
        print("Table tbl_cust2seg_mapping does not exist.")
        return
    sql = "TRUNCATE TABLE tbl_cust2seg_mapping"
    cursor.execute(sql)
    db.commit()
    cursor.close()
    db.close()
    return

def repopulate_tbl_cust2seg_mapping(df):
    db = mysql.connector.connect(**db_config)
    cursor = db.cursor()
    sql = f"""INSERT INTO tbl_cust2seg_mapping (cust_id, seg_id) 
    VALUES (%s, %s)"""
    data = [(i, int(df['cluster'][i])) for i in range(len(df))]
    cursor.executemany(sql, data)
    db.commit()
    cursor.close()
    db.close()
    return


# NAIVE CLUSTERING BEGINS HERE


def gemini_define_clusters(medoids: list):
    # define the clusters based on the medoids
    # return a list of dictionaries, each dictionary contains the cluster number and the medoid
    cluster_instruction = (f"Give a meaning and definition to each of the provided clusters of customers.\n" +
                           f"Each cluster is represented by a medoid, which is a customer that is most representative of the cluster.\n" +
                           f"Describe any notable characteristics of the customers in each cluster based on the medoid.\n" +
                           f"Give the cluster a name based on the description.\n" +
                           f"The schema of the medoid is as following: \n" +
                           f"[age_group, member_tier, travel_spending_per_year, travel_group, income_level, flying_frequency, miles_spent_on_travelling, miles_spent_on_food, miles_spent_on_shopping]\n\n" +
                           f"Write it in the format of:\n" +
                           f"Name: [name1]\n" +
                           f"Description: [description1]\n" +
                           f"Name: [name2]\n" + 
                           f"Description: [description2]\n" +
                           f"Name: [name3]\n" + 
                           f"...\n\n" +
                           f"Here are a list of the 20 medoids:\n" +                           
                           f"{medoids}") 
    # print(cluster_instruction)
    gemini_response = get_gemini_response(cluster_instruction)
    if gemini_response:
        return gemini_response['candidates'][0]['content']['parts'][0]['text']
    return None

NUM_OF_CLUSTERS = 20

def define_clusters(meds: pd.DataFrame) -> list:
    m = meds.drop("cluster", axis=1)
    cluster_info = gemini_define_clusters(m.values).splitlines()
    # print("Cluster_info: \n", cluster_info)
    # print("len: ", len(cluster_info))
    info_lines = (len(cluster_info)+1) // NUM_OF_CLUSTERS
    # print("Info lines: ", info_lines)
    info_list = [(cluster_info[i][6:], cluster_info[i+1][13:]) for i in range(0, len(cluster_info), info_lines)]
    return info_list

def create_tbl_segments():
    db = mysql.connector.connect(**db_config)
    cursor = db.cursor()
    if table_exists(cursor, "tbl_segments"):
        print("Table tbl_segments already exists.")
        return
    sql = """CREATE TABLE `tbl_segments` (
  `seg_id` int NOT NULL,
  `seg_name` varchar(45) DEFAULT NULL,
  `seg_desc` varchar(500) DEFAULT NULL,
  PRIMARY KEY (`seg_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci 
COMMENT='List of defined (preferably through clustering algorithms, but for now predetermined) customer segments. ';
"""
    cursor.execute(sql)
    db.commit()
    print("Table tbl_segments created.")
    cursor.close()
    db.close()
    return

def drop_tbl_segments():
    db = mysql.connector.connect(**db_config)
    cursor = db.cursor()
    if not table_exists(cursor, "tbl_segments"):
        print("Table tbl_segments does not exist.")
        return
    sql = "DROP TABLE tbl_segments"
    cursor.execute(sql)
    db.commit()
    print("Table tbl_segments dropped.")
    cursor.close()
    db.close()
    return

def repopulate_tbl_segments(medoids: pd.DataFrame):
    db = mysql.connector.connect(**db_config)
    cursor = db.cursor()
    sql = f"""INSERT INTO tbl_segments (seg_id, seg_name, seg_desc) 
    VALUES (%s, %s, %s)"""
    m = medoids.values
    data = [(m[i][-3], m[i][-2], m[i][-1]) for i in range(len(medoids))]
    cursor.executemany(sql, data)
    db.commit()
    cursor.close()
    db.close()
    return


# A very basic, naive example of clustering customers based on their attributes
# Fetch customer data from the database
def cluster_customers():
    db = mysql.connector.connect(**db_config)
    cursor = db.cursor()
    cursor.execute("SELECT * FROM tbl_customers")
    data = cursor.fetchall()
    cursor.close()
    print(len(data)) 
    data = list(zip(*data))

    df_dict = {
        'age_group': data[4],
        'member_tier': data[5],
        'travel_spending': data[7],
        'travel_group': data[8],
        'income_level': data[9],
        'flying_frequency': data[11],
        'miles_on_travelling': data[12],
        'miles_on_food': data[13],
        'miles_on_shopping': data[14]
    }

    df = pd.DataFrame(df_dict)

    # Convert categorical data to numerical data
    df_encoded = pd.get_dummies(df)

    # Normalize the data
    scaler = StandardScaler()
    df_normalized = scaler.fit_transform(df_encoded)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=20, random_state=42)
    df['cluster'] = kmeans.fit_predict(df_normalized)

    # Print the resulting clusters
    print(df)
    distmatrix = calculate_gower_distance(df)
    print(distmatrix)
    print(len(distmatrix), len(distmatrix[0]))
    # import kmedoids
    c = kmedoids.fasterpam(distmatrix, 20)
    medoids = df.iloc[c.medoids]
    # cluster_i = medoids['cluster'].to_list()
    # cluster_i.sort()
    # print(cluster_i)
    create_tbl_cust2seg_mapping()
    truncate_tbl_cust2seg_mapping()
    repopulate_tbl_cust2seg_mapping(df)
    income_levels = ["0-10k", "10k-25k", "25k-50k", "50k+"]
    member_tiers = ["green", "silver", "gold", "diamond"]
    medoids.loc[:, "income_level"] = list(map(lambda a: income_levels[a-1], medoids["income_level"].to_list()))
    medoids.loc[:, "member_tier"] = list(map(lambda a: member_tiers[a-1], medoids["member_tier"].to_list()))
    # print(medoids)
    medoid_names, medoid_descs = list(zip(*define_clusters(medoids)))
    # for i, desc in enumerate(medoid_descriptions):
    #     print(f"Cluster {i}: {desc}")
    # insert medoid descriptions into the medoids dataframe
    medoids.loc[:, "cluster_name"] = medoid_names
    medoids.loc[:, "cluster_desc"] = medoid_descs
    print("FINAL MEDOIDS: \n", medoids)
    # Now we insert this into tbl_segments:



# KEY FEATURE: RELEVANCY MATCHING BETWEEN CUSTOMER SEGMENTS & ACTIVITIES

def relevancy_review(act, seg):
    promote_activity_dict = dict()
    for i in range(20):
        promote_activity_dict[i] = []
    review_instruction = (f"Based on the given activity and all the given customer segments, \n" +
                          f"Review and decide whether we should promote that activity to each customer segment. \n" +
                          f"You should ONLY promote if you believe that customer segment will definitely enjoy the activity. \n" +
                          f"The decision to promote will be expensive, so do not promote everything. \n" +
                          f"The information of the activity will be given in the following format: \n" +
                          f"(id, name, description, price in HKD, rating out of 10, date, duration) \n" +
                          f"The information of each customer segment will be given in the following format: \n" +
                          f"(id, name, description) \n\n" +
                          f"Write it in the format of: \n" +
                          f"[1, 4, 15, 18] (a list of segment ids that the activity should be promoted to) \n")
    for a in act:
        review_query = (f"ACTIVITY: \n{a}\n\n" +
                        f"CUSTOMER SEGMENTS: \n{seg}\n\n" +
                        f"INSTRUCTION: \n{review_instruction}")
        gemini_response = get_gemini_response(review_query)
        if gemini_response:
            gemini_response = gemini_response['candidates'][0]['content']['parts'][0]['text'][:-1]
            res_list = ast.literal_eval(gemini_response)
            for i in res_list:
                promote_activity_dict[i].append(a[0])
        print(gemini_response)
    return promote_activity_dict


CATHAY_WEEKLY_LIMIT = 3     # IMPORTANT RESTRICTION

def main():
    db = mysql.connector.connect(**db_config)
    cursor = db.cursor()
    cursor.execute("SELECT * FROM tbl_activities ORDER BY act_id")
    activities_all = cursor.fetchall()
    cursor.execute("SELECT * FROM tbl_segments ORDER BY seg_id")
    segments_all = cursor.fetchall()
    cursor.close()
    db.close()

    print(activities_all)
    print(segments_all)
    act_dict = relevancy_review(activities_all, segments_all)
    # We will choose the first activity here to promote.
    # In real scenario, the activity is chosen by the employee. 
    all_segment_notifs = dict()
    for k in act_dict.keys():
        print(f"Promoting to customer segment {k} ({segments_all[k][1]}), \n",
              f"The following activity ids are suitable: {act_dict[k]}")
        if len(act_dict[k]) == 0:
            print(f"No activities to promote to customer segment {k}. ")
            continue
        for i in range(min(CATHAY_WEEKLY_LIMIT, len(act_dict[k]))):
            print(f"Promoting activity {activities_all[act_dict[k][i]-1]} to customer segment {k}")
            act_details = activities_all[act_dict[k][i]-1][1:]
            act_details = dict(zip(['name', 'description', 'price', 'rating out of 10', 'date', 'duration'], act_details))
            extra_criteria = {}     
            # Add any extra criteria here in extra_criteria for the prompt to consider during prompt generation, e.g. age, gender, etc.
            ans = generate_notification(act_details, segments_all[k], extra_criteria)
            formatted = ans.splitlines()
            notif_lines = (len(formatted)+1) // 5
            notif_list = [{"Title": formatted[i+1][7:], 
                           "Description": formatted[i+2][13:]} 
                           for i in range(0, len(formatted), notif_lines)]
            # for notif in notif_list:
            #     print(notif)
            all_segment_notifs[k] = notif_list
    print(all_segment_notifs)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"customer_data_{current_time}.txt"

    # Save the generated notifications to a file
    # Note that the emoji characters may not be displayed correctly in the file.
    with open(file_name, 'w') as file:
        json.dump(all_segment_notifs, file, indent=4)

main()
    
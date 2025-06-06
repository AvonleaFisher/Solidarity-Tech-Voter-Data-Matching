import pandas as pd
from nicknames import NickNamer
from fuzzywuzzy import fuzz
import numpy as np
import re

voterfile = pd.read_parquet( "/Users/avonleafisher/Downloads/AllNYSVoters_20250402/voterfile.parquet", engine="pyarrow")

# Load Solidarity Tech file, ensuring ZIP code is a string
sol_tech = pd.read_csv("/Users/avonleafisher/Downloads/AllNYSVoters_20250402/solidarity_tech_unknown.csv", dtype={"solidarity_tech_zip5": str})

print(len(sol_tech))
print(len(voterfile))

# Ensure ZIP codes are strings and take first 5 characters
voterfile["RZIP5"] = voterfile["RZIP5"].astype(str).str[:5]
sol_tech["solidarity_tech_zip5"] = sol_tech["solidarity_tech_zip5"].astype(str).str[:5]

# Convert names to uppercase for case-insensitive matching
voterfile["LASTNAME"] = voterfile["LASTNAME"].str.upper()
voterfile["FIRSTNAME"] = voterfile["FIRSTNAME"].str.upper()
sol_tech["solidarity_tech_last_name"] = sol_tech["solidarity_tech_last_name"].str.upper()
sol_tech["solidarity_tech_first_name"] = sol_tech["solidarity_tech_first_name"].str.upper()

# Match on last name + ZIP
match_ln = sol_tech.merge(
    voterfile,
    left_on=["solidarity_tech_last_name", "solidarity_tech_zip5"],
    right_on=["LASTNAME", "RZIP5"],
    how="inner"
)

# Match on first name + ZIP
match_fn = sol_tech.merge(
    voterfile,
    left_on=["solidarity_tech_first_name", "solidarity_tech_zip5"],
    right_on=["FIRSTNAME", "RZIP5"],
    how="inner"
)

# Combine match results
match_results_df = pd.concat([match_ln, match_fn], ignore_index=True)



# Select and rename relevant columns
match_results_df = match_results_df.rename(columns={
    "solidarity_tech_address2": "solidarity_tech_apt_num",
    "FIRSTNAME": "voter_first",
    "LASTNAME": "voter_last",
    "RADDNUMBER": "voter_house_number",
    "RSTREETNAME": "voter_street_name",
    "RAPARTMENT": "voter_apt_num",
    "RZIP5": "voter_zip",
    "SBOEID": "voter_boe_id",
    "STATUS": "registration_status",
    "ENROLLMENT": "party_enrollment",
    "MAILADD1": "voter_mailing_address1",
    "MAILADD2": "voter_mailing_address2",
    "MAILADD3": "voter_mailing_address3",
    "MAILADD4": "voter_mailing_address4",
    "DOB": "voter_dob",
    "PREVNAME": "voter_previous_name",
    "PREVADDRESS": "voter_previous_address"
})[[
    "solidarity_tech_id", "solidarity_tech_first_name", "solidarity_tech_last_name", 
    "solidarity_tech_address1", "solidarity_tech_apt_num", "solidarity_tech_zip5", 
    "solidarity_tech_phone_number", "solidarity_tech_email", "voter_first", "voter_last", 
    "voter_house_number", "voter_street_name", "voter_apt_num", "voter_zip", "voter_boe_id", 
    "registration_status", "party_enrollment", "voter_mailing_address1", 
    "voter_mailing_address2", "voter_mailing_address3", "voter_mailing_address4", 
    "voter_dob", "voter_previous_name", "voter_previous_address"
]]


# Save to CSV
match_results_df.to_csv(
    "/Users/avonleafisher/Downloads/AllNYSVoters_20250402/match_results_fn_and_ln.csv",
    index=False
)





df=pd.read_csv("/Users/avonleafisher/Downloads/AllNYSVoters_20250402/match_results_fn_and_ln.csv")

#uppercase string cols
df = df.apply(lambda x: x.str.upper() if x.dtype == 'object' else x)

# create house # and street name cols from address1 col, convert to uppercase
df[['solidarity_tech_house_num', 'solidarity_tech_street_name']] = df['solidarity_tech_address1']\
    .str.extract(r'^(\d+)\s+(.*)', expand=True)

df['solidarity_tech_house_num'] = df['solidarity_tech_house_num']
df['solidarity_tech_street_name'] = df['solidarity_tech_street_name']

# Define common suffixes to remove when isolated
suffixes = r'\s+(STREET|ST|AVENUE|AVE|BLVD|BOULEVARD|DRIVE|DR|ROAD|RD|PARKWAY|PKWY|LANE|LN|COURT|CT|TERRACE|TER|PLACE|PL|CIRCLE|CIR|HIGHWAY|HWY|WAY|SQ|SQUARE|EXPY|EXPRESSWAY|STATION|STN|ALLEY|ALY|PLAZA|PLZ|LOOP)\b'

# Define common directional replacements
directions = {
    r'\bEAST\b': 'E',
    r'\bWEST\b': 'W',
    r'\bNORTH\b': 'N',
    r'\bSOUTH\b': 'S'
}

# Define function for advanced normalization
def normalize_street_name(street_name):
    if pd.isna(street_name):
        return street_name  # Keep NaNs as they are

    # Standardize directional prefixes (e.g., EAST 117 -> E 117)
    for full, abbrev in directions.items():
        street_name = re.sub(full, abbrev, street_name, flags=re.IGNORECASE)

    # Handle numbered streets (e.g., 117th -> 117)
    street_name = re.sub(r'(\d+)(TH|ST|ND|RD)\b', r'\1', street_name, flags=re.IGNORECASE)

    # Remove common street suffixes only when they are isolated
    street_name = re.sub(suffixes, '', street_name, flags=re.IGNORECASE).strip()

    return street_name

# Apply the function to both street name columns
df['solidarity_tech_street_name_clean'] = df['solidarity_tech_street_name'].apply(normalize_street_name)
df['voter_street_name_clean'] = df['voter_street_name'].apply(normalize_street_name)


def clean_apt_num(apt_num):
    if pd.isna(apt_num):
        return None
    # Remove any non-alphanumeric characters (except digits and letters) and convert to uppercase
    apt_num = re.sub(r'[^a-zA-Z0-9]', '', apt_num).upper()

    # Remove terms like 'APARTMENT', 'UNIT', 'APT', 'NUM' at the beginning
    apt_num = re.sub(r'^(APARTMENT|UNIT|APT|NUM)', '', apt_num)

    # Normalize floor-related terms (e.g., 'FL', 'FLOOR', 'PH', 'PENTHOUSE')
    apt_num = re.sub(r'\b(FLOOR|FL|FLR|1ST|2ND|3RD|UPPER|LOWER|PENTHOUSE|PH|BASEMENT|B)\b', '', apt_num)

    # Handle variations like 'UNIT 4C' -> '4C' and 'C4' -> '4C'
    apt_num = re.sub(r'UNIT\s*(\d+[A-Za-z]*)', r'\1', apt_num)
    apt_num = re.sub(r'(\d+)([A-Za-z]{1})', r'\1\2', apt_num)  # Normalize 'C4' -> '4C'

    # Remove any excess spaces after stripping terms
    apt_num = apt_num.strip()

    return apt_num

# Apply cleaning function to apt # cols
df['clean_solidarity_tech_apt_num'] = df['solidarity_tech_apt_num'].apply(clean_apt_num)
df['clean_voter_apt_num'] = df['voter_apt_num'].apply(clean_apt_num)

# Create col to check if the cleaned apartment numbers match
df['apt_num_match'] = df['clean_solidarity_tech_apt_num'] == df['clean_voter_apt_num']

# If 'solidarity_tech_apt_num' is null, set 'apt_num_match' to "No apt # in Solidarity Tech"
df.loc[df['solidarity_tech_apt_num'].isna(), 'apt_num_match'] = "No apt # in Solidarity Tech"


# 1. Last name exact match
df['last_name_match'] = (df['solidarity_tech_last_name'].fillna('') == df['voter_last'].fillna('')).astype(int)

# 2. Fuzzy last name match
df['fuzzy_last_name_match'] = df.apply(
    lambda row: fuzz.partial_ratio(str(row['solidarity_tech_last_name'] or ''), str(row['voter_last'] or '')) >= 80,
    axis=1
).astype(int)

# 3. First name exact match
df['first_name_match'] = (df['solidarity_tech_first_name'].fillna('') == df['voter_first'].fillna('')).astype(int)

# 4. Nickname match
nn = NickNamer()
def is_nickname_match(name1, name2):
    if pd.isna(name1) or pd.isna(name2):  # Handle NaN values
        return False
    name1, name2 = str(name1).strip(), str(name2).strip()
    return name1 == name2 or name2 in nn.nicknames_of(name1) or name1 in nn.nicknames_of(name2)

df['nickname_match'] = df.apply(lambda row: is_nickname_match(row['solidarity_tech_first_name'], row['voter_first']), axis=1).astype(int)

# 5. Street Name Match
df['street_name_match'] = (df['solidarity_tech_street_name_clean'].fillna('') == df['voter_street_name_clean'].fillna('')).astype(int)

# 6. House Number Match
df['house_num_match'] = (df['solidarity_tech_house_num'].fillna('') == df['voter_house_number'].fillna('')).astype(int)

# 7. Apartment Number Match
df['apt_num_match'] = (df['clean_solidarity_tech_apt_num'].fillna('') == df['clean_voter_apt_num'].fillna('')).astype(int)

# 8. Compute match strength
match_columns = ['nickname_match', 'last_name_match', 'fuzzy_last_name_match',
                 'first_name_match', 'street_name_match', 'house_num_match', 'apt_num_match']
df['match_strength'] = df[match_columns].sum(axis=1)

# 9. If match_strength <=2 1, remove voter columns
columns_to_null = ['voter_first', 'voter_last', 'voter_house_number', 'voter_street_name_clean',
                   'voter_apt_num', 'voter_zip', 'voter_boe_id', 'registration_status',
                   'party_enrollment', 'voter_mailing_address1', 'voter_mailing_address2',
                   'voter_mailing_address3', 'voter_mailing_address4', 'voter_dob', 'voter_previous_name',
                   'voter_previous_address']

df.loc[df['match_strength'] <= 1, columns_to_null] = None

# 10. Remove duplicate rows where match_strength == 0
df_no_match = df[df['match_strength'] == 0].drop_duplicates(subset='solidarity_tech_id', keep='first')

# 11. Merge back high-confidence matches
df_matched = df[df['match_strength'] > 0]
df = pd.concat([df_matched, df_no_match])

# 12. Assign highest match strength per ID
df['is_highest_match_strength'] = df.groupby('solidarity_tech_id')['match_strength'].transform(lambda x: x == x.max()).astype(int)

print(df.columns)
# Define column order with matched fields next to each other
new_column_order = [
    'solidarity_tech_id', 'solidarity_tech_phone_number', 
    'solidarity_tech_first_name', 'voter_first',
    'solidarity_tech_last_name', 'voter_last',
    # Address Columns
    'solidarity_tech_address1', 'voter_house_number',
    'solidarity_tech_house_num', 'voter_house_number',  # Reordered to keep house number together
    'solidarity_tech_street_name', 'voter_street_name',
    'solidarity_tech_street_name_clean', 'voter_street_name_clean',
    'clean_solidarity_tech_apt_num', 'clean_voter_apt_num', 
    'solidarity_tech_apt_num', 'voter_apt_num',  # Reordered for apartment number
    'solidarity_tech_zip5', 'voter_zip',  # Zip codes grouped together
    # Email & Registration
    'solidarity_tech_email',
    'voter_boe_id', 'registration_status', 'party_enrollment',
    'voter_mailing_address1', 'voter_mailing_address2',
    'voter_mailing_address3', 'voter_mailing_address4', 
    'voter_dob', 'voter_previous_name', 'voter_previous_address',
    # Matching Criteria
    'last_name_match', 'fuzzy_last_name_match', 'first_name_match',
    'nickname_match', 'street_name_match', 'house_num_match', 'apt_num_match',
    'match_strength', 'is_highest_match_strength'
]

# Reorder columns in DataFrame
df = df[new_column_order]

# Reorder the DataFrame columns
df = df[new_column_order]
df=df.drop_duplicates(keep='first')

# Ensure consistent data types and handle NaN values
for col in ['solidarity_tech_first_name', 'voter_first', 'solidarity_tech_last_name', 
            'voter_last', 'solidarity_tech_zip5', 'voter_zip']:
    df[col] = df[col].astype(str).fillna('')

# Filter to keep only rows where 'is_highest_match_strength' is 1
df = df[df['is_highest_match_strength'] == 1]

# Create a temporary column to count non-null values in each row
df['_non_null_count'] = df.notnull().sum(axis=1)

# Sort by 'match_strength' (descending) and '_non_null_count' (descending) to prioritize most complete rows
df = df.sort_values(by=['match_strength', '_non_null_count'], ascending=[False, False])

#label duplicates
df['is_duplicate'] = df.duplicated(subset=['solidarity_tech_id'], keep='first').astype(int)

# Generate ZIP code ranges for each borough
manhattan_zips = list(range(10001, 10293))        # 10001–10292
brooklyn_zips = list(range(11201, 11240)) + [11249]
queens_zips = (
    list(range(11004, 11006)) +    # Still keep this
    list(range(11101, 11121)) +
    list(range(11351, 11698)) +
    [11001]  # Add this if you want to include it
)
bronx_zips = list(range(10451, 10476))            # 10451–10475
staten_island_zips = list(range(10301, 10315))    # 10301–10314

# Combine all borough ZIPs and convert to 5-character strings
nyc_zip_codes = [
    str(zipcode).zfill(5) for zipcode in (
        manhattan_zips + brooklyn_zips + queens_zips + bronx_zips + staten_island_zips
    )
]

#clean zips
df['solidarity_tech_zip5'] = (
    df['solidarity_tech_zip5']
    .astype(str)
    .str.strip()
    .str.replace(".0", "", regex=False)
    .str.split("-").str[0]
    .str.zfill(5)
)


#Flag exact full match
df['exact_full_match'] = (
    (df['solidarity_tech_first_name'] == df['voter_first']) &
    (df['solidarity_tech_last_name'] == df['voter_last']) &
    (df['solidarity_tech_street_name_clean'] == df['voter_street_name_clean'])
)

#Find ambiguous IDs (same ID with multiple match_strength >= 5)
match_counts = df[df['match_strength'] >= 5].groupby('solidarity_tech_id').size()
ambiguous_ids = match_counts[match_counts > 1].index

# Assign match_types
def assign_match_type(row):
    zip5 = row.get("solidarity_tech_zip5", "")
    if pd.notna(zip5) and zip5 not in nyc_zip_codes:
        return "Ineligible: Address outside of NYC"
    if row['solidarity_tech_id'] in ambiguous_ids:
        return "Cannot be determined"
    if row['exact_full_match']:
        return "Perfect match"
    if row['match_strength'] >= 5:
        return "Strong match"
    if 2 <= row['match_strength'] <= 4:
        return "Weak match"
    return "No match"

df['match_type'] = df.apply(assign_match_type, axis=1)

df['match_type'] = df.apply(
    lambda row: 'No match'
    if (
        (
            (row['first_name_match'] == 0 and row['last_name_match'] == 0) or
            (row['first_name_match'] == 0 and row['fuzzy_last_name_match'] == 0) or
            (row['nickname_match'] == 0 and row['last_name_match'] == 0)
        )
        and row.get('street_name_match', 0) != 1
    )
    else row['match_type'],
    axis=1
)

party_mapping = {
    "DEM": "Democrat",
    "BLK": "Unaffiliated",
    "WOR": "WFP",
    "REP": "Republican",
    "OTH": "Not registered"
}
df['phone_number'] = df['solidarity_tech_phone_number']
df["Registration Status"] = df["party_enrollment"].map(party_mapping).fillna("Cannot Be Determined")
df.loc[df['match_type'] == 'Ineligible: Address outside of NYC', "Registration Status"] = 'Not eligible'
df.loc[df['match_type'] == 'No match', "Registration Status"] = 'Not Registered'


df.to_csv("/Users/avonleafisher/Downloads/AllNYSVoters_20250402/final_combined_matches.csv", index=False)

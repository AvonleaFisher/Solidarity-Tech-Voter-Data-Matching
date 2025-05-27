import marimo

__generated_with = "0.13.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import pandas as pd
    from nicknames import NickNamer
    from fuzzywuzzy import fuzz
    import numpy as np
    import re
    from datetime import datetime
    import unicodedata
    return NickNamer, datetime, fuzz, pd, re, unicodedata


@app.cell
def _():
    # set up file locations
    voterfile_parquet = "AllNYSVoters_20250521/AllNYSVoters.parquet"
    nyccontribute_file = "data/2025_MAMDANI_NYC_Votes_Contribute_Credit_Card_Contributions_05_12_2025.csv"
    ab_file = "data/zohran-for-nyc-164341-contributions-all.csv"
    return ab_file, nyccontribute_file, voterfile_parquet


@app.cell
def _(pd, voterfile_parquet):
    #load in voter file and filter to NYC voters
    voterfile = pd.read_parquet(voterfile_parquet, engine="pyarrow")
    voterfile = voterfile[voterfile['countycode'].isin([24,31,41,43,3])]
    voterfile = voterfile.rename(columns={
        "raddnumber": "voter_house_number",
        "rstreetname": "voter_street_name",
        "rapartment": "voter_apt_num",
        "voter_zip": "voter_zip",
        "sboeid": "voter_boe_id",
        "status": "registration_status",
        "enrollment": "party_enrollment",
        "mailadd1": "voter_mailing_address1",
        "mailadd2": "voter_mailing_address2",
        "mailadd3": "voter_mailing_address3",
        "mailadd4": "voter_mailing_address4",
        "dob": "voter_dob",
        "prevname": "voter_previous_name",
        "prevaddress": "voter_previous_address"
    })
    return (voterfile,)


@app.cell
def _(ab_file, nyccontribute_file, pd):
    #load in NYCContribute file and filter to NY donors
    nyccontribute_donors = pd.read_csv(nyccontribute_file).apply(lambda x: x.str.upper() if x.dtype == 'object' else x)
    nyccontribute_donors = nyccontribute_donors[nyccontribute_donors["Residential Street Address"].str.rsplit(',').str[-2].str[1:3]  == 'NY']

    #load in Act Blue donor file and filter to NY donors
    ab_donors = pd.read_csv(ab_file).apply(lambda x: x.str.upper() if x.dtype == 'object' else x)
    ab_donors = ab_donors[ab_donors["Donor State"]=='NY']
    return ab_donors, nyccontribute_donors


@app.cell
def _(pd, unicodedata):
    def normalize_name(name):
        if pd.isna(name):
            return name
        # Remove apostrophes
        name = name.replace("'", "")
        # Remove accents
        name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode()
        # Uppercase
        name = name.upper()
        return name
    return (normalize_name,)


@app.cell
def _(pd, re):
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
    return (clean_apt_num,)


@app.cell
def _(pd, re):
    # Define common suffixes to remove when isolated
    suffixes = r'\s+(STREET|ST|AVENUE|AVE|BLVD|BOULEVARD|DRIVE|DR|ROAD|RD|PARKWAY|PKWY|LANE|LN|COURT|CT|TERRACE|TER|PLACE|PL|CIRCLE|CIR|HIGHWAY|HWY|WAY|SQ|SQUARE|EXPY|EXPRESSWAY|STATION|STN|ALLEY|ALY|PLAZA|PLZ|LOOP)\.?(?=\s|$)'

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

        # Remove apartment/unit info at the end
        street_name = re.sub(r'\b(APT|UNIT|FL|#)\s*\w+[.]?$', '', street_name, flags=re.IGNORECASE).strip()



        # Standardize directional prefixes (e.g., EAST 117 -> E 117)
        for full, abbrev in directions.items():
            street_name = re.sub(full, abbrev, street_name, flags=re.IGNORECASE)

        # Handle numbered streets (e.g., 117th -> 117)
        street_name = re.sub(r'(\d+)(TH|ST|ND|RD)\b', r'\1', street_name, flags=re.IGNORECASE)

        # Remove common street suffixes only when they are isolated
        street_name = re.sub(suffixes, '', street_name, flags=re.IGNORECASE).strip()

        #remove any remaining periods
        street_name = re.sub(r'\.', '', street_name)

        # Replace SAINT with ST
        street_name = re.sub(r'\bSAINT \b', 'ST ', street_name, flags=re.IGNORECASE)

        return street_name

    return (normalize_street_name,)


@app.cell
def _(
    ab_donors,
    clean_apt_num,
    normalize_name,
    normalize_street_name,
    nyccontribute_donors,
    voterfile,
):
    # Ensure ZIP codes are strings and take first 5 characters
    voterfile["voter_zip5"] = voterfile["rzip5"].astype(str).str[:5]
    ab_donors["donor_zip5"] = ab_donors["Donor ZIP"].astype(str).str[:5]
    nyccontribute_donors["donor_zip5"] = nyccontribute_donors["Residential Street Address"].str.rsplit(',').str[-1].str[1:6]

    # Convert names to uppercase for case-insensitive matching
    voterfile["voter_last_name"] = voterfile["lastname"].apply(normalize_name)
    voterfile["voter_first_name"] = voterfile["firstname"].apply(normalize_name)
    ab_donors["donor_last_name"] = ab_donors["Donor Last Name"].apply(normalize_name)
    ab_donors["donor_first_name"] = ab_donors["Donor First Name"].apply(normalize_name)
    nyccontribute_donors["donor_last_name"] = nyccontribute_donors["Last Name"].apply(normalize_name)
    nyccontribute_donors["donor_first_name"] = nyccontribute_donors["First Name"].apply(normalize_name)

    voterfile['voter_apt_num_clean'] = voterfile['voter_apt_num'].apply(clean_apt_num)
    voterfile['voter_street_name_clean'] = voterfile['voter_street_name'].apply(normalize_street_name)
    return


@app.cell
def _(ab_donors, clean_apt_num, normalize_street_name, pd):
    #ab_donor processing
    ab_donors["donor_id"] = ab_donors["Donor Email"]
    ab_donors["donor_phone_number"] = ab_donors["Donor Phone"].str.replace(r'\D', '', regex=True)
    ab_donors["donor_email"] = ab_donors["Donor Email"]

    apt_keywords = r'(?:APT\.?|UNIT|FL|#|APARTMENT)?\s*[A-Z]?\d+[A-Z0-9]*'
    split_regex = rf'^(.*?)(?:\s+({apt_keywords}))?$'
    ab_donors[["donor_address1","donor_apt_num"]] = ab_donors['Donor Addr1'].str.extract(split_regex)
    ab_donors['donor_address1'] = ab_donors['donor_address1'].str.strip()
    ab_donors['donor_apt_num'] = ab_donors['donor_apt_num'].str.strip()
    ab_donors['donor_apt_num_clean'] = ab_donors["donor_apt_num"].apply(clean_apt_num)
    ab_donors[['donor_house_num','donor_street_name']] = ab_donors['donor_address1'].str.extract(r'^([\d\-]+)\s+(.*)', expand=True)

    ab_donors['donor_street_name_clean'] = ab_donors['donor_street_name'].apply(normalize_street_name)

    ab_donors['date'] = pd.to_datetime(ab_donors['Date']).dt.date
    return


@app.cell
def _(clean_apt_num, normalize_street_name, nyccontribute_donors, pd):
    #nyccontribute_donor processing
    nyccontribute_donors["donor_id"] = nyccontribute_donors["Email"]

    nyccontribute_donors["donor_address1"] = nyccontribute_donors["Residential Street Address"].str.rsplit(',').str[0]
    nyccontribute_donors["donor_phone_number"] = nyccontribute_donors["Phone"].str.replace(r'\D', '', regex=True)
    nyccontribute_donors["donor_email"] = nyccontribute_donors["Email"]

    pattern = r'^[^,]+,\s*([^,]+),'
    apt_pattern = r'APT|UNIT|FL|[0-9]|#|BSMNT'
    nyccontribute_donors["donor_apt_num"] = nyccontribute_donors["Residential Street Address"].str.extract(pattern)
    nyccontribute_donors["donor_apt_num"] = nyccontribute_donors["donor_apt_num"].where(
        nyccontribute_donors["donor_apt_num"].str.upper().str.contains(apt_pattern, na=False)
    )

    nyccontribute_donors["donor_apt_num_clean"] = nyccontribute_donors["donor_apt_num"].apply(clean_apt_num)

    nyccontribute_donors[['donor_house_num','donor_street_name']]  = nyccontribute_donors['donor_address1'].str.extract(r'^([\d\-]+)\s+(.*)', expand=True)

    nyccontribute_donors['donor_street_name_clean'] = nyccontribute_donors['donor_street_name'].apply(normalize_street_name)

    nyccontribute_donors['date'] = pd.to_datetime(nyccontribute_donors['Date']).dt.date
    return


@app.cell
def _(nyccontribute_donors):
    # QA cell - replace donor_id with email address
    nyccontribute_donors[nyccontribute_donors['donor_id'] == 'donor_id'][['donor_id','donor_first_name','donor_last_name','donor_apt_num_clean','donor_house_num','donor_street_name','donor_street_name_clean','donor_zip5','donor_phone_number','donor_email']]

    return


@app.cell
def _(ab_donors, nyccontribute_donors, pd):
    ab_donor_info = ab_donors[['donor_id','date','donor_first_name','donor_last_name','donor_apt_num_clean','donor_house_num','donor_street_name_clean','donor_zip5','donor_phone_number','donor_email']]

    nyccontribute_donor_info = nyccontribute_donors[['donor_id','date','donor_first_name','donor_last_name','donor_apt_num_clean','donor_house_num','donor_street_name_clean','donor_zip5','donor_phone_number','donor_email']]

    donors = pd.concat([ab_donor_info,nyccontribute_donor_info])

    # Record most recent date per donor
    latest_dates = donors.groupby('donor_id')['date'].max().reset_index().rename(columns={'date': 'latest_date'})

    donors = donors.drop('date', axis=1)

    # dedupe on donor_id, keep record with most complete data
    donors['non_nulls'] = donors.notna().sum(axis=1)
    donors = donors.loc[donors.groupby('donor_id')['non_nulls'].idxmax()].drop(columns='non_nulls')

    # add in latest donation date (for van "canvass date")
    donors = donors.merge(latest_dates, on='donor_id', how='left')
    return (donors,)


@app.cell
def _(donors, pd, voterfile):
    # Match on last name + ZIP
    match_ln = donors.merge(
        voterfile,
        left_on=["donor_last_name", "donor_zip5"],
        right_on=["voter_last_name", "voter_zip5"],
        how="inner"
    )

    # Match on first name + ZIP
    match_fn = donors.merge(
        voterfile,
        left_on=["donor_first_name", "donor_zip5"],
        right_on=["voter_first_name", "voter_zip5"],
        how="inner"
    )

    # Combine match results
    df = pd.concat([match_ln, match_fn], ignore_index=True)
    df = df.drop_duplicates()
    return (df,)


@app.cell
def _(NickNamer, df, fuzz, pd):
    # 1. Last name exact match
    df['last_name_match'] = (df['donor_last_name'].fillna('') == df['voter_last_name'].fillna('')).astype(int)

    # 2. Fuzzy last name match
    df['fuzzy_last_name_match'] = df.apply(
        lambda row: fuzz.partial_ratio(str(row['donor_last_name'] or ''), str(row['voter_last_name'] or '')) >= 80,
        axis=1
    ).astype(int)


    # 3. First name exact match
    df['first_name_match'] = (df['donor_first_name'].fillna('') == df['voter_first_name'].fillna('')).astype(int)

    # 4. Nickname match
    nn = NickNamer()
    def is_nickname_match(name1, name2):
        if pd.isna(name1) or pd.isna(name2):  # Handle NaN values
            return False
        name1, name2 = str(name1).strip(), str(name2).strip()
        return name1 == name2 or name2 in nn.nicknames_of(name1) or name1 in nn.nicknames_of(name2)

    df['nickname_match'] = df.apply(lambda row: is_nickname_match(row['donor_first_name'], row['voter_first_name']), axis=1).astype(int)

    # 5. Street Name Match
    df['street_name_match'] = (df['donor_street_name_clean'].fillna('') == df['voter_street_name_clean'].fillna('')).astype(int)

    # 6. House Number Match
    df['house_num_match'] = (df['donor_house_num'].fillna('') == df['voter_house_number'].fillna('')).astype(int)

    # 7. Apartment Number Match
    df['apt_num_match'] = (df['donor_apt_num_clean'].fillna('') == df['voter_apt_num_clean'].fillna('')).astype(int)

    # 8. Compute match strength
    match_columns = ['nickname_match', 'last_name_match', 'fuzzy_last_name_match',
                     'first_name_match', 'street_name_match', 'house_num_match', 'apt_num_match']
    df['match_strength'] = df[match_columns].sum(axis=1)


    # 9. If match_strength <=2 1, remove voter columns
    columns_to_null = ['voter_first_name', 'voter_last_name', 'voter_house_number', 'voter_street_name_clean',
                       'voter_apt_num', 'voter_zip', 'voter_boe_id', 'registration_status',
                       'party_enrollment', 'voter_mailing_address1', 'voter_mailing_address2',
                       'voter_mailing_address3', 'voter_mailing_address4', 'voter_dob', 'voter_previous_name',
                       'voter_previous_address']

    df.loc[df['match_strength'] <= 1, columns_to_null] = None


    # df_no_match = df[df['match_strength'] == 0].drop_duplicates(subset='donor_id', keep='first')
    # df_matched = df[df['match_strength'] > 2]

    df['is_highest_match_strength'] = df.groupby('donor_id')['match_strength'].transform(lambda x: x == x.max()).fillna(0).astype(int)
    return


@app.cell
def _():
    new_column_order = [
        'donor_id', 'donor_phone_number', 'latest_date',
        'donor_first_name', 'voter_first_name',
        'donor_last_name', 'voter_last_name',
        # Address Columns
        'donor_house_num', 'voter_house_number',  # Reordered to keep house number together
        'donor_street_name_clean', 'voter_street_name_clean',
        'donor_apt_num_clean', 'voter_apt_num_clean', 
        'donor_zip5', 'voter_zip5',  # Zip codes grouped together
        # Email & Registration
        'donor_email',
        'voter_boe_id', 'registration_status', 'party_enrollment',
        'voter_mailing_address1', 'voter_mailing_address2',
        'voter_mailing_address3', 'voter_mailing_address4', 
        'voter_dob', 'voter_previous_name', 'voter_previous_address',
        # Matching Criteria
        'last_name_match', 'fuzzy_last_name_match', 'first_name_match',
        'nickname_match', 'street_name_match', 'house_num_match', 'apt_num_match',
        'match_strength', 'is_highest_match_strength'
    ]
    return (new_column_order,)


@app.cell
def _(df, new_column_order):
    # Reorder the DataFrame columns
    df_cleaned = df[new_column_order][df['match_strength'] > 2]
    df_cleaned=df_cleaned.drop_duplicates(keep='first')

    # Ensure consistent data types and handle NaN values
    for col in ['donor_first_name', 'voter_first_name', 'donor_last_name', 
                'voter_last_name', 'donor_zip5', 'voter_zip5']:
        df_cleaned[col] = df_cleaned[col].astype(str).fillna('')

    # Filter to keep only rows where 'is_highest_match_strength' is 1
    df_cleaned = df_cleaned[df_cleaned['is_highest_match_strength'] == 1]

    # Create a temporary column to count non-null values in each row
    df_cleaned['_non_null_count'] = df_cleaned.notnull().sum(axis=1)

    # Sort by 'match_strength' (descending) and '_non_null_count' (descending) to prioritize most complete rows
    df_cleaned = df_cleaned.sort_values(by=['match_strength', '_non_null_count'], ascending=[False, False])

    #label duplicates
    df_cleaned['is_duplicate'] = df_cleaned.duplicated(subset=['donor_id'], keep='first').astype(int)
    return (df_cleaned,)


@app.cell
def _(df_cleaned, pd):
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
    df_cleaned['donor_zip5'] = (
        df_cleaned['donor_zip5']
        .astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
        .str.split("-").str[0]
        .str.zfill(5)
    )


    df_cleaned['exact_full_match'] = (
        (df_cleaned['donor_first_name'] == df_cleaned['voter_first_name']) &
        (df_cleaned['donor_last_name'] == df_cleaned['voter_last_name']) &
        (df_cleaned['donor_street_name_clean'] == df_cleaned['voter_street_name_clean'])
    )

    #Find ambiguous IDs (same ID with multiple match_strength >= 5)
    match_counts = df_cleaned[df_cleaned['match_strength'] >= 5].groupby('donor_id').size()
    ambiguous_ids = match_counts[match_counts > 1].index

    # Assign match_types
    def assign_match_type(row):
        zip5 = row.get("donor_zip5", "")
        if pd.notna(zip5) and zip5 not in nyc_zip_codes:
            return "Ineligible: Address outside of NYC"
        if row['donor_id'] in ambiguous_ids:
            return "Cannot be determined"
        if row['exact_full_match']:
            return "Perfect match"
        if row['match_strength'] >= 5:
            return "Strong match"
        if 2 <= row['match_strength'] <= 4:
            return "Weak match"
        return "No match"

    df_cleaned['match_type'] = df_cleaned.apply(assign_match_type, axis=1)

    df_cleaned['match_type'] = df_cleaned.apply(
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
    return


@app.cell
def _(datetime, df_cleaned):
    party_mapping = {
        "DEM": "Democrat",
        "BLK": "Unaffiliated",
        "WOR": "WFP",
        "REP": "Republican",
        "OTH": "Not registered"
    }
    df_cleaned['phone_number'] = df_cleaned['donor_phone_number']
    df_cleaned["Registration Status"] = df_cleaned["party_enrollment"].map(party_mapping).fillna("Cannot Be Determined")
    df_cleaned.loc[df_cleaned['match_type'] == 'Ineligible: Address outside of NYC', "Registration Status"] = 'Not eligible'
    df_cleaned.loc[df_cleaned['match_type'] == 'No match', "Registration Status"] = 'Not Registered'
    df_cleaned['VAN_BOE_ID'] = df_cleaned['voter_boe_id'].str[-10:]

    today_str = datetime.today().strftime('%Y-%m-%d')

    df_van = df_cleaned[['VAN_BOE_ID','donor_first_name','donor_last_name','donor_house_num','donor_street_name_clean','latest_date']][df_cleaned['match_type'] == 'Perfect match']
    return df_van, today_str


@app.cell
def _(df_cleaned):
    # QA cell - replace donor_id with donor email
    df_cleaned[df_cleaned['donor_id'] == 'donor_id']
    return


@app.cell
def _(df_van, today_str):
    df_van.to_csv(f"data/VAN_donor_upload_{today_str}.csv", index=False)
    return


@app.cell
def _(df_cleaned):
    df_cleaned.to_csv("data/final_combined_matches.csv", index=False)
    return


if __name__ == "__main__":
    app.run()

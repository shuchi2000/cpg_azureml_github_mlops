
#pip install azure-identity azure-storage-file-datalake pandas

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from azure.identity import ClientSecretCredential
from azure.storage.filedatalake import DataLakeServiceClient
import pandas as pd
from io import StringIO


# --- App Registration credentials from environment variables ---
import os
tenant_id = os.environ.get("AZURE_TENANT_ID")
client_id = os.environ.get("AZURE_CLIENT_ID")
client_secret = os.environ.get("AZURE_CLIENT_SECRET")

# Azure Storage Parameters
storage_account_name = "optimalchanneldata"
container_name = "optimalchannel"

# Authenticate with Azure using App Registration credentials
credential = ClientSecretCredential(
    tenant_id=tenant_id,
    client_id=client_id,
    client_secret=client_secret
)

# Initialize the Data Lake Service Client using App Registration credentials
service_client = DataLakeServiceClient(
    account_url=f"https://{storage_account_name}.dfs.core.windows.net",
    credential=credential
)


# Function to read a file from Azure Data Lake and return it as a DataFrame
def read_file_from_container(service_client, container_name, file_name):
    try:
        # Get filesystem client
        file_system_client = service_client.get_file_system_client(file_system=container_name)

        # Get file client
        file_client = file_system_client.get_file_client(file_name)

        # Read file content
        download = file_client.download_file()
        file_content = download.readall()

        # Try decoding the file with UTF-8 encoding first
        try:
            file_content_decoded = file_content.decode("utf-8")
        except UnicodeDecodeError:
            file_content_decoded = file_content.decode("ISO-8859-1")
        
        # Convert the decoded content into a Pandas DataFrame
        df = pd.read_csv(StringIO(file_content_decoded))
        return df
    
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

# Example function call for reading a dataset
def load_datasets(service_client, container_name, datasets):
    """
    This function loads multiple datasets and returns them as a dictionary of DataFrames.
    :param service_client: The Data Lake service client.
    :param datasets: A list of tuples containing container names and file names.
    :return: A dictionary of datasets.
    """
    data_frames = {}
    for file_name in datasets:
        df = read_file_from_container(service_client, container_name, file_name)
        if df is not None:
            data_frames[file_name] = df
    return data_frames

# Define the datasets you want to load
datasets_to_load = [
    "city.csv",
    "retailsalesdistribution.csv",
    "products.csv",
    "consumerbehavior.csv",
    "competitivelandscape.csv",
    "externalmarketinfluencer.csv"
]


# Load datasets
data = load_datasets(service_client, container_name, datasets_to_load)






def process_weekly_retail_sales(retail):
    """
    Processes the retail sales data to generate a weekly aggregation of total sales and units sold.
    The function performs the following steps:
    1. Converts the 'Date' column to a datetime format.
    2. Creates a new column 'Week_Start_Date' to represent the start of the week for each 'Date'.
    3. Aggregates sales and units sold by week, city, and channel.
    4. Creates a pivot table with total sales and units sold by week, city, and channel.
    5. Flattens the multi-level column headers and resets the index.
    
    Args:
        retail (pd.DataFrame): The retail sales DataFrame containing 'Date', 'Sales', 'Units_Sold', 
                               'City_ID', 'SKU_ID', and 'Channel' columns.
    
    Returns:
        pd.DataFrame: A processed DataFrame with weekly aggregated sales data by city and channel.
    """
    # Remove rows where the 'Date' column is equal to a specific date (e.g., '2023-01-01')
    # This is useful as we don't have this particular date information further in other weekly based dataset
    # We will not face missing values problem while merging.
    retail = retail[retail['Date'] != '2023-01-01']

    # Convert 'Date' column to datetime format for easier manipulation
    retail["Date"] = pd.to_datetime(retail["Date"])

    # Create a new column 'Week_Start_Date' that aligns the 'Date' to the start of the week (Monday)
    retail["Week_Start_Date"] = retail["Date"] - pd.to_timedelta(retail["Date"].dt.weekday, unit='D')

    # Aggregate the sales data at the weekly level per 'City_ID', 'SKU_ID', and 'Channel'
    weekly_retail_sales = retail.groupby(["Week_Start_Date", "City_ID", "Channel"]).agg(
        Total_Sales=('Sales', 'sum'),
        Total_Units_Sold=('Units_Sold', 'sum')
    ).reset_index()

    # Create a pivot table to get the Total Sales and Total Units Sold by Week, City, and Channel
    weekly_retail_sales = weekly_retail_sales.pivot_table(
        index=['Week_Start_Date', 'City_ID'],
        columns='Channel',
        values=['Total_Sales', 'Total_Units_Sold'],
        aggfunc='sum',
        fill_value=0
    )

    # Flatten the multi-level columns to a single level by concatenating the 'Channel', 'Total_Sales', and 'Total_Units_Sold' names
    weekly_retail_sales.columns = [f'{col[1]}_{col[0]}' for col in weekly_retail_sales.columns]

    # Reset the index to make 'Week_Start_Date' and 'City_ID' columns again
    weekly_retail_sales.reset_index(inplace=True)

    return weekly_retail_sales

# Example usage:
# processed_sales = process_weekly_retail_sales(retail)
# processed_sales.head()


def process_and_merge_city_data(city, weekly_retail_sales):
    """
    Processes the city data to categorize cities into income brackets and merge it with weekly retail sales data.
    
    The function performs the following steps:
    1. Selects specific columns from the 'city' DataFrame.
    2. Converts 'Per_Capita_Income (INR)' column to numeric and handles errors.
    3. Calculates percentiles for 'Per_Capita_Income (INR)' and creates income brackets.
    4. One-hot encodes the income brackets.
    5. Drops unnecessary columns from the 'city' DataFrame.
    6. Merges the processed 'city' DataFrame with 'weekly_retail_sales' on 'City_ID'.
    
    Args:
        city (pd.DataFrame): The DataFrame containing city-level data.
        weekly_retail_sales (pd.DataFrame): The DataFrame containing weekly retail sales data.
    
    Returns:
        pd.DataFrame: A DataFrame containing the merged data with additional city-level features.
    """
    # Step 1: Select relevant columns from the 'city' DataFrame
    city = city[['City_ID', 'City_Name', 'City_tier', 'Population_Density(persons/km²)', 'Per_Capita_Income (INR)']]

    # Step 2: Convert 'Per_Capita_Income (INR)' column to numeric, coercing errors to NaN
    city['Per_Capita_Income (INR)'] = pd.to_numeric(city['Per_Capita_Income (INR)'], errors='coerce')

    # Step 3: Calculate percentiles for 'Per_Capita_Income (INR)'
    percentiles = city['Per_Capita_Income (INR)'].quantile([0, 0.25, 0.50, 0.75, 1]).to_list()

    # Step 4: Define income bracket labels based on percentiles
    labels = ['Low', 'Lower Middle', 'Upper Middle', 'High']

    # Step 5: Create a new 'Income Bracket Indicator' column using the percentiles and labels
    city['Income Bracket Indicator'] = pd.cut(city['Per_Capita_Income (INR)'], bins=percentiles, labels=labels, include_lowest=True)

    # Step 6: Apply one-hot encoding to the 'Income Bracket Indicator' column
    city = pd.get_dummies(city, columns=['Income Bracket Indicator'], prefix='Income_Bracket')

    # Step 7: Drop unnecessary columns from the 'city' DataFrame
    city.drop(['Population_Density(persons/km²)', 'Per_Capita_Income (INR)'], axis=1, inplace=True)

    # Step 8: Merge the processed 'city' DataFrame with the 'weekly_retail_sales' DataFrame on 'City_ID'
    retail_city = weekly_retail_sales.merge(city, on="City_ID", how="left")

    # Return the merged DataFrame
    return retail_city

# Example usage:
# processed_data = process_and_merge_city_data(city, weekly_retail_sales)
# processed_data.head()


def process_and_merge_consumer_data(consumer):
    """
    Processes the consumer data to calculate various metrics, pivot tables, and merge them into a comprehensive DataFrame.
    
    The function calculates:
    1. Unique customers per city.
    2. Pivot table of customers across preferred channels.
    3. Percentage share of customers per channel.
    4. Total purchase frequency per city.
    5. Pivot table of purchase frequency across channels.
    6. Price sensitivity: percentage of high sensitivity customers.
    7. Median income per city.
    8. Age group distribution percentage in each city.
    9. Top preferred flavor in each city.
    10. Flavor preferences percentage per city.
    
    Args:
        consumer (pd.DataFrame): The consumer data containing 'City_ID', 'Customer_ID', 'Preferred_Channel', 
                                 'Purchase_Frequency', 'Price_Sensitivity', 'Income_Level', 'Age_Group', and 'Preferred_Flavor' columns.
    
    Returns:
        pd.DataFrame: A DataFrame containing all the calculated metrics merged on 'City_ID'.
    """
    
    # 1. Unique Customers per City
    unique_customers = consumer.groupby('City_ID')['Customer_ID'].nunique().reset_index()
    unique_customers.rename(columns={'Customer_ID': 'Unique_Customers'}, inplace=True)

    # 2. Customers Pivoted Across Channel
    channel_pivot = consumer.pivot_table(
        index='City_ID',
        columns='Preferred_Channel',
        values='Customer_ID',
        aggfunc='count',
        fill_value=0
    ).reset_index()
    channel_pivot.columns = ['City_ID'] + [f'Customers_Channel_{col}' for col in channel_pivot.columns[1:]]

    # 3. Percentage Share of Customers per Channel
    channel_pivot = channel_pivot.merge(unique_customers, on="City_ID", how="inner")
    for col in channel_pivot.columns[1:]:
        channel_pivot[f'{col}_Share'] = channel_pivot[col] / channel_pivot['Unique_Customers']

    # 4. Total Purchase Frequency per City
    total_purchase_freq = consumer.groupby('City_ID')['Purchase_Frequency'].sum().reset_index()
    total_purchase_freq.rename(columns={'Purchase_Frequency': 'Total_Purchase_Frequency'}, inplace=True)

    # 6. Purchase Frequency Pivot Across Channels
    purchase_freq_pivot = consumer.pivot_table(
        index='City_ID',
        columns='Preferred_Channel',
        values='Purchase_Frequency',
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    purchase_freq_pivot.columns = ['City_ID'] + [f'Purchase_Frequency_{col}' for col in purchase_freq_pivot.columns[1:]]

    # 7. Price Sensitivity: % of High Sensitivity Customers
    price_sensitivity_counts = consumer.groupby(['City_ID', 'Price_Sensitivity'])['Customer_ID'].count().unstack(fill_value=0)
    price_sensitivity_counts = price_sensitivity_counts.merge(unique_customers, on="City_ID", how="inner")
    if 'High' in price_sensitivity_counts.columns:
        price_sensitivity_counts['%_High_Price_Sensitive_Customers'] = price_sensitivity_counts['High'] / price_sensitivity_counts['Unique_Customers']
    else:
        price_sensitivity_counts['%_High_Price_Sensitive_Customers'] = 0

    # 8. Median Income Per City
    median_income = consumer.groupby('City_ID')['Income_Level'].median().reset_index()
    median_income.rename(columns={'Income_Level': 'Median_Income'}, inplace=True)

    # 9. Age Group Distribution: % of Customers in Each Age Group
    age_group_pivot = consumer.pivot_table(
        index='City_ID',
        columns='Age_Group',
        values='Customer_ID',
        aggfunc='count',
        fill_value=0
    ).reset_index()
    for col in age_group_pivot.columns[1:]:
        age_group_pivot[col] = age_group_pivot[col] / unique_customers['Unique_Customers']
    age_group_pivot.columns = ['City_ID'] + [f'Age_Group_{col}_Share' for col in age_group_pivot.columns[1:]]

    # 10. Top Preferred Flavor in Each City
    top_flavor = consumer.groupby(['City_ID', 'Preferred_Flavor'])['Customer_ID'].count().reset_index()
    top_flavor = top_flavor.loc[top_flavor.groupby('City_ID')['Customer_ID'].idxmax(), ['City_ID', 'Preferred_Flavor']]
    top_flavor.rename(columns={'Preferred_Flavor': 'Top_Preferred_Flavor'}, inplace=True)

    # 11. Flavor Preferences Percentage in City
    flavor_pivot = consumer.pivot_table(
        index='City_ID',
        columns='Preferred_Flavor',
        values='Customer_ID',
        aggfunc='count',
        fill_value=0
    ).reset_index()
    flavor_pivot = flavor_pivot.merge(unique_customers, on="City_ID", how="inner")
    for col in flavor_pivot.columns[1:]:
        flavor_pivot[col] = flavor_pivot[col] / flavor_pivot['Unique_Customers']
    flavor_pivot.columns = ['City_ID'] + [f'Flavor_{col}_Share' for col in flavor_pivot.columns[1:]]
    
    # Drop unnecessary column from flavor_pivot (if exists)
    if 'Flavor_Unique_Customers_Share' in flavor_pivot.columns:
        flavor_pivot.drop(['Flavor_Unique_Customers_Share'], axis=1, inplace=True)

    # Merge all the metrics into one DataFrame
    consumer_df = unique_customers.merge(channel_pivot, on="City_ID", how="inner") \
        .merge(purchase_freq_pivot, on="City_ID", how="inner") \
        .merge(total_purchase_freq[['City_ID', 'Total_Purchase_Frequency']], on="City_ID", how="inner") \
        .merge(price_sensitivity_counts[["City_ID", "%_High_Price_Sensitive_Customers"]], on="City_ID", how="inner") \
        .merge(median_income, on="City_ID", how="inner") \
        .merge(age_group_pivot, on="City_ID", how="inner") \
        .merge(flavor_pivot, on="City_ID", how="inner")

    return consumer_df

# Example usage:
# processed_consumer_data = process_and_merge_consumer_data(consumer)
# processed_consumer_data.head()




def process_and_merge_channel_scores(retail_city, consumer_df):
    """
    Processes the consumer data to calculate channel scores and merge it with the retail data.
    
    The function performs the following:
    1. Normalizes selected columns using MinMaxScaler.
    2. Calculates channel scores by combining the customer share and purchase frequency for each channel.
    3. Drops unnecessary columns.
    4. Merges the processed consumer data with the retail data.
    
    Args:
        retail_city (pd.DataFrame): The retail sales data with city information.
        consumer_df (pd.DataFrame): The consumer data containing customer and channel information.
    
    Returns:
        pd.DataFrame: The merged DataFrame with channel scores and retail sales data.
    """
    # Define the columns to normalize
    columns_to_normalize = [
        'Customers_Channel_E-commerce_Share', 'Purchase_Frequency_E-commerce',
        'Customers_Channel_General Trade_Share', 'Purchase_Frequency_General Trade',
        'Customers_Channel_HoReCa_Share', 'Purchase_Frequency_HoReCa',
        'Customers_Channel_Modern Trade_Share', 'Purchase_Frequency_Modern Trade',
        'Customers_Channel_Q-commerce_Share', 'Purchase_Frequency_Q-commerce'
    ]

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Normalize the selected columns
    consumer_df[columns_to_normalize] = scaler.fit_transform(consumer_df[columns_to_normalize])

    # Define the channel score columns by calculating the weighted average of customer share and purchase frequency for each channel.
    consumer_df['Channel_Score_E-commerce'] = (0.3 * consumer_df['Customers_Channel_E-commerce_Share'] + 0.7 * consumer_df['Purchase_Frequency_E-commerce']) 
    consumer_df['Channel_Score_General_Trade'] = (0.3 * consumer_df['Customers_Channel_General Trade_Share'] + 0.7 * consumer_df['Purchase_Frequency_General Trade']) 
    consumer_df['Channel_Score_HoReCa'] = (0.3 * consumer_df['Customers_Channel_HoReCa_Share'] + 0.7 * consumer_df['Purchase_Frequency_HoReCa']) 
    consumer_df['Channel_Score_Modern_Trade'] = (0.3 * consumer_df['Customers_Channel_Modern Trade_Share'] + 0.7 * consumer_df['Purchase_Frequency_Modern Trade']) 
    consumer_df['Channel_Score_Q-commerce'] = (0.3 * consumer_df['Customers_Channel_Q-commerce_Share'] + 0.7 * consumer_df['Purchase_Frequency_Q-commerce']) 

    # Drop unnecessary columns
    consumer_df.drop([
        'Customers_Channel_E-commerce_Share',
        'Customers_Channel_General Trade_Share',
        'Customers_Channel_HoReCa_Share',
        'Customers_Channel_Modern Trade_Share',
        'Customers_Channel_Q-commerce_Share'
    ], axis=1, inplace=True)

    # Merge the processed consumer data with the retail data
    retail_city_consumer = retail_city.merge(consumer_df, on="City_ID", how="left")

    return retail_city_consumer

# Example usage:
# retail_city_consumer = process_and_merge_channel_scores(retail_city, consumer_df)
# retail_city_consumer.head()




def process_and_merge_competitive_data(retail_city_consumer, competitive):
    """
    Processes the competitive data to calculate various metrics, performs normalization, 
    and merges it with the retail and consumer data.
    
    The function performs the following:
    1. Renames and converts date columns to datetime format.
    2. Normalizes selected columns for competitive data.
    3. Creates a 'Competitor_Intensity' metric based on Mentions and Share of Voice.
    4. Calculates 'Sentiment_Impact' based on Sentiment Score and Competitor Intensity.
    5. Creates a pivot table for sentiment impact by Week, Channel, and Brand.
    6. Merges the processed competitive data with the retail and consumer data.
    
    Args:
        retail_city_consumer (pd.DataFrame): The retail and consumer data.
        competitive (pd.DataFrame): The competitive data containing 'Mentions_Count', 'Share_of_Voice', and 'Sentiment_Score'.
    
    Returns:
        pd.DataFrame: The merged DataFrame containing both retail and competitive data.
    """
    
    # Rename the 'Date' column in competitive to 'Week_Start_Date'
    competitive.rename(columns={'Date': "Week_Start_Date"}, inplace=True)
    
    # Convert 'Week_Start_Date' to datetime format
    retail_city_consumer["Week_Start_Date"] = pd.to_datetime(retail_city_consumer["Week_Start_Date"])
    competitive["Week_Start_Date"] = pd.to_datetime(competitive["Week_Start_Date"])

    # Initialize MinMaxScaler
    scaler = MinMaxScaler()

    # Normalize the selected columns in competitive data
    columns_to_normalize = ["Mentions_Count", "Share_of_Voice", "Sentiment_Score"]
    competitive[columns_to_normalize] = scaler.fit_transform(competitive[columns_to_normalize])

    # **Competitive Influence**
    # a. Competitor Intensity: Combine 'Mentions_Count' and 'Share_of_Voice' to create a new metric
    competitive["Competitor_Intensity"] = competitive["Mentions_Count"] + competitive["Share_of_Voice"]

    # b. Sentiment Impact: Calculate impact of sentiment using 'Sentiment_Score' and 'Competitor_Intensity'
    competitive["Sentiment_Impact"] = competitive["Sentiment_Score"] * competitive["Competitor_Intensity"]

    # Drop the original columns as they are now incorporated into 'Competitor_Intensity'
    competitive.drop(["Mentions_Count", "Share_of_Voice", "Sentiment_Score"], axis=1, inplace=True)

    # Create a pivot table to get sentiment impact by Week, Channel, and Brand
    sentiment_pivot = competitive.pivot_table(
        index='Week_Start_Date',
        columns=['Channel', 'Brand'],
        values='Sentiment_Impact',
        aggfunc='sum',  # Aggregate using sum
        fill_value=0  # Fill missing values with 0
    )

    # Flatten the multi-level columns for better readability
    sentiment_pivot.columns = [f'{channel}_{brand}_sentiment_impact' for channel, brand in sentiment_pivot.columns]

    # Reset index to make 'Week_Start_Date' a column
    sentiment_pivot.reset_index(inplace=True)

    # Merge the competitive data (sentiment pivot) with the retail city consumer data
    retail_city_consumer["Week_Start_Date"] = pd.to_datetime(retail_city_consumer["Week_Start_Date"])
    sentiment_pivot["Week_Start_Date"] = pd.to_datetime(sentiment_pivot["Week_Start_Date"])

    # Merge the two DataFrames on 'Week_Start_Date'
    retail_city_consumer_competitive = retail_city_consumer.merge(sentiment_pivot, on='Week_Start_Date', how="left")

    return retail_city_consumer_competitive

# Example usage:
# retail_city_consumer_competitive = process_and_merge_competitive_data(retail_city_consumer, competitive)
# retail_city_consumer_competitive.head()




def merge_with_external_data(retail_city_consumer_competitive, external):
    """
    Merges the retail and consumer data with external data based on 'Week_Start_Date' and 'City_ID'.
    
    The function performs the following steps:
    1. Converts 'Week_Start_Date' columns in both DataFrames to datetime format.
    2. Merges the two DataFrames on 'Week_Start_Date' and 'City_ID'.
    
    Args:
        retail_city_consumer_competitive (pd.DataFrame): The DataFrame containing retail and competitive data.
        external (pd.DataFrame): The external data to be merged.
    
    Returns:
        pd.DataFrame: The merged DataFrame containing data from both `retail_city_consumer_competitive` and `external`.
    """
    
    # Convert 'Week_Start_Date' to datetime format in both DataFrames
    retail_city_consumer_competitive["Week_Start_Date"] = pd.to_datetime(retail_city_consumer_competitive["Week_Start_Date"])
    external["Week_Start_Date"] = pd.to_datetime(external["Week_Start_Date"])

    # Merge the two DataFrames on 'Week_Start_Date' and 'City_ID'
    final_df = retail_city_consumer_competitive.merge(external, on=['Week_Start_Date', 'City_ID'], how='left')

    return final_df

# Example usage:
# final_df = merge_with_external_data(retail_city_consumer_competitive, external)
# final_df.head()



def process_channel_scores(final_df):
    # Define the columns for normalization (Income, Age Group, Sentiment, and Sales)
    income_age_columns = [
        'Income_Bracket_Low', 'Income_Bracket_Upper Middle', 'Income_Bracket_High',
        'Age_Group_18-25_Share', 'Age_Group_26-35_Share', 'Age_Group_36-45_Share', 
        'Age_Group_46-55_Share', 'Age_Group_55+_Share'
    ]
    
    sentiment_columns = [
        'E Commerce_Amazon Solimo_sentiment_impact', 
        'E Commerce_Minute Maid_sentiment_impact', 
        'E Commerce_Real Fruit Juice_sentiment_impact',
        'General Trade_Minute Maid_sentiment_impact', 
        'General Trade_Paper Boat_sentiment_impact',
        'HoReCa_B Natural_sentiment_impact', 
        'HoReCa_Minute Maid_sentiment_impact', 
        'HoReCa_Paper Boat_sentiment_impact',
        'Modern Trade_B Natural_sentiment_impact', 
        'Modern Trade_Minute Maid_sentiment_impact',
        'Q Commerce_Minute Maid_sentiment_impact', 
        'Q Commerce_Tropicana_sentiment_impact'
    ]
    
    sales_columns = [
        'E Commerce_Total_Sales', 'General Trade_Total_Sales', 'HoReCa_Total_Sales',
        'Modern Trade_Total_Sales', 'Q Commerce_Total_Sales'
    ]
    
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Apply normalization to all relevant columns in one go
    final_df[income_age_columns + sentiment_columns + sales_columns] = scaler.fit_transform(
        final_df[income_age_columns + sentiment_columns + sales_columns]
    )
    
    # Calculate the income_age_score (weighted sum)
    final_df["income_age_score"] = (
        final_df['Income_Bracket_Low'] * 0.1 + 
        final_df['Income_Bracket_Upper Middle'] * 0.2 + 
        final_df['Income_Bracket_High'] * 0.3 + 
        final_df['Age_Group_18-25_Share'] * 0.2 + 
        final_df['Age_Group_26-35_Share'] * 0.2 + 
        final_df['Age_Group_36-45_Share'] * 0.1 + 
        final_df['Age_Group_46-55_Share'] * 0.1 + 
        final_df['Age_Group_55+_Share'] * 0.1
    )

    # Calculate the channel scores in one go
    final_df["E-commerce_score"] = (
        final_df['Channel_Score_E-commerce'] * 0.6 + 
        final_df[['E Commerce_Amazon Solimo_sentiment_impact', 
                  'E Commerce_Minute Maid_sentiment_impact', 
                  'E Commerce_Real Fruit Juice_sentiment_impact']].mean(axis=1) * 0.4
    )

    final_df["General_Trade_score"] = (
        final_df['Channel_Score_General_Trade'] * 0.6 + 
        final_df[['General Trade_Minute Maid_sentiment_impact', 
                  'General Trade_Paper Boat_sentiment_impact']].mean(axis=1) * 0.4
    )

    final_df["HoReCa_score"] = (
        final_df['Channel_Score_HoReCa'] * 0.6 + 
        final_df[['HoReCa_B Natural_sentiment_impact', 
                  'HoReCa_Minute Maid_sentiment_impact', 
                  'HoReCa_Paper Boat_sentiment_impact']].mean(axis=1) * 0.4
    )

    final_df["Modern_Trade_score"] = (
        final_df['Channel_Score_Modern_Trade'] * 0.6 + 
        final_df[['Modern Trade_B Natural_sentiment_impact', 
                  'Modern Trade_Minute Maid_sentiment_impact']].mean(axis=1) * 0.4
    )

    final_df["Q-Commerce_score"] = (
        final_df['Channel_Score_Q-commerce'] * 0.6 + 
        final_df[['Q Commerce_Minute Maid_sentiment_impact', 
                  'Q Commerce_Tropicana_sentiment_impact']].mean(axis=1) * 0.4
    )
    
    # Drop unnecessary columns in a single step
    final_df.drop(columns=[
        'Channel_Score_E-commerce', 'Channel_Score_General_Trade', 'Channel_Score_HoReCa', 
        'Channel_Score_Modern_Trade', 'Channel_Score_Q-commerce', 'E Commerce_Amazon Solimo_sentiment_impact',
        'E Commerce_Minute Maid_sentiment_impact', 'E Commerce_Real Fruit Juice_sentiment_impact',
        'General Trade_Minute Maid_sentiment_impact', 'General Trade_Paper Boat_sentiment_impact',
        'HoReCa_B Natural_sentiment_impact', 'HoReCa_Minute Maid_sentiment_impact', 
        'HoReCa_Paper Boat_sentiment_impact', 'Modern Trade_B Natural_sentiment_impact',
        'Modern Trade_Minute Maid_sentiment_impact', 'Q Commerce_Minute Maid_sentiment_impact',
        'Q Commerce_Tropicana_sentiment_impact'
    ], inplace=True)
    
    # Calculate the combined scores for each channel
    final_df['Combined_E-commerce_Channel_Score'] = (
        0.4 * final_df['E-commerce_score'] + 
        0.3 * final_df['income_age_score'] + 
        0.3 * final_df['E Commerce_Total_Sales']
    )

    final_df['Combined_General_Trade_Score'] = (
        0.4 * final_df['General_Trade_score'] + 
        0.3 * final_df['income_age_score'] + 
        0.3 * final_df['General Trade_Total_Sales']
    )

    final_df['Combined_HoReCa_Score'] = (
        0.4 * final_df['HoReCa_score'] + 
        0.3 * final_df['income_age_score'] + 
        0.3 * final_df['HoReCa_Total_Sales']
    )

    final_df['Combined_Modern_Trade_Score'] = (
        0.4 * final_df['Modern_Trade_score'] + 
        0.3 * final_df['income_age_score'] + 
        0.3 * final_df['Modern Trade_Total_Sales']
    )

    final_df['Combined_Q_Commerce_Score'] = (
        0.4 * final_df['Q-Commerce_score'] + 
        0.3 * final_df['income_age_score'] + 
        0.3 * final_df['Q Commerce_Total_Sales']
    )
    
    # Determine the optimal channel by selecting the one with the highest combined score
    channel_columns = [
        "Combined_E-commerce_Channel_Score", 
        "Combined_General_Trade_Score", 
        "Combined_HoReCa_Score", 
        "Combined_Modern_Trade_Score", 
        "Combined_Q_Commerce_Score"
    ]
    
    final_df["Optimal_Channel"] = final_df[channel_columns].idxmax(axis=1).str.replace("Combined_", "").str.replace("_Score", "")
    
    return final_df

# Upload processed CSV to ADLS
def upload_file_to_adls(service_client, container_name, local_file_path, remote_file_name):
    try:
        file_system_client = service_client.get_file_system_client(file_system=container_name)
        file_client = file_system_client.get_file_client(remote_file_name)
        with open(local_file_path, "rb") as data:
            file_client.upload_data(data, overwrite=True)
        print(f"File '{local_file_path}' uploaded to ADLS as '{remote_file_name}'.")
    except Exception as e:
        print(f"Error uploading file to ADLS: {e}")

# Main data processing pipeline
if __name__ == "__main__":
    # Unpack loaded datasets
    city = data.get("city.csv")
    retail = data.get("retailsalesdistribution.csv")
    products = data.get("products.csv")
    consumer = data.get("consumerbehavior.csv")
    competitive = data.get("competitivelandscape.csv")
    external = data.get("externalmarketinfluencer.csv")

    # Step 1: Weekly retail sales
    weekly_retail_sales = process_weekly_retail_sales(retail)

    # Step 2: Merge city data
    retail_city = process_and_merge_city_data(city, weekly_retail_sales)

    # Step 3: Consumer data
    consumer_df = process_and_merge_consumer_data(consumer)

    # Step 4: Channel scores
    retail_city_consumer = process_and_merge_channel_scores(retail_city, consumer_df)

    # Step 5: Competitive data
    retail_city_consumer_competitive = process_and_merge_competitive_data(retail_city_consumer, competitive)

    # Step 6: External data
    final_df = merge_with_external_data(retail_city_consumer_competitive, external)

    # Step 7: Final channel scores
    final_df = process_channel_scores(final_df)

    # Save processed data to CSV locally
    final_df.to_csv("processed_data.csv", index=False)



    # Upload processed_data.csv to ADLS
    upload_file_to_adls(service_client, container_name, "processed_data.csv", "processed_data.csv")
    print("Data processing complete. Output saved locally and uploaded to ADLS.")
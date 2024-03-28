import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
# Define a dictionary to map cluster_name values to cluster_topic values
cluster_topic_mapping = {
    "Leaflet Affair" : ['leaflet_affair'],
    "Scholz decision making": ["SPD", "Die_Grünen", "FDP"], # RECHECK
    "migration": ["migration"],
    "Foreign policy": ["foreign_policy"],
    "SPD": ["SPD"],
    "housing crisis and heating law": ["housing"],
    "AfD": ["AfD"],
    "CDU":["CDU"],
    "Ukraine":["Ukraine"],
    "right-wing extremism":["right_wing_extremism"],
    "Grüne": ["Die_Grünen"],
    "climate": ["climate"],
    "Merz": ["CDU"],
    "asylum": ["migration"],
    "Refugees": ['migration'],
    "Faeser": ["SPD"],
    "border control": ["border_control"],
    "CSU":["CDU"],
    "Söder": ["CDU"],
    "Merz on asylum": ["CDU", "migration"],
    "Chrupalla": ["AfD"],
    "Ampel": ["SPD", "Die_Grünen", "FDP"],
    "FDP": ["FDP"]
}
def fix_headline_df(headline_df):
  """
    Fixes the formatting of the headline DataFrame by properly splitting the 'date' column into 'date' and 'headline'
    columns. Removes extra quotation marks and replaces NaN headline entries with properly formatted data.

    Args:
        headline_df (DataFrame): The DataFrame containing headlines.

    Returns:
        DataFrame: The fixed DataFrame with correctly formatted 'date' and 'headline' columns.
    """

  papers = headline_df['paper'].unique()
  papers = [x for x in papers if not pd.isna(x)]
  papers
  # Iterate over rows
  for index, row in headline_df.iterrows():

      if pd.isnull(row['headline']):  # Check if headline is NaN

        parts = row['date'].split(",")  # Split headline by comma
        merged = parts[1]
        for i in range(2, len(parts)):
          part = str(parts[i])
          if part not in papers:

            merged = merged + "," + part  # Merge second and third elements
          else:
            break
        merged = merged.replace('""', '"')
        if merged[0] == '"':
          merged = merged[1:]
        if merged[-1] == '"':
          merged = merged[:-1]
        correct_split = []
        correct_split.append(parts[0])
        correct_split.append(merged)
        correct_split.extend(parts[i:])
        headline_df.iloc[index] = correct_split  # Replace the row with properly split data
  return headline_df

def rename_columns(clustering_and_sentiment_df):
    clustering_and_sentiment_df = clustering_and_sentiment_df.rename(columns={'sentiment_right wing extremism': 'sentiment_right_wing_extremism'})
    clustering_and_sentiment_df = clustering_and_sentiment_df.rename(columns={'sentiment_leaflet affair': 'sentiment_leaflet_affair'})
    clustering_and_sentiment_df = clustering_and_sentiment_df.rename(columns={'sentiment_Border control': 'sentiment_border_control'})
    clustering_and_sentiment_df = clustering_and_sentiment_df.rename(columns={'sentiment_foreign policy': 'sentiment_foreign_policy'})
    clustering_and_sentiment_df = clustering_and_sentiment_df.rename(columns={'sentiment_Weapon delivery Taurus': 'sentiment_weapon_delivery_Taurus'})
    clustering_and_sentiment_df = clustering_and_sentiment_df.rename(columns={'sentiment_CDU_en': 'sentiment_CDU'})
    clustering_and_sentiment_df = clustering_and_sentiment_df.rename(columns={'sentiment_SPD_en': 'sentiment_SPD'})
    clustering_and_sentiment_df = clustering_and_sentiment_df.rename(columns={'sentiment_AfD_en': 'sentiment_AfD'})
    clustering_and_sentiment_df = clustering_and_sentiment_df.rename(columns={'sentiment_Linke_en': 'sentiment_Linke'})
    clustering_and_sentiment_df = clustering_and_sentiment_df.rename(columns={'sentiment_FDP_en': 'sentiment_FDP'})
    clustering_and_sentiment_df = clustering_and_sentiment_df.rename(columns={'sentiment_Die_Grünen_en': 'sentiment_Die_Grünen'})
    clustering_and_sentiment_df = clustering_and_sentiment_df.rename(columns={'sentiment_Freie_Wähler_en': 'sentiment_Freie_Wähler'})
    clustering_and_sentiment_df = clustering_and_sentiment_df.rename(columns={'Cluster Name': 'cluster_name'})
    clustering_and_sentiment_df = clustering_and_sentiment_df.rename(columns={'sentiment_Migration_en': 'sentiment_migration'})
    return clustering_and_sentiment_df

def sentiment_reassignment(clustering_and_sentiment_df): 
    """
    Reassigns sentiment values from the 'sentiment_whole' column to specific sentiment columns based on cluster names.

    Args:
        clustering_and_sentiment_df (DataFrame): The DataFrame containing sentiment analysis results.

    Returns:
        None
    """
    for index, row in clustering_and_sentiment_df.iterrows():
        # Check if all values in the "sentiment_" columns (except "sentiment_whole") are None
        if all(pd.isnull(row[column]) for column in clustering_and_sentiment_df.columns if column.startswith('sentiment_') and column != 'sentiment_whole'):
            # Assign the "sentiment_whole" value to the "sentiment_{Cluster Name}" column
            if row["cluster_name"] in cluster_topic_mapping:
                topics = cluster_topic_mapping[row["cluster_name"]]
                for topic in topics:
                    clustering_and_sentiment_df[f'sentiment_{topic}'][index]= row['sentiment_whole']

    clustering_and_sentiment_df

def get_paper_counts(clustering_and_sentiment_df):
    """
    Calculates the counts of headlines per paper and adds the total count.

    Args:
        clustering_and_sentiment_df (DataFrame): The DataFrame containing sentiment analysis results.

    Returns:
        DataFrame: The DataFrame with paper counts.
    """
    # Calculate paper counts
    paper_counts = clustering_and_sentiment_df.groupby('paper').size()
    paper_counts_df = paper_counts.to_frame(name='Count')

    # Calculate column totals
    column_totals = paper_counts_df.sum()
    totals_df = pd.DataFrame(column_totals).T
    totals_df.index = ['Total'] 
    paper_counts_with_totals = pd.concat([paper_counts_df, totals_df])
    return paper_counts_with_totals

def count_pos(series):
    return (series > 0).sum()

def count_neg(series):
    return (series < 0).sum()

def get_analysis_per_paper_and_topic(clustering_and_sentiment_df, count_function):
    """
    Calculates sentiment analysis results per paper and sentiment topic.

    Args:
        clustering_and_sentiment_df (DataFrame): The DataFrame containing sentiment analysis results.
        count_function (function): The function to use for counting sentiment values.

    Returns:
        DataFrame: The DataFrame with sentiment analysis results per paper and topic.
    """
    sentiment_columns = clustering_and_sentiment_df.filter(like='sentiment')
    sentiment_columns['paper'] = clustering_and_sentiment_df['paper']
    # Count positive and negative values for each sentiment column grouped by the "paper" column
    pos_neg_counts_grouped = sentiment_columns.groupby('paper').apply(count_function)
    column_sums = pos_neg_counts_grouped.sum(axis=0)
    df_cs = pd.DataFrame(columns = ["total"])
    df_cs["total"] = column_sums
    pos_neg_counts_grouped = pos_neg_counts_grouped.append(df_cs.T)
    # Add the sum as a new column into the dataframe
    row_sums = pos_neg_counts_grouped.iloc[:, :-1].sum(axis=1)
    pos_neg_counts_grouped['total (topic only)'] = row_sums
    return pos_neg_counts_grouped


def avg_pos(series):
    # Calculate the average of positive values
    positive_values = series[series > 0]
    if len(positive_values) == 0:
        return 0  # Avoid division by zero
    return positive_values.mean()

def avg_neg(series):
    # Calculate the average of negative values
    negative_values = series[series < 0]
    if len(negative_values) == 0:
        return 0  # Avoid division by zero
    return negative_values.mean()

def avg_sentiment(series):
    # Calculate the average sentiment
    return series.mean()
def get_analysis_per_paper_and_topic(clustering_and_sentiment_df, avg_function):
    """
    Calculates sentiment analysis results per paper and sentiment topic.

    Args:
        clustering_and_sentiment_df (DataFrame): The DataFrame containing sentiment analysis results.
        avg_function (function): The function to use for calculating average sentiment values.

    Returns:
        DataFrame: The DataFrame with sentiment analysis results per paper and topic.
    """

    sentiment_columns = clustering_and_sentiment_df.filter(like='sentiment')
    sentiment_columns['paper'] = clustering_and_sentiment_df['paper']

    # Calculate average sentiment for each sentiment column grouped by the "paper" column
    avg_sentiment_grouped = sentiment_columns.groupby('paper').apply(avg_function)

    column_sums = avg_sentiment_grouped.mean(axis=0)
    df_cs = pd.DataFrame(columns=["total"])
    df_cs["total"] = column_sums

    avg_sentiment_grouped = avg_sentiment_grouped.append(df_cs.T)
    row_sums = avg_sentiment_grouped.iloc[:, :-1].mean(axis=1)


    avg_sentiment_grouped['total (topic only)'] = row_sums

    return avg_sentiment_grouped

def get_analysis_per_topic_and_time(df):
    """
    Calculates sentiment analysis results per sentiment topic over time.

    Args:
        df (DataFrame): The DataFrame containing sentiment analysis results with date information.

    Returns:
        DataFrame: The DataFrame with sentiment analysis results per topic over time.
    """

    sentiment_columns = [col for col in df.columns if col.startswith('sentiment_')]

    result_df = pd.DataFrame(index=df['date'].unique()) 

    # Calculate average sentiment for each topic over time
    for column in sentiment_columns:
        topic_name = column.split('_')[-1]  # Extract the topic name
        avg_sentiment_series = df.groupby('date')[column].mean()
        result_df[topic_name] = avg_sentiment_series

    result_df['total'] = result_df.mean(axis=1)

    print("Average sentiment over time for each topic:")
    return result_df


def plot_sentiment_over_time(df):
    fig = go.Figure()

    # Plot each topic separately
    for column in df.columns:
        if column != 'date':  
            fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name=column))

    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Average Sentiment',
        title='Average Sentiment Over Time'
    )

    fig.update_xaxes(tickangle=45)

    fig.show()

# Function to plot each column separately
def plot_individual_sentiment(df):
    for column in df.columns:
        if column != 'date':
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name=column))
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Average Sentiment',
                title=f'Average Sentiment Over Time - {column}',
                xaxis=dict(type='date')
            )
            fig.update_xaxes(tickangle=45)  
            fig.show()



def get_voting_prediction(clustering_and_sentiment_df, party_df, paper = None):
    """
    Calculates voting predictions based on sentiment analysis results and political party programs.

    Args:
        clustering_and_sentiment_df (DataFrame): The DataFrame containing sentiment analysis results.
        party_df (DataFrame): The DataFrame containing political party program data.
        paper (str, optional): The specific paper to consider for analysis. Defaults to None.

    Returns:
        DataFrame: The DataFrame containing voting predictions.
    """
    if paper is not None:
      clustering_and_sentiment_df = clustering_and_sentiment_df[clustering_and_sentiment_df['paper'] == paper]
    topic_counts = pd.DataFrame({f'{party_df.columns[i]}': 0 for i in range(1, len(party_df.columns))}, index=party_df['programme'])

    
    for headline_row in clustering_and_sentiment_df.itertuples():
        
        for party_row in party_df.itertuples():
            # Check if both sentiments are either above 0 or below 0 for each topic
            for i in range(3, len(clustering_and_sentiment_df.columns)):  # Start from 3 to skip 'date', 'headline', 'paper' columns
                topic_name = clustering_and_sentiment_df.columns[i]

                headline_sentiment = getattr(headline_row,topic_name)

                try:
                  party_sentiment = getattr(party_row,topic_name)
                #if it fails then this is the party sentiment which was not gathered for the programmes
                except:
                  continue
                if (headline_sentiment > 0 and party_sentiment > 0) or (headline_sentiment < 0 and party_sentiment < 0):
                    # Increase the count of the associated topic for that party

                    topic_counts[topic_name][party_row.programme] += 1

    topic_counts_df = pd.DataFrame(topic_counts)

    row_sums = topic_counts_df.sum(axis=1)
    pos_counts = get_analysis_per_paper_and_topic(count_pos)

    # Add the sum as a new column into the dataframe
    topic_counts_df['number of topic votes'] = row_sums
    # calulate party counts
    topic_counts_df["party_counts"] = 0
    for party in topic_counts_df.index:
      for column in pos_counts.columns:
        if column == f"sentiment_{party}":
          if paper is None:
            topic_counts_df["party_counts"][party] = pos_counts[column]["total"]
          else:
            topic_counts_df["party_counts"][party] = pos_counts[column][paper]
    topic_counts_df['total votes'] = topic_counts_df['party_counts'] + topic_counts_df['number of topic votes']
    votes_sum = topic_counts_df['total votes'].sum()

    # calculate percentage
    topic_counts_df['percentage'] = round((topic_counts_df['total votes'] / votes_sum) * 100,2)

    return topic_counts_df